from dataclasses import dataclass

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from app.config import (
    PLATE_CLASS_NAME,
    PLATE_CONF_THRESHOLD,
    PLATE_DETECTION_IMGSZ,
    VEHICLE_TRACK_MIN_CONFIDENCE,
)
from app.detection.plate_detector import PlateDetector
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedVehicle:
    track_id: int
    bbox: tuple[int, int, int, int]  # vehicle box, full-frame coordinates
    confidence: float  # vehicle detection confidence
    vehicle_type: str
    # None when no plate was detected inside this vehicle's box THIS frame —
    # the vehicle is still tracked/counted regardless; there's no more
    # "untracked vehicle" concept now that vehicle is the primary identity.
    plate_bbox: tuple[int, int, int, int] | None
    plate_confidence: float | None
    # True only on the frame where this track_id was first assigned (the
    # same event that bumps total_vehicle_count) — lets a caller do
    # once-per-vehicle work (e.g. saving a reference crop) without keeping
    # its own duplicate seen-track_ids bookkeeping.
    is_new: bool


# Overrides on top of Ultralytics' bundled bytetrack.yaml defaults, tuned
# for this pipeline's decimated frame rate (PROCESSING_FPS) rather than the
# near-native-fps video ByteTrack's defaults assume. All three address the
# same root cause: consecutive PROCESSED frames are much further apart in
# real time here than in a typical tracking benchmark, so a vehicle's box
# shifts further between them than the defaults expect.
_BYTETRACK_OVERRIDES = {
    # Default 30 (processed-frame count, not real time) — at this
    # pipeline's decimation, 30 processed frames can span several real
    # seconds. Raised to give a briefly-occluded/missed vehicle more real
    # time to reappear as the SAME track_id rather than fragmenting into a
    # new one. Trade-off: a lost track lingers longer before being dropped,
    # so it has more opportunity to (rarely) get wrongly re-matched to a
    # different vehicle that later passes through the same area.
    "track_buffer": 60,
    # Default 0.8 (max IoU-based cost tolerated for a valid match — higher
    # means looser/more forgiving). Raised slightly to tolerate the larger
    # inter-frame displacement from decimation. Trade-off: slightly higher
    # risk of merging two distinct nearby vehicles into one track in dense
    # traffic.
    "match_thresh": 0.9,
    # Default 0.25. Originally lowered to 0.20 to match this pipeline's raw
    # detection floor (PLATE_CONF_THRESHOLD) so no valid detection was
    # silently excluded from full first-stage tracking. Now moot in
    # practice — VEHICLE_TRACK_MIN_CONFIDENCE (0.50) filters out anything
    # below 0.50 before it ever reaches the tracker, so every box seen here
    # already clears this threshold either way. Left as-is rather than
    # raised to 0.50: harmless overlap, and keeps this override meaningful
    # again if the confidence gate above is ever loosened.
    "track_high_thresh": 0.20,
}


def _load_bytetrack_args() -> IterableSimpleNamespace:
    config = YAML.load(check_yaml("bytetrack.yaml"))
    config.update(_BYTETRACK_OVERRIDES)
    return IterableSimpleNamespace(**config)


class VehicleTracker:
    """Tracks VEHICLE boxes frame-to-frame using Ultralytics' BYTETracker,
    driven manually rather than via the high-level `model.track()` API —
    see the "Why not model.track()" section below, this isn't a style
    preference.

    Replaces the previous plate-centric nearest-center + velocity-prediction
    matcher (see git history / TRACKING_APPROACH_COMPARISON_REPORT.md for
    why ByteTrack was originally rejected — that evaluation was PLATE-level
    tracking only). A standalone test on real footage measured ByteTrack as
    noticeably more reliable for VEHICLE-sized boxes at this pipeline's
    frame decimation: many individual vehicle tracks held 90-100% frame
    continuity across multi-second spans, vs. a small fraction for
    plate-sized boxes — a vehicle's larger box tolerates the same
    inter-frame pixel displacement far better in IoU terms than a plate's
    tiny box does.

    Known, accepted limitation: still fragments some vehicles (particularly
    small/distant/briefly-visible ones) into more than one track_id —
    measured directly at ~1.7x over-count on a real test video (63 track_ids
    for ~35-40 actual vehicles). For a vehicle whose plate gets read,
    PlateIdentity reconciles this the same way it always has (matching OCR
    text folds a later fragment back into the earlier track_id). For a
    PLATELESS vehicle there is no OCR text to reconcile fragments against,
    so `total_vehicle_count` can still overcount those specifically — a
    real, deliberate trade-off made for an overall more accurate count, not
    an oversight.

    Why not `model.track()`: BYTETracker is one shared IoU/Kalman
    association pool across every class fed into a single `update()` call —
    class is stored on each STrack but never used to gate matching. Feeding
    plate AND vehicle boxes into one `model.track()` call risks a plate box
    IoU-matching against a lost/predicted VEHICLE track (their boxes overlap
    by construction, since a plate sits inside its vehicle's box), silently
    corrupting that vehicle's identity for a frame. Driving a dedicated
    `BYTETracker` instance with ONLY vehicle-class boxes avoids this
    entirely; plate geometry comes from the same underlying detection call
    but never touches the tracker's association state.

    Why one `BYTETracker` PER vehicle subclass (car/motorcycle/truck/...),
    not one shared across all of them: the same "class stored but never
    checked" gap applies one level deeper. A single shared tracker pools
    a car's lost track and a motorcycle's new detection into the same
    IoU/Kalman association space — nothing stops a lost car (still inside
    `track_buffer`'s window) from being re-matched to an unrelated
    motorcycle that happens to pass through the same screen region,
    silently relabeling that motorcycle under the car's track_id (observed
    directly: the same track_id reported as one vehicle_type in one frame's
    detections and a different vehicle_type in a later frame — the exact
    "same id, different vehicle" symptom). Splitting into one tracker per
    class makes this structurally impossible: a car can now only ever
    IoU-match against other cars. `BaseTrack._count` (the global id counter
    `next_id()`/`reset_id()` operate on) is a process-wide class attribute
    shared by every `BYTETracker` instance regardless of how many exist, so
    per-class trackers still hand out one non-colliding, monotonically
    increasing sequence of track_ids — no extra bookkeeping needed to keep
    them from overlapping, AS LONG AS every per-class BYTETracker is
    constructed before any of them starts tracking (see __init__: building
    one per class lazily, mid-stream, would re-zero the shared counter out
    from under classes that already handed out ids — BYTETracker.__init__
    itself calls reset_id()).
    """

    def __init__(self, detector: PlateDetector):
        self._detector = detector
        self._names = detector.model.names
        matches = [class_id for class_id, name in self._names.items() if name == PLATE_CLASS_NAME]
        if not matches:
            raise ValueError(
                f"PLATE_CLASS_NAME={PLATE_CLASS_NAME!r} not found in model classes: {self._names}"
            )
        self._plate_class_id = matches[0]
        # One BYTETracker per vehicle subclass — see class docstring's "Why
        # one BYTETracker PER vehicle subclass". All built up front, here,
        # rather than lazily on first sighting of each class: BYTETracker's
        # own __init__ calls reset_id(), which zeroes the *global* id
        # counter (BaseTrack._count, shared by every instance) — creating
        # one lazily mid-stream, after other classes' trackers already
        # handed out ids, would zero the counter out from under them and
        # immediately start colliding with ids already in use. Building all
        # of them here means every reset_id() call happens before any real
        # tracking starts.
        self._byte_trackers: dict[int, BYTETracker] = {
            class_id: BYTETracker(args=_load_bytetrack_args())
            for class_id in self._names
            if class_id != self._plate_class_id
        }
        # Every track_id BYTETracker has ever assigned, across this
        # tracker's whole lifetime — not reset per source-loop cycle (see
        # reset_for_new_cycle), same as the old tracker's _next_id/
        # total_vehicle_count semantics: a Play-button restart on the same
        # video file keeps counting up rather than resetting to zero.
        self._seen_track_ids: set[int] = set()
        self.total_vehicle_count = 0
        # Same "new track_id" event as total_vehicle_count, broken down by
        # vehicle_type — e.g. {"car": 12, "motorcycle": 8}. Only classes
        # actually seen so far appear as keys.
        self.count_by_type: dict[str, int] = {}

    def reset_for_new_cycle(self) -> None:
        """Call at the start of each new source-loop cycle (e.g. a
        Play-button restart of the same video file) — NOT needed for a live
        RTSP source, which never restarts within one process's lifetime.

        Full reset, by design: track_ids restart from 1 and
        `total_vehicle_count` resets to 0, so each Play click starts
        completely fresh rather than continuing to count across restarts.
        This also fixes the same underlying problem a partial reset would
        have needed to handle anyway — BYTETracker's `track_buffer` is a
        frame_id difference, not a time/position check like the old
        tracker's `_MAX_MISSED_SECONDS`, so frame_id would otherwise keep
        incrementing straight through a naive reuse of the same tracker
        instance across cycles, and a track lost in roughly the last
        track_buffer (30) processed frames of one playthrough could
        spuriously IoU-match a fresh detection at frame 0 of the next one.
        `BYTETracker.reset()` clears the pools and frame_id AND resets
        STrack's global id counter — exactly what's wanted here, since
        `_seen_track_ids`/`total_vehicle_count` are cleared in lockstep
        below rather than kept (which is what would have made reusing
        track_id 1, 2, 3... unsafe). Calling `.reset()` on every per-class
        tracker is safe even though the id counter it resets is shared
        process-wide — resetting the same global counter to 0 more than
        once is a no-op past the first call.

        Callers must also reset any of their OWN state keyed by track_id
        (Pipeline.reset_for_new_cycle does this for PlateIdentity/
        _accepted_track_ids/_duplicate_track_ids) — otherwise a reused
        track_id number could inherit stale bookkeeping from the previous
        cycle's unrelated vehicle.
        """
        for tracker in self._byte_trackers.values():
            tracker.reset()
        self._seen_track_ids.clear()
        self.total_vehicle_count = 0
        self.count_by_type.clear()

    def track(self, frame: np.ndarray, timestamp: float) -> list[TrackedVehicle]:
        # timestamp kept for signature parity with the call site (Pipeline
        # passes frame.video_time/timestamp) and the old tracker — this
        # tracker's own frame-to-frame matching doesn't consume it directly
        # (BYTETracker's tolerance is frame-count based, see
        # reset_for_new_cycle's docstring for the trade-off that implies).
        del timestamp

        results = self._detector.model.predict(
            frame, conf=PLATE_CONF_THRESHOLD, verbose=False, imgsz=PLATE_DETECTION_IMGSZ
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        is_plate = boxes.cls.int() == self._plate_class_id
        plate_boxes = boxes[is_plate]
        vehicle_boxes = boxes[~is_plate]
        if len(vehicle_boxes) == 0:
            return []

        # Vehicle-only confidence gate — a box below this never reaches any
        # tracker, so it can't get a track_id (new or continuing) and isn't
        # counted. Plate boxes are unaffected (their own floor is
        # PLATE_CONF_THRESHOLD, already applied above at the predict call).
        vehicle_boxes = vehicle_boxes[vehicle_boxes.conf >= VEHICLE_TRACK_MIN_CONFIDENCE]
        if len(vehicle_boxes) == 0:
            return []

        # Only vehicle-class boxes ever reach any tracker's association
        # state — see class docstring's "Why not model.track()". Further
        # split by vehicle subclass so a car can never IoU-match against a
        # motorcycle's box — see "Why one BYTETracker PER vehicle subclass".
        vehicle_cls = vehicle_boxes.cls.int()
        per_class_tracks = []
        for class_id in vehicle_cls.unique().tolist():
            class_boxes = vehicle_boxes[vehicle_cls == class_id]
            class_tracks = self._byte_trackers[class_id].update(class_boxes.cpu().numpy())
            if len(class_tracks):
                per_class_tracks.append(class_tracks)
        tracks = np.concatenate(per_class_tracks, axis=0) if per_class_tracks else np.empty((0, 8))
        if len(tracks) == 0:
            return []

        vehicle_entries = []  # (track_id, bbox, area, vehicle_type, confidence, is_new)
        for x1, y1, x2, y2, track_id, score, cls, _idx in tracks:
            track_id = int(track_id)
            vtype = self._names[int(cls)]
            is_new = track_id not in self._seen_track_ids
            if is_new:
                self._seen_track_ids.add(track_id)
                self.total_vehicle_count += 1
                self.count_by_type[vtype] = self.count_by_type.get(vtype, 0) + 1
            vehicle_entries.append(
                (track_id, (x1, y1, x2, y2), (x2 - x1) * (y2 - y1), vtype, float(score), is_new)
            )

        # For each plate, find the smallest containing vehicle box (same
        # containment-priority rule as the old _find_containing_vehicle,
        # inverted: a plate looks up its vehicle here, instead of a vehicle
        # looking up its plate) — keep the highest-confidence plate per
        # vehicle if more than one plate box lands inside the same vehicle.
        plate_for_vehicle: dict[int, tuple[tuple[int, int, int, int], float]] = {}
        for plate_box in plate_boxes:
            px1, py1, px2, py2 = plate_box.xyxy[0].tolist()
            pconf = float(plate_box.conf.item())
            center = ((px1 + px2) / 2, (py1 + py2) / 2)
            best_tid, best_area = None, float("inf")
            for tid, (vx1, vy1, vx2, vy2), varea, _, _, _ in vehicle_entries:
                if vx1 <= center[0] <= vx2 and vy1 <= center[1] <= vy2 and varea < best_area:
                    best_tid, best_area = tid, varea
            if best_tid is not None:
                existing = plate_for_vehicle.get(best_tid)
                if existing is None or pconf > existing[1]:
                    plate_for_vehicle[best_tid] = ((int(px1), int(py1), int(px2), int(py2)), pconf)

        tracked: list[TrackedVehicle] = []
        for tid, (x1, y1, x2, y2), _area, vtype, vconf, is_new in vehicle_entries:
            plate_bbox, plate_conf = plate_for_vehicle.get(tid, (None, None))
            tracked.append(
                TrackedVehicle(
                    track_id=tid,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=vconf,
                    vehicle_type=vtype,
                    plate_bbox=plate_bbox,
                    plate_confidence=plate_conf,
                    is_new=is_new,
                )
            )
        return tracked
