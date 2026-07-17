import math
from dataclasses import dataclass

import numpy as np

from app.config import PLATE_CLASS_NAME, PLATE_CONF_THRESHOLD, PLATE_DETECTION_IMGSZ
from app.detection.plate_detector import PlateDetector
from app.utils.logger import get_logger

logger = get_logger(__name__)

# How close (pixels, in full-frame coordinates) a new detection's center must
# be to a recently-seen plate's center to be treated as the same physical
# plate. Generous on purpose: consecutive *processed* frames are often
# 0.2-1s+ apart (frame decimation + CPU load), during which a moving plate
# can shift well beyond what IoU-based matching (e.g. ByteTrack) tolerates
# for a small box — that mismatch was silently dropping most real plate
# detections rather than just losing continuity on them.
_MATCH_RADIUS_PX = 150
# How many *seconds of real/video time* a plate can go unseen before its
# identity is forgotten and a new one is assigned on next sighting. Time-based
# rather than a processed-frame count: a frame-count threshold means a much
# shorter real-time tolerance when frame decimation is reduced/removed (e.g.
# running every frame instead of every 5th) since the same "N frames missed"
# then covers far fewer seconds — which paradoxically made *less* decimation
# fragment the same physical plate into more track_ids, not fewer. A
# real-seconds threshold stays consistent regardless of decimation rate.
_MAX_MISSED_SECONDS = 1.5


@dataclass
class TrackedPlate:
    track_id: int
    bbox: tuple[int, int, int, int]  # coordinates in the full frame
    confidence: float
    vehicle_type: str
    vehicle_bbox: tuple[int, int, int, int] | None  # None if no containing vehicle box found


@dataclass
class UntrackedVehicle:
    """A detected vehicle box with no plate matched inside it this frame —
    display-only (drawn on the preview), never OCR'd or given a track_id:
    this tracker's identity/continuity system is plate-centric (see
    PlateTracker's docstring), and a plateless box has no plate center to
    match frame-to-frame, so it isn't tracked at all."""

    bbox: tuple[int, int, int, int]
    confidence: float
    vehicle_type: str


class _Candidate:
    def __init__(self, track_id: int, center: tuple[float, float], seen_at: float):
        self.track_id = track_id
        self.center = center
        # px/second velocity estimate, updated from the actual displacement
        # each time this candidate matches a new detection. Zero until a
        # second sighting gives something to measure.
        self.velocity = (0.0, 0.0)
        self.seen_at = seen_at  # seconds (video_time for files, wall-clock for RTSP)

    def predicted_center(self, at_time: float) -> tuple[float, float]:
        elapsed = at_time - self.seen_at
        return (
            self.center[0] + self.velocity[0] * elapsed,
            self.center[1] + self.velocity[1] * elapsed,
        )


def _find_containing_vehicle(
    plate_center: tuple[float, float],
    vehicle_boxes: list[tuple[tuple[float, float, float, float], str, float]],
) -> tuple[str, tuple[int, int, int, int] | None]:
    # A plate's center can fall inside more than one vehicle box when
    # vehicles overlap (e.g. one parked behind another) — the smallest
    # (most specific/closest) containing box is the more plausible match,
    # not just the first one found.
    cx, cy = plate_center
    best_bbox: tuple[float, float, float, float] | None = None
    best_type = "unknown"
    best_area = float("inf")
    for (vx1, vy1, vx2, vy2), vehicle_type, area in vehicle_boxes:
        if vx1 <= cx <= vx2 and vy1 <= cy <= vy2 and area < best_area:
            best_bbox, best_type, best_area = (vx1, vy1, vx2, vy2), vehicle_type, area
    if best_bbox is None:
        return "unknown", None
    x1, y1, x2, y2 = best_bbox
    return best_type, (int(x1), int(y1), int(x2), int(y2))


class PlateTracker:
    """Detects plates directly on the full frame and assigns them a loose,
    best-effort identity across frames — no *separate* vehicle-detection
    stage; the current weight is a combined vehicle+plate model, so both are
    found in the same detection call, and each plate is matched to whichever
    detected vehicle box contains it (for `vehicle_type`/`vehicle_bbox`).

    Deliberately NOT built on Ultralytics' ByteTrack integration: measured on
    a real video, ByteTrack's IoU-based matching (whether tracking plate
    boxes or vehicle boxes) confirmed a track for only a small fraction of
    processed frames at this pipeline's frame decimation rate — see
    TRACKING_APPROACH_COMPARISON_REPORT.md for the full comparison. A
    confident detection should always produce a box and go to OCR; this
    simple nearest-center matcher provides continuity where it can (for OCR
    throttling / Detection Table grouping) without ever discarding a
    detection just because it couldn't be linked to a previous frame.

    This still occasionally fragments one physical plate into more than one
    track_id (if it goes undetected for longer than _MAX_MISSED_SECONDS) —
    that's reconciled downstream by app.services.plate_identity.PlateIdentity
    once OCR resolves a repeated plate text, not by this class (it has no
    OCR text to go on at tracking time).
    """

    def __init__(self, detector: PlateDetector):
        self._detector = detector
        self._next_id = 1
        self._candidates: list[_Candidate] = []
        # Every new track_id minted below counts one more vehicle "crossed" —
        # not reset per source-loop cycle (same as _next_id), so a Play-button
        # restart on the same video file keeps counting up rather than
        # resetting to zero, consistent with track identity already doing so.
        self.total_vehicle_count = 0
        # The weight is a combined vehicle+plate model (multiple classes) —
        # resolved once here so plate boxes can be told apart from vehicle
        # boxes in the single detection pass below (one model call finds
        # both, rather than a second call per vehicle crop).
        self._names = detector.model.names
        matches = [class_id for class_id, name in self._names.items() if name == PLATE_CLASS_NAME]
        if not matches:
            raise ValueError(
                f"PLATE_CLASS_NAME={PLATE_CLASS_NAME!r} not found in model classes: {self._names}"
            )
        self._plate_class_id = matches[0]

    def track(
        self, frame: np.ndarray, timestamp: float
    ) -> tuple[list[TrackedPlate], list[UntrackedVehicle]]:
        """`timestamp` should be real/video seconds (Frame.video_time for a
        video file, Frame.timestamp for RTSP/live where there's no content
        timeline) — used for the missed-frame tolerance and velocity
        prediction below, so both stay meaningful regardless of how many
        frames get decimated between calls.
        """
        results = self._detector.model.predict(
            frame, conf=PLATE_CONF_THRESHOLD, verbose=False, imgsz=PLATE_DETECTION_IMGSZ
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return [], []

        plate_boxes = []
        # (bbox, vehicle_type, area, confidence) — used both to look up each
        # plate's containing vehicle and to report ones with no plate inside.
        vehicle_boxes = []
        for box in boxes:
            class_id = int(box.cls.item())
            xyxy = box.xyxy[0].tolist()
            if class_id == self._plate_class_id:
                plate_boxes.append((xyxy, float(box.conf.item())))
            else:
                x1, y1, x2, y2 = xyxy
                vehicle_boxes.append(
                    ((x1, y1, x2, y2), self._names[class_id], (x2 - x1) * (y2 - y1), float(box.conf.item()))
                )

        if not plate_boxes:
            # No plate anywhere this frame — every vehicle box is untracked
            # (nothing to match a track_id against; see UntrackedVehicle).
            return [], [
                UntrackedVehicle(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    vehicle_type=vehicle_type,
                )
                for (x1, y1, x2, y2), vehicle_type, _area, conf in vehicle_boxes
            ]

        self._candidates = [
            c for c in self._candidates if timestamp - c.seen_at <= _MAX_MISSED_SECONDS
        ]

        tracked: list[TrackedPlate] = []
        matched_vehicle_boxes: set[tuple[float, float, float, float]] = set()
        for (x1, y1, x2, y2), plate_conf in plate_boxes:
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            candidate = self._match(center, timestamp)
            if candidate is None:
                candidate = _Candidate(self._next_id, center, timestamp)
                self._next_id += 1
                self.total_vehicle_count += 1
                self._candidates.append(candidate)
            else:
                elapsed = timestamp - candidate.seen_at
                if elapsed > 0:
                    observed_velocity = (
                        (center[0] - candidate.center[0]) / elapsed,
                        (center[1] - candidate.center[1]) / elapsed,
                    )
                    # Light smoothing (equal-weight average with the prior
                    # estimate) so one noisy detection doesn't swing the
                    # prediction wildly, while still adapting quickly to
                    # real speed/direction changes.
                    candidate.velocity = (
                        (candidate.velocity[0] + observed_velocity[0]) / 2,
                        (candidate.velocity[1] + observed_velocity[1]) / 2,
                    )
                candidate.center = center
                candidate.seen_at = timestamp

            vehicle_type, vehicle_bbox = _find_containing_vehicle(
                center, [(vb[0], vb[1], vb[2]) for vb in vehicle_boxes]
            )
            if vehicle_bbox is not None:
                matched_vehicle_boxes.add(vehicle_bbox)
            tracked.append(
                TrackedPlate(
                    track_id=candidate.track_id,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=plate_conf,
                    vehicle_type=vehicle_type,
                    vehicle_bbox=vehicle_bbox,
                )
            )

        untracked = [
            UntrackedVehicle(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=conf,
                vehicle_type=vehicle_type,
            )
            for (x1, y1, x2, y2), vehicle_type, _area, conf in vehicle_boxes
            if (int(x1), int(y1), int(x2), int(y2)) not in matched_vehicle_boxes
        ]
        return tracked, untracked

    def _match(self, center: tuple[float, float], timestamp: float) -> _Candidate | None:
        # Match against where the candidate is *predicted* to be now (last
        # known position + velocity * elapsed seconds), not where it was last
        # seen — a plate moving at a steady speed across a multi-frame gap
        # (decimation + CPU load) can easily end up outside a fixed radius
        # of its old position while still landing right at the prediction.
        best: _Candidate | None = None
        best_dist = _MATCH_RADIUS_PX
        for candidate in self._candidates:
            predicted = candidate.predicted_center(timestamp)
            dist = math.dist(predicted, center)
            if dist < best_dist:
                best_dist = dist
                best = candidate
        return best
