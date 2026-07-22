import base64
import time

import cv2
import numpy as np

from app.camera.frame_source import Frame
from app.config import VEHICLE_CROPS_DIR
from app.detection.plate_detector import PlateDetector
from app.ocr.plate_category import classify_plate_category
from app.ocr.plate_reader import PlateReader
from app.services import segment_store
from app.services.ocr_worker import OcrJob, OcrWorker
from app.services.plate_identity import PlateIdentity
from app.services.result_sink import DetectionResult, ResultSink
from app.tracking.vehicle_tracker import TrackedVehicle, VehicleTracker
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Only an accepted reading at least this confident is trusted enough to stop
# further OCR attempts on that track — a lower-confidence accept is still
# format-valid but shaky, so it's worth letting later frames try for a
# stronger read rather than locking in a mediocre one.
_STOP_OCR_CONFIDENCE_THRESHOLD = 0.95


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def _encode_jpeg(image: np.ndarray) -> str:
    # A data URL keeps this self-contained in the JSON the Detection Table
    # already polls — no separate image-serving endpoint/file storage needed
    # for what's a small plate-crop thumbnail.
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


class Pipeline:
    """Wires camera -> vehicle detection+tracking -> OCR -> result
    aggregation -> sink. No *separate* vehicle-detection stage: the plate
    model is a combined vehicle+plate weight, so VehicleTracker finds both
    in one pass and reports each tracked vehicle's plate_bbox (if any plate
    fell inside its box this frame) alongside it. Vehicle is the primary
    tracked identity — plate is an attribute of a tracked vehicle, not the
    other way around (see VehicleTracker's docstring for why).
    """

    def __init__(self, camera_id: str, sink: ResultSink):
        self.camera_id = camera_id
        self.sink = sink

        # One reference crop saved per newly-counted track_id, for manually
        # checking whether vehicle_count is real — see _process_vehicle.
        self._crops_dir = VEHICLE_CROPS_DIR / camera_id
        self._crops_dir.mkdir(parents=True, exist_ok=True)

        self.plate_detector = PlateDetector()
        self.tracker = VehicleTracker(self.plate_detector)
        self.ocr_reader = PlateReader()
        # OCR runs on its own thread — it's ~1-2s per call on CPU, and running
        # it inline on the main loop stalled frame consumption long enough to
        # overflow the frame queue and drop everything that arrived meanwhile.
        self.ocr_worker = OcrWorker(self.ocr_reader, self._on_ocr_result)
        # Folds a vehicle that VehicleTracker fragmented into more than one
        # track_id (see PlateIdentity's docstring) back into a single
        # identity once OCR confirms it's a repeat reading. `_duplicate_track_ids`
        # is subtracted from the tracker's raw count so "vehicles crossed"
        # doesn't over-count a fragmented vehicle as more than one. Keyed to
        # each duplicate's vehicle_type (not just a plain set of ids) so the
        # same correction can also be applied per-type, not just to the
        # overall total — see process()'s vehicle_count_by_type.
        self.plate_identity = PlateIdentity()
        self._duplicate_track_ids: dict[int, str] = {}
        # Once a track_id has one accepted reading confident enough (see
        # _STOP_OCR_CONFIDENCE_THRESHOLD), further OCR attempts on it are
        # just repeat work — same crop, same result — that only wastes CPU
        # and piles up identical rows in the Detection Table. A lower-
        # confidence accept doesn't count yet, so weaker reads keep getting
        # a chance to improve. Deliberately keyed on the raw
        # (pre-PlateIdentity) track_id, since that's what _process_vehicle
        # sees before any result comes back.
        self._accepted_track_ids: set[int] = set()
        # Bumped by reset_for_new_cycle — see DetectionResult.generation's
        # docstring for why this exists (DB upsert-key collision across a
        # static video's Play-button restarts). Stays 0 forever for a live
        # RTSP source, which never calls reset_for_new_cycle — correctly
        # harmless there, since a live source's track_ids never reset/reuse
        # numbers in the first place.
        self.cycle_generation = 0

        self._warmup()

    def reset_for_new_cycle(self) -> None:
        """Call at the start of each new source-loop cycle (a Play-button
        restart of the same video file) — every Play click starts
        completely fresh: track_ids restart from 1, vehicle_count resets to
        0. Not needed for a live RTSP source, which never restarts within
        one process's lifetime.

        Resets everything keyed by track_id, not just the tracker itself —
        VehicleTracker.reset_for_new_cycle makes track_id numbers reusable
        again (e.g. a fresh vehicle can become track_id 1), so any of this
        class's OWN state still referencing the old cycle's track_id 1
        would otherwise silently apply to the new cycle's unrelated
        vehicle.
        """
        self.cycle_generation += 1
        self.tracker.reset_for_new_cycle()
        self.plate_identity.reset()
        self._duplicate_track_ids.clear()
        self._accepted_track_ids.clear()

    def _warmup(self) -> None:
        # YOLO and PaddleOCR both pay a one-time cold-start cost (thread pool
        # init, graph compilation) on their first real inference call — often
        # several seconds. Paying it here, before frames start flowing, keeps
        # it from silently eating the start of a live/paced video stream.
        start = time.monotonic()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            self.tracker.track(dummy_frame, 0.0)
            self.ocr_reader.read(dummy_crop)
        except Exception:
            logger.exception("Model warmup failed (non-fatal, continuing)")
        logger.info("Model warmup took %.2fs", time.monotonic() - start)

    def process(self, frame: Frame) -> list[DetectionResult]:
        # video_time (file content-time) when available; falls back to
        # wall-clock timestamp for RTSP, where there's no content timeline
        # and real elapsed time ~= wall-clock time anyway.
        track_time = frame.video_time if frame.video_time is not None else frame.timestamp
        tracked = self.tracker.track(frame.image, track_time)
        store = segment_store.get(self.camera_id)
        if store is not None:
            # Corrected for any track_id(s) PlateIdentity has since folded
            # into an earlier one — see PlateIdentity's docstring for why
            # the tracker itself can't avoid minting these in the first place.
            corrected_count = self.tracker.total_vehicle_count - len(self._duplicate_track_ids)
            corrected_by_type = dict(self.tracker.count_by_type)
            for vtype in self._duplicate_track_ids.values():
                corrected_by_type[vtype] = corrected_by_type.get(vtype, 0) - 1
            store.set_vehicle_count(corrected_count, corrected_by_type)
        # Every detected vehicle is tracked now (VehicleTracker assigns a
        # track_id to every vehicle box regardless of whether a plate was
        # found inside it this frame) — no more separate "untracked vehicle,
        # display-only" branch.
        return [self._process_vehicle(frame, vehicle) for vehicle in tracked]

    def _on_ocr_result(
        self,
        track_id: int,
        frame_id: int,
        timestamp: float,
        vehicle_type: str,
        plate_category: str,
        plate_text: str | None,
        confidence: float | None,
        plate_crop: np.ndarray,
        vehicle_crop: np.ndarray | None,
        status: str,
    ) -> None:
        # Runs on the OcrWorker thread, once a result comes back — arrives
        # asynchronously relative to whichever frame triggered it, so it's
        # surfaced through a separate per-track store (polled by the
        # frontend's Detection Table) rather than retrofitted into a
        # frame-by-frame detections list that may already be served. Records
        # every OCR attempt (accepted, rejected, or nothing readable), not
        # just successful ones, so failures are visible for debugging rather
        # than silently disappearing.

        # A validated reading can be the same physical vehicle VehicleTracker
        # already assigned an earlier track_id to (see PlateIdentity) — fold
        # it back into that earlier identity instead of reporting/counting
        # it as a second vehicle. Only possible once text is validated
        # ("accepted"); rejected/no_text attempts have no reading to compare.
        if status == "accepted" and plate_text is not None:
            if confidence is not None and confidence >= _STOP_OCR_CONFIDENCE_THRESHOLD:
                self._accepted_track_ids.add(track_id)
            canonical_track_id, is_new_vehicle = self.plate_identity.resolve(track_id, plate_text)
            if not is_new_vehicle:
                self._duplicate_track_ids[track_id] = vehicle_type
            track_id = canonical_track_id

        store = segment_store.get(self.camera_id)
        if store is not None:
            store.record_ocr_attempt(
                track_id,
                vehicle_type,
                plate_category,
                plate_text,
                confidence,
                _encode_jpeg(plate_crop),
                _encode_jpeg(vehicle_crop) if vehicle_crop is not None and vehicle_crop.size > 0 else "",
                status,
            )

        # The regular per-frame emit() below always has plate=None (OCR
        # hasn't resolved yet at that point) — without this second emit, a
        # validated plate reading never reached the DB sink at all, only
        # the in-memory store above. Only "accepted" carries a plate value;
        # other statuses still emit so last_seen/vehicle_confidence in the
        # DB row keep advancing for a track that's still being read.
        self.sink.emit(
            DetectionResult(
                camera_id=self.camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                track_id=track_id,
                generation=self.cycle_generation,
                vehicle_type=vehicle_type,
                vehicle_bbox=None,
                plate_bbox=None,
                plate=plate_text if status == "accepted" else None,
                plate_category=plate_category,
                vehicle_confidence=None,
                plate_confidence=None,
                ocr_confidence=confidence if status == "accepted" else None,
            )
        )

    def _process_vehicle(self, frame: Frame, vehicle: TrackedVehicle) -> DetectionResult:
        # vehicle.bbox IS the tracked entity's own detection box now — unlike
        # the old plate-centric design, there's no "containing vehicle box
        # not found" case to handle, so this crop is unconditional.
        vehicle_crop = _crop(frame.image, vehicle.bbox)

        # is_new fires exactly once per track_id — the same event that
        # bumped VehicleTracker.total_vehicle_count — so this folder ends up
        # with exactly one image per counted vehicle: a manual sanity check
        # for whether vehicle_count matches what a human sees. generation is
        # in the filename because track_id numbers get reused across a
        # static video's Play-button restarts (see cycle_generation).
        if vehicle.is_new and vehicle_crop.size > 0:
            filename = f"gen{self.cycle_generation}_track{vehicle.track_id}_{vehicle.vehicle_type}.jpg"
            cv2.imwrite(str(self._crops_dir / filename), vehicle_crop)

        plate_category = None
        if vehicle.plate_bbox is not None:
            plate_crop = _crop(frame.image, vehicle.plate_bbox)
            # Plain color analysis of the crop, independent of OCR/text —
            # computed here (once per tracked frame) and threaded into the
            # OcrJob below so the OCR thread doesn't need to redo it.
            plate_category = classify_plate_category(plate_crop)

            # No throttling gate before a track has a confirmed reading —
            # OCR is attempted on every frame a plate is found inside this
            # vehicle's box, as long as it isn't already mid-OCR for this
            # track (see OcrWorker.is_pending). Throttling traded away real
            # reads (a track's one gated attempt often landed on a
            # blurry/angled frame) for less CPU work; removed in favor of
            # just doing the work. But once a track_id has already produced
            # one accepted reading, every further attempt is just the same
            # crop yielding the same result — pure waste that also piles up
            # identical rows in the Detection Table — so that's still gated.
            if (
                vehicle.track_id not in self._accepted_track_ids
                and not self.ocr_worker.is_pending(vehicle.track_id)
            ):
                self.ocr_worker.submit(
                    OcrJob(
                        track_id=vehicle.track_id,
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        vehicle_type=vehicle.vehicle_type,
                        plate_category=plate_category,
                        plate_crop=plate_crop,
                        vehicle_crop=vehicle_crop,
                    )
                )

        result = DetectionResult(
            camera_id=self.camera_id,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            track_id=vehicle.track_id,
            generation=self.cycle_generation,
            vehicle_type=vehicle.vehicle_type,
            vehicle_bbox=vehicle.bbox,
            plate_bbox=vehicle.plate_bbox,
            plate=None,
            plate_category=plate_category,
            vehicle_confidence=vehicle.confidence,
            plate_confidence=vehicle.plate_confidence,
            ocr_confidence=None,
        )
        self.sink.emit(result)
        return result
