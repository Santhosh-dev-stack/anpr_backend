import base64
import time

import cv2
import numpy as np

from app.camera.frame_source import Frame
from app.detection.plate_detector import PlateDetector
from app.ocr.plate_category import classify_plate_category
from app.ocr.plate_reader import PlateReader
from app.services import segment_store
from app.services.ocr_worker import OcrJob, OcrWorker
from app.services.plate_identity import PlateIdentity
from app.services.result_sink import DetectionResult, ResultSink
from app.tracking.plate_tracker import PlateTracker, TrackedPlate
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
    """Wires camera -> plate detection+tracking -> OCR -> result aggregation
    -> sink. No *separate* vehicle-detection stage: the plate model is a
    combined vehicle+plate weight, so PlateTracker finds both in one pass
    and reports each plate's containing vehicle_type/vehicle_bbox alongside it.
    """

    def __init__(self, camera_id: str, sink: ResultSink):
        self.camera_id = camera_id
        self.sink = sink

        self.plate_detector = PlateDetector()
        self.tracker = PlateTracker(self.plate_detector)
        self.ocr_reader = PlateReader()
        # OCR runs on its own thread — it's ~1-2s per call on CPU, and running
        # it inline on the main loop stalled frame consumption long enough to
        # overflow the frame queue and drop everything that arrived meanwhile.
        self.ocr_worker = OcrWorker(self.ocr_reader, self._on_ocr_result)
        # Folds a plate that PlateTracker fragmented into more than one
        # track_id (see PlateIdentity's docstring) back into a single
        # identity once OCR confirms it's a repeat reading. `_duplicate_track_ids`
        # is subtracted from the tracker's raw count so "vehicles crossed"
        # doesn't over-count a fragmented plate as multiple vehicles.
        self.plate_identity = PlateIdentity()
        self._duplicate_track_ids: set[int] = set()
        # Once a track_id has one accepted reading confident enough (see
        # _STOP_OCR_CONFIDENCE_THRESHOLD), further OCR attempts on it are
        # just repeat work — same crop, same result — that only wastes CPU
        # and piles up identical rows in the Detection Table. A lower-
        # confidence accept doesn't count yet, so weaker reads keep getting
        # a chance to improve. Deliberately keyed on the raw
        # (pre-PlateIdentity) track_id, since that's what _process_plate
        # sees before any result comes back.
        self._accepted_track_ids: set[int] = set()

        self._warmup()

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
        tracked, untracked_vehicles = self.tracker.track(frame.image, track_time)
        store = segment_store.get(self.camera_id)
        if store is not None:
            # Corrected for any track_id(s) PlateIdentity has since folded
            # into an earlier one — see PlateIdentity's docstring for why
            # the tracker itself can't avoid minting these in the first place.
            corrected_count = self.tracker.total_vehicle_count - len(self._duplicate_track_ids)
            store.set_vehicle_count(corrected_count)
        results = [self._process_plate(frame, plate) for plate in tracked]
        # Display-only: a plateless vehicle box never reaches OCR or the
        # sink (no track_id to anchor continuity/dedup on, nothing to read),
        # it just needs to be drawn on the frontend's preview alongside the
        # plate-bearing ones.
        results.extend(
            DetectionResult(
                camera_id=self.camera_id,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                track_id=None,
                vehicle_type=vehicle.vehicle_type,
                vehicle_bbox=vehicle.bbox,
                plate_bbox=None,
                plate=None,
                plate_category=None,
                vehicle_confidence=vehicle.confidence,
                plate_confidence=None,
                ocr_confidence=None,
            )
            for vehicle in untracked_vehicles
        )
        return results

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

        # A validated reading can be the same physical plate PlateTracker
        # already assigned an earlier track_id to (see PlateIdentity) — fold
        # it back into that earlier identity instead of reporting/counting
        # it as a second vehicle. Only possible once text is validated
        # ("accepted"); rejected/no_text attempts have no reading to compare.
        if status == "accepted" and plate_text is not None:
            if confidence is not None and confidence >= _STOP_OCR_CONFIDENCE_THRESHOLD:
                self._accepted_track_ids.add(track_id)
            canonical_track_id, is_new_vehicle = self.plate_identity.resolve(track_id, plate_text)
            if not is_new_vehicle:
                self._duplicate_track_ids.add(track_id)
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

    def _process_plate(self, frame: Frame, plate: TrackedPlate) -> DetectionResult:
        plate_crop = _crop(frame.image, plate.bbox)
        # Plain color analysis of the crop, independent of OCR/text — computed
        # here (once per tracked frame) and threaded into the OcrJob below so
        # the OCR thread doesn't need to redo it.
        plate_category = classify_plate_category(plate_crop)

        # No throttling gate before a track has a confirmed reading — OCR is
        # attempted on every frame a plate is tracked, as long as it isn't
        # already mid-OCR for this track (see OcrWorker.is_pending).
        # Throttling traded away real reads (a track's one gated attempt
        # often landed on a blurry/angled frame) for less CPU work; removed
        # in favor of just doing the work. But once a track_id has already
        # produced one accepted reading, every further attempt is just the
        # same crop yielding the same result — pure waste that also piles
        # up identical rows in the Detection Table — so that's still gated.
        if (
            plate.track_id not in self._accepted_track_ids
            and not self.ocr_worker.is_pending(plate.track_id)
        ):
            vehicle_crop = _crop(frame.image, plate.vehicle_bbox) if plate.vehicle_bbox else None
            self.ocr_worker.submit(
                OcrJob(
                    track_id=plate.track_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    vehicle_type=plate.vehicle_type,
                    plate_category=plate_category,
                    plate_crop=plate_crop,
                    vehicle_crop=vehicle_crop,
                )
            )

        result = DetectionResult(
            camera_id=self.camera_id,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            track_id=plate.track_id,
            vehicle_type=plate.vehicle_type,
            vehicle_bbox=plate.vehicle_bbox,
            plate_bbox=plate.bbox,
            plate=None,
            plate_category=plate_category,
            vehicle_confidence=None,
            plate_confidence=plate.confidence,
            ocr_confidence=None,
        )
        self.sink.emit(result)
        return result
