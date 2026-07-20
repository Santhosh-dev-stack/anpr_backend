import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.ocr.plate_reader import PlateReader
from app.ocr.plate_validator import is_standard_format, normalize_plate
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OcrJob:
    track_id: int
    frame_id: int
    timestamp: float
    vehicle_type: str
    # Computed once by the caller (from the same plate_crop) rather than
    # here, since it's plain color analysis independent of OCR — avoids
    # recomputing it on this thread too.
    plate_category: str
    plate_crop: np.ndarray
    # None when PlateTracker couldn't find a containing vehicle box for this
    # plate (see _find_containing_vehicle in plate_tracker.py) — carried
    # through purely for display (Detection Table), not used by OCR itself.
    vehicle_crop: np.ndarray | None = None


class OcrWorker:
    """Runs OCR on a dedicated background thread instead of inline on the
    main per-frame loop.

    Measured directly on real plate crops on this CPU: PaddleOCR's `.read()`
    call costs ~0.07-0.4s (avg ~0.12s) — cheaper than once assumed here, but
    still enough that running it inline stalls frame consumption on every
    OCR-eligible sighting, and multiple plates in one frame compound that.
    Moving it off the main thread lets tracking + plate detection keep
    consuming frames at their own pace while OCR catches up whenever it can,
    instead of the incoming (already-decimated) frame queue filling up and
    dropping frames during every OCR call.

    No throttling gate — OCR is attempted on every frame a plate is tracked
    (as long as it isn't already mid-OCR), not just first-sighting/periodic
    retry. Measured directly: gating traded away real reads (a track's one
    throttled attempt often landed on a blurry/angled frame) for less CPU
    work — removed in favor of just doing the work.

    A track can have at most one outstanding OCR job at a time (`_pending`)
    so a slow OCR call doesn't pile up duplicate jobs for the same track.
    """

    def __init__(
        self,
        ocr_reader: PlateReader,
        on_result: Callable[
            [int, int, float, str, str, str | None, float | None, np.ndarray, np.ndarray | None, str],
            None,
        ],
        maxsize: int = 12,
    ):
        self._ocr_reader = ocr_reader
        self._on_result = on_result
        self._queue: queue.Queue[OcrJob] = queue.Queue(maxsize=maxsize)
        self._pending: set[int] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_pending(self, track_id: int) -> bool:
        with self._lock:
            return track_id in self._pending

    def submit(self, job: OcrJob) -> None:
        with self._lock:
            if job.track_id in self._pending:
                return
            self._pending.add(job.track_id)
        # If OCR is already backlogged, drop the oldest queued job rather than
        # blocking the caller (the main pipeline thread) or growing unbounded
        # — a missed attempt just means the same track gets submitted again
        # on its next tracked frame (Pipeline._process_plate runs every frame).
        if self._queue.full():
            try:
                dropped = self._queue.get_nowait()
                with self._lock:
                    self._pending.discard(dropped.track_id)
                logger.info(
                    "OCR queue full (%d) — dropped track %d's queued job to make room for track %d",
                    self._queue.maxsize,
                    dropped.track_id,
                    job.track_id,
                )
            except queue.Empty:
                pass
        self._queue.put_nowait(job)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                result = self._ocr_reader.read(job.plate_crop)
                if result is None:
                    logger.info("OCR found no readable text for track %d", job.track_id)
                    self._on_result(
                        job.track_id, job.frame_id, job.timestamp, job.vehicle_type, job.plate_category,
                        None, None, job.plate_crop, job.vehicle_crop, "no_text",
                    )
                else:
                    normalized = normalize_plate(result.text)
                    # normalize_plate's wider format (1-2 RTO digits, 0-3
                    # series letters, 3-4 number digits) accepts real plate
                    # variance, but that same width lets a garbled OCR read
                    # coincidentally land on a completely different, valid-
                    # looking plate (e.g. a misread landing on a real state
                    # code like 'MZ' by chance) — format + state-code alone
                    # can't tell that apart from a genuine read. Gating the
                    # final accept on the strict standard shape trades away
                    # correctly reading real series-less plates (a real but
                    # rarer case) for rejecting those coincidental garbage
                    # matches (measured directly: this is what let a
                    # misread of a real TN plate through as an unrelated-
                    # looking 'MZ8J3333').
                    if normalized is not None and is_standard_format(normalized):
                        self._on_result(
                            job.track_id, job.frame_id, job.timestamp, job.vehicle_type, job.plate_category,
                            normalized, result.confidence, job.plate_crop, job.vehicle_crop, "accepted",
                        )
                    else:
                        logger.info(
                            "Rejected OCR read %r (conf=%.2f) for track %d (fails plate format check)",
                            result.text,
                            result.confidence,
                            job.track_id,
                        )
                        self._on_result(
                            job.track_id, job.frame_id, job.timestamp, job.vehicle_type, job.plate_category,
                            result.text, result.confidence, job.plate_crop, job.vehicle_crop, "rejected",
                        )
            except Exception:
                logger.exception("OCR worker failed on track %d", job.track_id)
            finally:
                with self._lock:
                    self._pending.discard(job.track_id)
