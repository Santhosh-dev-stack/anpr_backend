import queue
import threading

from app.services.result_sink import DetectionResult, ResultSink
from app.utils.logger import get_logger

logger = get_logger(__name__)

# DetectionResult is a small dataclass with no images, so a generous queue
# costs little memory — sized to absorb a burst of per-frame updates (see
# emit()'s docstring) without needing to drop anything under normal load.
_QUEUE_MAXSIZE = 500


class AsyncSink(ResultSink):
    """Wraps a slow sink (e.g. DbSink, which does a network round-trip per
    emit) so the main pipeline thread never blocks on it — mirrors
    OcrWorker's background-thread pattern (see ocr_worker.py).

    Measured directly: with DbSink's Postgres write happening inline on the
    main frame-processing thread, a single long-running session accumulated
    686,403 dropped frames (network round-trip latency to the DB stalling
    every frame that had a tracked plate, not just accepted OCR reads).
    Moving the write off-thread removes that stall entirely; a write that's
    merely queued a few hundred ms late is harmless, but a frame dropped
    because the pipeline was blocked on a network call is not recoverable.
    """

    def __init__(self, inner: ResultSink, maxsize: int = _QUEUE_MAXSIZE):
        self._inner = inner
        self._queue: queue.Queue[DetectionResult] = queue.Queue(maxsize=maxsize)
        # emit() is called from more than one producer thread (the main
        # pipeline thread via Pipeline._process_plate, and OcrWorker's own
        # thread via Pipeline._on_ocr_result) — without this lock, two
        # threads can each see the queue as full, each drop one item to
        # make room, and then both call put_nowait(), where the second one
        # still raises queue.Full because the first thread's put already
        # retook the freed slot. That crashed the whole process (uncaught
        # queue.Full propagating out of the main frame loop) the first time
        # this class ran against real concurrent traffic — this lock makes
        # the whole check-drop-put sequence atomic across producers.
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def emit(self, result: DetectionResult) -> None:
        # Drop the oldest queued write rather than block the caller (the
        # main pipeline thread) — under sustained backpressure a slightly
        # stale/missed persist is far cheaper than stalling frame
        # processing, which is exactly the failure mode this class exists
        # to eliminate.
        with self._lock:
            if self._queue.full():
                try:
                    dropped = self._queue.get_nowait()
                    logger.warning(
                        "AsyncSink queue full (%d) — dropped a queued write for track %s",
                        self._queue.maxsize,
                        dropped.track_id,
                    )
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(result)
            except queue.Full:
                # Should be unreachable now that the sequence above is
                # locked — guarded anyway so a result sink can never crash
                # the detection pipeline itself, no matter what.
                logger.warning("AsyncSink queue still full after dropping — discarding this write")

    def _run(self) -> None:
        while True:
            result = self._queue.get()
            try:
                self._inner.emit(result)
            except Exception:
                logger.exception("AsyncSink: inner sink emit failed for track %s", result.track_id)
