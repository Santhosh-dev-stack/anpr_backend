import queue

from app.camera.frame_source import Frame
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FrameQueue:
    """Bounded queue that drops the oldest frame when full instead of blocking
    the producer. Prevents unbounded lag growth when inference can't keep up
    with the incoming frame rate.
    """

    def __init__(self, maxsize: int):
        self._q: queue.Queue[Frame] = queue.Queue(maxsize=maxsize)
        self.dropped_frames = 0

    def put(self, frame: Frame) -> None:
        if self._q.full():
            try:
                self._q.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
        self._q.put_nowait(frame)

    def get(self, timeout: float | None = None) -> Frame | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._q.qsize()
