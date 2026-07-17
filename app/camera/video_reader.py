import time

import cv2

from app.camera.frame_source import Frame, FrameSource
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VideoReader(FrameSource):
    """Reads frames from a local video file, sequentially, at its native pace.

    Optionally decimates to a fixed target fps (e.g. 28fps source -> 10fps
    processed). This is deliberately a *fixed, uniform* stride, not the
    earlier load-adaptive frame-skipping that caused ByteTrack to lose
    moving vehicles' track IDs — that broke because skip size varied
    unpredictably with CPU load, sometimes spanning a large, inconsistent
    gap. A fixed 10fps stride means consecutive processed frames are always
    ~100ms apart, small enough for ByteTrack's IOU matching to still track
    normal traffic motion, while cutting the pipeline's workload by ~65%.
    """

    def __init__(self, path: str, target_fps: float | None = None):
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        self._frame_id = 0
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._start_time: float | None = None
        self._frame_stride = max(1, round(self.fps / target_fps)) if target_fps and self.fps else 1
        logger.info(
            "Opened video file %s (native fps=%.2f, processing stride=%d)",
            path,
            self.fps,
            self._frame_stride,
        )

    def read(self) -> Frame | None:
        while True:
            if not self._cap.isOpened():
                return None
            ok, image = self._cap.read()
            if not ok:
                return None
            self._frame_id += 1

            if self._frame_id % self._frame_stride != 0:
                continue  # decimated frame — decode cost is unavoidable, but skip pacing/processing

            # CAP_PROP_POS_MSEC reflects the decoder's actual PTS for the
            # frame just read, not a frame_id/fps estimate — stays correct
            # even with variable frame rate or the occasional dropped frame.
            video_time = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Decoding is CPU-bound and much faster than real playback, which
            # would otherwise blow through the whole file before inference
            # can keep up. Pacing to the frame's own PTS (not frame_id/fps,
            # which would drift once frames are decimated) makes this behave
            # like a live feed, so the queue only drops frames under genuine
            # CPU backpressure.
            if self.fps:
                if self._start_time is None:
                    self._start_time = time.monotonic()
                target_time = self._start_time + video_time
                delay = target_time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

            return Frame(image=image, frame_id=self._frame_id, timestamp=time.time(), video_time=video_time)

    @property
    def is_open(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        self._cap.release()
