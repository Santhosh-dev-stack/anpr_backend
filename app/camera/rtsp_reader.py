import re
import threading
import time

import cv2

from app.camera.frame_source import Frame, FrameSource
from app.config import RTSP_RECONNECT_INITIAL_DELAY, RTSP_RECONNECT_MAX_DELAY
from app.utils.logger import get_logger

logger = get_logger(__name__)

# RTSP URLs commonly embed the camera's login credentials
# (rtsp://user:password@host/...) — logged messages must not leak that
# password in plaintext into run.log.
_CREDENTIALS_PATTERN = re.compile(r"//[^:@/]+:[^@/]+@")


def _redact_url(url: str) -> str:
    return _CREDENTIALS_PATTERN.sub("//<redacted>@", url)


class RTSPReader(FrameSource):
    """Reads frames from an RTSP stream in a dedicated thread.

    Keeps only the latest decoded frame (drops older ones) so pipeline
    consumption speed never blocks or lags behind live capture. Reconnects
    with exponential backoff on read failure/disconnect.
    """

    def __init__(self, url: str):
        self._url = url
        self._safe_url = _redact_url(url)  # for logging only — never the real credentials
        self._frame_id = 0
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._healthy = False
        self._stopped = False
        self._cap: cv2.VideoCapture | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _connect(self) -> bool:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self._url)
        return self._cap.isOpened()

    def _run(self) -> None:
        delay = RTSP_RECONNECT_INITIAL_DELAY
        while not self._stopped:
            if self._cap is None or not self._cap.isOpened():
                if not self._connect():
                    logger.warning(
                        "Failed to connect to RTSP %s, retrying in %.1fs", self._safe_url, delay
                    )
                    self._healthy = False
                    time.sleep(delay)
                    delay = min(delay * 2, RTSP_RECONNECT_MAX_DELAY)
                    continue
                logger.info("Connected to RTSP %s", self._safe_url)
                delay = RTSP_RECONNECT_INITIAL_DELAY
                self._healthy = True

            ok, image = self._cap.read()
            if not ok:
                logger.warning("Lost RTSP stream %s, reconnecting", self._safe_url)
                self._healthy = False
                self._cap.release()
                self._cap = None
                continue

            self._frame_id += 1
            frame = Frame(image=image, frame_id=self._frame_id, timestamp=time.time())
            with self._lock:
                self._latest = frame

    def read(self) -> Frame | None:
        with self._lock:
            frame = self._latest
            self._latest = None
        return frame

    @property
    def is_open(self) -> bool:
        return not self._stopped

    @property
    def healthy(self) -> bool:
        return self._healthy

    def release(self) -> None:
        self._stopped = True
        self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
