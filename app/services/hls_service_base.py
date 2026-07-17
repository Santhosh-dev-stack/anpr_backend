import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import FFMPEG_BINARY, HLS_OUTPUT_DIR, HLS_SEGMENT_SECONDS
from app.services import segment_store
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ffmpeg has to open the input, init the x264 encoder, and buffer
# HLS_SEGMENT_SECONDS worth of frames before index.m3u8 references its
# first segment. hls.js requesting the manifest before that write happens
# gets a 404 it doesn't always recover from (see HLS manifest race) — this
# poll lets the frontend wait for an actual readiness signal instead of
# assuming start() returning means the stream is loadable.
MANIFEST_POLL_INTERVAL = 0.1
MANIFEST_WAIT_TIMEOUT = 20.0

# Shared by both flavors — encoding cost scales with pixel count;
# downscaling to 960px wide (~1/4 the pixels of a 1080p source) cuts x264
# encode time roughly 4x, which matters a lot when ffmpeg is competing with
# the detection pipeline's YOLO/OCR inference for the same CPU. This is
# only the browser preview stream; detection reads the full-resolution
# source separately (see VideoReader/RTSPReader), so this doesn't affect
# detection accuracy. -2 keeps height even (a libx264 requirement) while
# preserving aspect ratio.
COMMON_FFMPEG_ARGS = [
    "-vf",
    "scale=960:-2",
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-threads",
    "2",
    "-g",
    "50",
    "-sc_threshold",
    "0",
    "-an",
    "-f",
    "hls",
    "-hls_time",
    str(HLS_SEGMENT_SECONDS),
]


class BaseHlsService(ABC):
    """Transcodes the same source the detection pipeline reads into an HLS
    stream, served by the FastAPI app's StaticFiles mount at /hls.

    Shared launch/manifest-poll/kill mechanics live here; `StaticHlsService`
    and `LiveHlsService` (app/services/static_hls_service.py,
    live_hls_service.py) differ only in their ffmpeg flags and background
    lifecycle (one-shot wait vs. a long-lived watchdog).
    """

    def __init__(self, camera_id: str, source: str):
        self._camera_id = camera_id
        self._source = source
        self._process: subprocess.Popen | None = None
        self._out_dir: Path | None = None
        self._manifest_path: Path | None = None
        self._background_thread: threading.Thread | None = None
        self._stopped = threading.Event()

    @abstractmethod
    def _build_cmd(self, out_dir: Path) -> list[str]:
        ...

    @abstractmethod
    def _spawn_background_work(self) -> None:
        """Start whatever background thread this flavor needs (a one-shot
        manifest wait, or a long-lived watchdog) and assign it to
        self._background_thread so stop() can join it generically.
        """

    def start(self) -> None:
        self._stopped.clear()
        self._spawn_background_work()

    def _launch(self) -> None:
        out_dir = HLS_OUTPUT_DIR / self._camera_id
        # ffmpeg only overwrites segment filenames the new encode reuses —
        # leftover segments from a longer previous cycle/generation would
        # otherwise sit on disk forever, unreferenced by the fresh manifest
        # but never cleaned up.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._manifest_path = out_dir / "index.m3u8"
        cmd = self._build_cmd(out_dir)
        self._process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Started HLS transcode for %s -> %s", self._camera_id, out_dir)

    def _wait_for_first_segment(self) -> bool:
        deadline = time.monotonic() + MANIFEST_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            if self._manifest_has_segment():
                return True
            if self._stopped.wait(MANIFEST_POLL_INTERVAL):
                return False
        return False

    def _manifest_has_segment(self) -> bool:
        # A freshly created manifest is just the HLS header (#EXTM3U etc.)
        # until ffmpeg finishes encoding its first segment — only once a
        # .ts entry is actually referenced can the browser fetch anything.
        try:
            return ".ts" in self._manifest_path.read_text()
        except OSError:
            return False

    def _notify_manifest_ready(self, extra: str = "") -> None:
        store = segment_store.get(self._camera_id)
        if store is not None:
            store.mark_hls_manifest_ready()
        logger.info("HLS manifest ready for %s%s", self._camera_id, extra)

    def _kill_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def stop(self) -> None:
        self._stopped.set()
        self._kill_process()
        if self._background_thread is not None:
            self._background_thread.join(timeout=5)
        logger.info("Stopped HLS transcode for %s", self._camera_id)
