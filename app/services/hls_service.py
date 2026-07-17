import shutil
import subprocess
import threading
import time
from pathlib import Path

from app.camera.frame_source import is_rtsp_source
from app.config import (
    FFMPEG_BINARY,
    HLS_LIVE_LIST_SIZE,
    HLS_OUTPUT_DIR,
    HLS_SEGMENT_SECONDS,
    RTSP_RECONNECT_INITIAL_DELAY,
    RTSP_RECONNECT_MAX_DELAY,
)
from app.services import segment_store
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ffmpeg has to open the input, init the x264 encoder, and buffer
# HLS_SEGMENT_SECONDS worth of frames before index.m3u8 references its
# first segment. hls.js requesting the manifest before that write happens
# gets a 404 it doesn't always recover from (see HLS manifest race) — this
# poll lets the frontend wait for an actual readiness signal instead of
# assuming start() returning means the stream is loadable.
_MANIFEST_POLL_INTERVAL = 0.1
_MANIFEST_WAIT_TIMEOUT = 20.0

# RTSP (live) only — how long the newest segment can go without advancing
# before treating ffmpeg as stalled and restarting it. A hung ffmpeg
# (blocked on a stuck decode, not just a stuck socket read) doesn't always
# exit on its own, so process-exit detection alone isn't enough. 3x the
# segment interval gives normal encode jitter room without masking a real
# stall.
_STALENESS_MULTIPLIER = 3

# RTSP (live) only — bounds how long ffmpeg's own input read blocks on a
# stalled socket before erroring out, turning a silent hang into a process
# exit the watchdog can react to. Verified against the installed ffmpeg
# (4.4.2, Ubuntu 22.04): `-stimeout` is listed under `-h demuxer=rtsp`
# specifically; the newer generic `-rw_timeout` is not rtsp-specific on
# this build — check `ffmpeg -h demuxer=rtsp` again if the ffmpeg version
# changes.
_RTSP_INPUT_TIMEOUT_US = 10_000_000


class HlsService:
    """Transcodes the same source the detection pipeline reads into an HLS
    stream, served by the FastAPI app's StaticFiles mount at /hls.

    File sources get a VOD playlist (the source is finite — ffmpeg
    completes and writes #EXT-X-ENDLIST) with a single bounded, one-shot
    manifest-wait: if ffmpeg fails to start, that's a real, actionable
    startup error, not something to retry forever.

    RTSP (live) sources instead get a sliding-window live playlist and are
    owned by a single long-lived watchdog thread that never gives up:
    launch -> wait for first segment -> monitor for crash/staleness -> on
    any failure, backoff and relaunch — the same "keep trying" philosophy
    as RTSPReader's own reconnect loop, since a live camera dropping and
    recovering is normal, not exceptional. Each relaunch bumps `generation`
    (ffmpeg's segment numbering resets to 0 on every relaunch, since the
    output dir is wiped each time) so callers can detect the discontinuity
    instead of assuming continuous playback.
    """

    def __init__(self, camera_id: str, source: str):
        self._camera_id = camera_id
        self._source = source
        self._is_live = is_rtsp_source(source)
        self._process: subprocess.Popen | None = None
        self._out_dir: Path | None = None
        self._manifest_path: Path | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def start(self) -> None:
        self._stopped.clear()
        if self._is_live:
            # Incremented synchronously here, before the watchdog thread
            # even starts, so `generation` is deterministically at its
            # first-launch value by the time start() returns — a caller
            # reading it right after start() (see main.py) must never race
            # against the background thread's own first increment.
            self._generation += 1
            self._watchdog_thread = threading.Thread(target=self._run_live, daemon=True)
            self._watchdog_thread.start()
        else:
            self._launch()
            threading.Thread(target=self._wait_for_manifest_once, daemon=True).start()

    def _build_cmd(self, out_dir: Path) -> list[str]:
        cmd = [FFMPEG_BINARY, "-y"]
        if self._is_live:
            # -rtsp_transport tcp avoids UDP packet loss/corruption; -stimeout
            # bounds a stalled socket read (see module docstring above).
            cmd += ["-rtsp_transport", "tcp", "-stimeout", str(_RTSP_INPUT_TIMEOUT_US)]
        cmd += [
            "-i",
            self._source,
            # Encoding cost scales with pixel count — downscaling to 960px
            # wide (~1/4 the pixels of a 1080p source) cuts x264 encode time
            # roughly 4x, which matters a lot when ffmpeg is competing with
            # the detection pipeline's YOLO/OCR inference for the same CPU.
            # This is only the browser preview stream; detection reads the
            # full-resolution source separately (see VideoReader), so this
            # doesn't affect detection accuracy. -2 keeps height even (a
            # libx264 requirement) while preserving aspect ratio.
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
        if self._is_live:
            # No hls_playlist_type vod here — a live source never ends, so
            # a growing/never-deleted playlist would run forever. A sliding
            # window (list_size + delete_segments) both keeps the manifest
            # small and bounds disk usage for a camera that runs for days.
            cmd += [
                "-hls_list_size",
                str(HLS_LIVE_LIST_SIZE),
                "-hls_flags",
                "delete_segments",
            ]
        else:
            cmd += ["-hls_playlist_type", "vod"]
        cmd += [
            "-hls_segment_filename",
            str(out_dir / "seg_%05d.ts"),
            str(out_dir / "index.m3u8"),
        ]
        return cmd

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

    def _wait_for_manifest_once(self) -> None:
        if self._wait_for_first_segment():
            store = segment_store.get(self._camera_id)
            if store is not None:
                store.mark_hls_manifest_ready()
            logger.info("HLS manifest ready for %s", self._camera_id)
        else:
            logger.warning(
                "Timed out after %.0fs waiting for HLS manifest for %s — ffmpeg may have failed to start",
                _MANIFEST_WAIT_TIMEOUT,
                self._camera_id,
            )

    def _wait_for_first_segment(self) -> bool:
        deadline = time.monotonic() + _MANIFEST_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            if self._manifest_has_segment():
                return True
            if self._stopped.wait(_MANIFEST_POLL_INTERVAL):
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

    def _run_live(self) -> None:
        delay = RTSP_RECONNECT_INITIAL_DELAY
        first_pass = True
        while not self._stopped.is_set():
            if not first_pass:
                # First launch's generation was already assigned
                # synchronously in start(), before this thread began —
                # only a genuine restart bumps it here.
                self._generation += 1
            first_pass = False
            self._launch()

            if not self._wait_for_first_segment():
                self._kill_process()
                if self._stopped.is_set():
                    return
                logger.warning(
                    "HLS manifest never appeared for %s (gen %d), retrying in %.1fs",
                    self._camera_id,
                    self._generation,
                    delay,
                )
                if self._stopped.wait(delay):
                    return
                delay = min(delay * 2, RTSP_RECONNECT_MAX_DELAY)
                continue

            store = segment_store.get(self._camera_id)
            if store is not None:
                store.mark_hls_manifest_ready()
            logger.info("HLS manifest ready for %s (gen %d)", self._camera_id, self._generation)
            delay = RTSP_RECONNECT_INITIAL_DELAY

            failure_reason = self._monitor_until_failure()
            self._kill_process()
            if self._stopped.is_set():
                return
            logger.warning(
                "HLS transcode for %s %s (gen %d) — restarting",
                self._camera_id,
                failure_reason,
                self._generation,
            )

    def _monitor_until_failure(self) -> str | None:
        # Two independent failure signals: a real process exit (crash), or
        # staleness (still alive but no new segment written in a while —
        # a stalled/hung ffmpeg; -stimeout bounds a stalled socket *read*
        # but not a stalled decode, so this catches what that doesn't).
        stale_after = _STALENESS_MULTIPLIER * HLS_SEGMENT_SECONDS
        last_mtime = self._newest_segment_mtime()
        last_change = time.monotonic()
        while not self._stopped.is_set():
            if self._process.poll() is not None:
                return "exited unexpectedly"
            mtime = self._newest_segment_mtime()
            now = time.monotonic()
            if mtime is not None and mtime != last_mtime:
                last_mtime = mtime
                last_change = now
            elif now - last_change > stale_after:
                return f"stalled (no new segment in {stale_after:.0f}s)"
            if self._stopped.wait(_MANIFEST_POLL_INTERVAL):
                return None
        return None

    def _newest_segment_mtime(self) -> float | None:
        try:
            segments = list(self._out_dir.glob("seg_*.ts"))
        except OSError:
            return None
        if not segments:
            return None
        return max(s.stat().st_mtime for s in segments)

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
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5)
        logger.info("Stopped HLS transcode for %s", self._camera_id)
