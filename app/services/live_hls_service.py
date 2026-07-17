import threading
import time
from pathlib import Path

from app.config import (
    FFMPEG_BINARY,
    HLS_LIVE_LIST_SIZE,
    HLS_SEGMENT_SECONDS,
    RTSP_RECONNECT_INITIAL_DELAY,
    RTSP_RECONNECT_MAX_DELAY,
)
from app.services.hls_service_base import (
    COMMON_FFMPEG_ARGS,
    MANIFEST_POLL_INTERVAL,
    BaseHlsService,
    logger,
)

# How long the newest segment can go without advancing before it's treated
# as stalled and force-restarted — a hung ffmpeg (blocked on a stuck
# decode, not just a stuck socket read) doesn't always exit on its own, so
# process-exit detection alone isn't enough. 3x the segment interval gives
# normal encode jitter room without masking a real stall.
_STALENESS_MULTIPLIER = 3

# Bounds how long ffmpeg's own input read blocks on a stalled socket before
# erroring out, turning a silent hang into a process exit the watchdog can
# react to. Verified against the installed ffmpeg (4.4.2, Ubuntu 22.04):
# `-stimeout` is listed under `-h demuxer=rtsp` specifically; the newer
# generic `-rw_timeout` is not rtsp-specific on this build — check
# `ffmpeg -h demuxer=rtsp` again if the ffmpeg version changes.
_RTSP_INPUT_TIMEOUT_US = 10_000_000


class LiveHlsService(BaseHlsService):
    """Sliding-window HLS for a continuous RTSP source — no #EXT-X-ENDLIST,
    old segments physically deleted as they roll off (bounds disk usage for
    a camera that runs for days). Owned by a single long-lived watchdog
    thread (same "keep trying" philosophy as RTSPReader._run()): launch ->
    wait for first segment -> monitor for crash/staleness -> on any
    failure, backoff and relaunch, forever. Each relaunch bumps
    `generation` (ffmpeg's segment numbering resets to 0 every relaunch,
    since the output dir is wiped each time) so callers can detect the
    discontinuity instead of assuming continuous playback.
    """

    def __init__(self, camera_id: str, source: str):
        super().__init__(camera_id, source)
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def _build_cmd(self, out_dir: Path) -> list[str]:
        return (
            [
                FFMPEG_BINARY,
                "-y",
                # -rtsp_transport tcp avoids UDP packet loss/corruption;
                # -stimeout bounds a stalled socket read (see module
                # docstring above).
                "-rtsp_transport",
                "tcp",
                "-stimeout",
                str(_RTSP_INPUT_TIMEOUT_US),
                "-i",
                self._source,
            ]
            + COMMON_FFMPEG_ARGS
            + [
                # No hls_playlist_type vod here — a live source never ends,
                # so a growing/never-deleted playlist would run forever. A
                # sliding window (list_size + delete_segments) both keeps
                # the manifest small and bounds disk usage.
                "-hls_list_size",
                str(HLS_LIVE_LIST_SIZE),
                "-hls_flags",
                "delete_segments",
                "-hls_segment_filename",
                str(out_dir / "seg_%05d.ts"),
                str(out_dir / "index.m3u8"),
            ]
        )

    def _spawn_background_work(self) -> None:
        # Incremented synchronously here, before the watchdog thread even
        # starts, so `generation` is deterministically at its first-launch
        # value by the time start() returns — a caller reading it right
        # after start() (see app/live/main.py) must never race against the
        # background thread's own first increment.
        self._generation += 1
        self._background_thread = threading.Thread(target=self._run_live, daemon=True)
        self._background_thread.start()

    def _run_live(self) -> None:
        delay = RTSP_RECONNECT_INITIAL_DELAY
        first_pass = True
        while not self._stopped.is_set():
            if not first_pass:
                # First launch's generation was already assigned
                # synchronously in _spawn_background_work(), before this
                # thread began — only a genuine restart bumps it here.
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

            self._notify_manifest_ready(f" (gen {self._generation})")
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
            if self._stopped.wait(MANIFEST_POLL_INTERVAL):
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
