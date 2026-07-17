import threading
from pathlib import Path

from app.config import FFMPEG_BINARY
from app.services.hls_service_base import (
    COMMON_FFMPEG_ARGS,
    MANIFEST_WAIT_TIMEOUT,
    BaseHlsService,
    logger,
)


class StaticHlsService(BaseHlsService):
    """VOD playlist for a finite video file — ffmpeg completes and writes
    #EXT-X-ENDLIST. A single bounded, one-shot manifest wait: if ffmpeg
    fails to start, that's a real, actionable startup error, not something
    to retry forever (contrast LiveHlsService's watchdog, which never
    gives up).
    """

    def _build_cmd(self, out_dir: Path) -> list[str]:
        return (
            [FFMPEG_BINARY, "-y", "-i", self._source]
            + COMMON_FFMPEG_ARGS
            + [
                "-hls_playlist_type",
                "vod",
                "-hls_segment_filename",
                str(out_dir / "seg_%05d.ts"),
                str(out_dir / "index.m3u8"),
            ]
        )

    def _spawn_background_work(self) -> None:
        self._launch()
        self._background_thread = threading.Thread(target=self._wait_once, daemon=True)
        self._background_thread.start()

    def _wait_once(self) -> None:
        if self._wait_for_first_segment():
            self._notify_manifest_ready()
        else:
            logger.warning(
                "Timed out after %.0fs waiting for HLS manifest for %s — ffmpeg may have failed to start",
                MANIFEST_WAIT_TIMEOUT,
                self._camera_id,
            )
