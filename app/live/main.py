import argparse
import threading
import time

from app.camera.frame_source import Frame, is_rtsp_source
from app.camera.rtsp_reader import RTSPReader
from app.config import FRAME_QUEUE_MAXSIZE, HLS_SEGMENT_SECONDS
from app.pipeline_runner import bootstrap, reader_loop
from app.services import segment_store
from app.services.frame_queue import FrameQueue
from app.services.live_hls_service import LiveHlsService
from app.services.pipeline import Pipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _run_live_cycle(source: str, camera_id: str, pipeline: Pipeline) -> None:
    """Runs the RTSP reader+HLS+pipeline loop for the process's lifetime —
    RTSPReader reconnects internally, so unlike the static/file case there
    is no "cycle end" to loop back around on; this runs once, until
    interrupted.
    """
    reader = RTSPReader(source)
    frame_queue = FrameQueue(maxsize=FRAME_QUEUE_MAXSIZE)

    # Registered before hls_service.start() — its manifest-readiness poll
    # calls segment_store.get(camera_id), which would silently no-op
    # (no store, no exception) if the store weren't already registered.
    store = segment_store.SegmentStore.for_live()
    segment_store.register(camera_id, store)

    hls_service = LiveHlsService(camera_id=camera_id, source=source)
    hls_service.start()
    # Tracked so a later ffmpeg watchdog restart (crash or stall — see
    # LiveHlsService) can be detected and reacted to below: its segment
    # numbering resets to 0 on relaunch, so the wall-clock epoch and
    # detection store must reset in lockstep with it, not silently drift.
    last_seen_hls_generation = hls_service.generation

    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=reader_loop, args=(reader, frame_queue, stop_event), daemon=True
    )
    reader_thread.start()

    current_segment_index = -1
    last_video_time = 0.0
    segment_durations: list[float] = []
    segment_wall_start = time.monotonic()
    cycle_start_wall = time.monotonic()

    try:
        while not stop_event.is_set() or frame_queue.qsize() > 0:
            frame: Frame | None = frame_queue.get(timeout=1.0)
            # Placed immediately after get(), before the None-check below,
            # so this keeps updating (at ~1s cadence, the get() timeout)
            # even while frames stop arriving during a real camera outage
            # — that's the exact case this is meant to surface, so it must
            # not get stuck reporting the last-known-good value.
            store.set_camera_healthy(reader.healthy)

            if hls_service.generation != last_seen_hls_generation:
                # ffmpeg was restarted by LiveHlsService's watchdog (crash
                # or stall) — its segment numbering restarted from 0, so
                # anything keyed off the old wall-clock epoch would
                # silently decouple from the new manifest. Reset both in
                # lockstep and swap in a fresh store so stale detection
                # data from the previous generation isn't mixed with the
                # new one.
                last_seen_hls_generation = hls_service.generation
                cycle_start_wall = time.monotonic()
                current_segment_index = -1
                segment_durations = []
                segment_wall_start = time.monotonic()
                store = segment_store.SegmentStore.for_live()
                store.hls_generation = last_seen_hls_generation
                segment_store.register(camera_id, store)
                logger.info(
                    "HLS generation changed to %d for camera %s — reset segment epoch",
                    last_seen_hls_generation,
                    camera_id,
                )

            if frame is None:
                continue

            if store.frame_width is None:
                height, width = frame.image.shape[:2]
                store.set_source_resolution(width, height)

            t0 = time.monotonic()
            results = pipeline.process(frame)
            duration = time.monotonic() - t0

            # RTSP has no content timeline (Frame.video_time is always None
            # here), so segment bucketing uses wall-clock-since-cycle-start
            # instead — a best-effort approximation: detection (OpenCV) and
            # HLS (ffmpeg) are two independent connections to the camera,
            # so their segment boundaries can drift apart over a long
            # session, unlike the file case where both sides read the
            # exact same deterministic content. Accepted, not a bug.
            effective_time = time.monotonic() - cycle_start_wall

            last_video_time = effective_time
            new_segment_index = int(effective_time // HLS_SEGMENT_SECONDS)
            if new_segment_index != current_segment_index and segment_durations:
                wall_elapsed = time.monotonic() - segment_wall_start
                logger.info(
                    "Segment %d done: %d frames | detection time min=%.3fs max=%.3fs avg=%.3fs "
                    "| wall_elapsed=%.2fs (budget=%.1fs) | queue_depth=%d dropped_total=%d",
                    current_segment_index,
                    len(segment_durations),
                    min(segment_durations),
                    max(segment_durations),
                    sum(segment_durations) / len(segment_durations),
                    wall_elapsed,
                    HLS_SEGMENT_SECONDS,
                    frame_queue.qsize(),
                    frame_queue.dropped_frames,
                )
                segment_durations = []
                segment_wall_start = time.monotonic()
            current_segment_index = new_segment_index
            store.add_frame(current_segment_index, effective_time, results)

            segment_durations.append(duration)
    finally:
        if current_segment_index >= 0:
            store.mark_source_ended(current_segment_index + 1, last_video_time)
        stop_event.set()
        reader.release()
        hls_service.stop()
        logger.info("Cycle finished, dropped frames: %d", frame_queue.dropped_frames)


def run(source: str, camera_id: str, api_host: str, api_port: int) -> None:
    pipeline, controller = bootstrap(camera_id, api_host, api_port)

    # A live camera should start monitoring immediately — there's no
    # "recording" for a viewer to click Play on. POST /start remains valid
    # (a no-op) if a frontend calls it anyway.
    controller.trigger_start()

    try:
        logger.info("Start signal received — beginning live camera processing for %s", camera_id)
        _run_live_cycle(source, camera_id, pipeline)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ANPR detection pipeline against a live RTSP camera")
    parser.add_argument("--source", required=True, help="An rtsp:// URL")
    parser.add_argument("--camera-id", required=True, help="Identifier for this camera")
    parser.add_argument("--api-host", default="0.0.0.0", help="Host to bind the API server to")
    parser.add_argument("--api-port", type=int, default=8765, help="Port for the API server")
    args = parser.parse_args()
    if not is_rtsp_source(args.source):
        parser.error(f"app.live.main requires an rtsp:// source, got: {args.source!r} — use app.static.main instead")
    run(args.source, args.camera_id, args.api_host, args.api_port)


if __name__ == "__main__":
    main()
