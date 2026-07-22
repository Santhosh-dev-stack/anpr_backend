import argparse
import threading
import time

from app.camera.frame_source import Frame, is_rtsp_source
from app.camera.video_reader import VideoReader
from app.config import FRAME_QUEUE_MAXSIZE, HLS_SEGMENT_SECONDS, PROCESSING_FPS
from app.pipeline_runner import bootstrap, reader_loop
from app.services import segment_store
from app.services.frame_queue import FrameQueue
from app.services.pipeline import Pipeline
from app.services.static_hls_service import StaticHlsService
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _run_static_cycle(source: str, camera_id: str, pipeline: Pipeline) -> None:
    """Runs the reader+HLS+pipeline loop for one pass over `source`, ending
    when the file is exhausted.
    """
    # A Play-button restart reuses the same Pipeline across cycles — reset
    # it so every Play click starts completely fresh (track_ids restart
    # from 1, vehicle_count back to 0) rather than continuing to count up
    # from the previous cycle. See Pipeline.reset_for_new_cycle's docstring.
    pipeline.reset_for_new_cycle()

    reader = VideoReader(source, target_fps=PROCESSING_FPS)
    frame_queue = FrameQueue(maxsize=FRAME_QUEUE_MAXSIZE)

    # Fresh store each cycle, matching the fresh reader/HLS output — a
    # client fetching mid-cycle only ever sees this cycle's segments, never
    # a mix. Registered before hls_service.start() — its manifest-readiness
    # poll calls segment_store.get(camera_id), which would silently no-op
    # (no store, no exception) if the store weren't already registered.
    store = segment_store.SegmentStore.for_file()
    # Mirrors pipeline.cycle_generation (bumped by reset_for_new_cycle just
    # above) — lets the frontend detect a Play-button restart and clear its
    # stale display instead of showing this cycle's low, reused track_ids
    # side-by-side with the previous cycle's unrelated vehicles.
    store.generation = pipeline.cycle_generation
    segment_store.register(camera_id, store)

    hls_service = StaticHlsService(camera_id=camera_id, source=source)
    hls_service.start()

    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=reader_loop, args=(reader, frame_queue, stop_event), daemon=True
    )
    reader_thread.start()

    current_segment_index = -1
    last_video_time = 0.0
    segment_durations: list[float] = []
    segment_wall_start = time.monotonic()

    try:
        while not stop_event.is_set() or frame_queue.qsize() > 0:
            frame: Frame | None = frame_queue.get(timeout=1.0)
            # Kept for uniformity with the live entrypoint (which needs
            # this to detect a real outage) even though VideoReader.healthy
            # always returns the FrameSource base-class default True — so
            # the field is populated the same way regardless of which
            # entrypoint's code you're reading.
            store.set_camera_healthy(reader.healthy)

            if frame is None:
                continue

            if store.frame_width is None:
                height, width = frame.image.shape[:2]
                store.set_source_resolution(width, height)

            t0 = time.monotonic()
            results = pipeline.process(frame)
            duration = time.monotonic() - t0

            # video_time is the file decoder's own PTS — always set for a
            # video file (see Frame docstring).
            video_time = frame.video_time
            last_video_time = video_time
            new_segment_index = int(video_time // HLS_SEGMENT_SECONDS)
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
            store.add_frame(current_segment_index, video_time, results)

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

    # Video files run exactly one cycle per Play click, not an
    # auto-restarting loop. Looping silently reset the HLS output/
    # segment_store to a fresh cycle while the browser's single hls.js
    # instance was still playing through the *previous* cycle's manifest —
    # segment numbers and detection data went out of sync mid-playback
    # (404s, missing boxes) once the backend had already moved on to a
    # later cycle than what the client was still watching. Resetting the
    # controller after each cycle lets a later Play click (POST /start)
    # trigger a fresh, clean cycle instead of looping unattended.
    try:
        while True:
            logger.info("Ready — waiting for start signal for camera %s", camera_id)
            controller.wait_for_start()
            logger.info("Start signal received — beginning video processing")
            _run_static_cycle(source, camera_id, pipeline)
            controller.reset()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ANPR detection pipeline against a recorded video file")
    parser.add_argument("--source", required=True, help="Path to a video file")
    parser.add_argument("--camera-id", required=True, help="Identifier for this camera")
    parser.add_argument("--api-host", default="0.0.0.0", help="Host to bind the API server to")
    parser.add_argument("--api-port", type=int, default=8765, help="Port for the API server")
    args = parser.parse_args()
    if is_rtsp_source(args.source):
        parser.error(f"app.static.main requires a file source, got an rtsp:// URL: {args.source!r} — use app.live.main instead")
    run(args.source, args.camera_id, args.api_host, args.api_port)


if __name__ == "__main__":
    main()
