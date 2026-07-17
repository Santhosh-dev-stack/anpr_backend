import argparse
import threading
import time

from app import config
from app.camera.frame_source import Frame, FrameSource, is_rtsp_source
from app.camera.rtsp_reader import RTSPReader
from app.camera.video_reader import VideoReader
from app.config import DATABASE_URL, FRAME_QUEUE_MAXSIZE, HLS_SEGMENT_SECONDS, PROCESSING_FPS
from app.services import pipeline_controller, segment_store
from app.services.api_server import ApiServer
from app.services.frame_queue import FrameQueue
from app.services.hls_service import HlsService
from app.services.pipeline import Pipeline
from app.services.result_sink import CompositeSink, PrintSink, ResultSink
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _reader_loop(reader: FrameSource, frame_queue: FrameQueue, stop_event: threading.Event) -> None:
    while not stop_event.is_set() and reader.is_open:
        frame = reader.read()
        if frame is None:
            if isinstance(reader, VideoReader):
                break
            time.sleep(0.01)
            continue
        frame_queue.put(frame)
    stop_event.set()


def _build_sink() -> ResultSink:
    sinks: list[ResultSink] = [PrintSink()]
    if DATABASE_URL:
        from app.database.db_sink import DbSink
        from app.database.schema import init_schema

        init_schema()
        sinks.append(DbSink())
        logger.info("DATABASE_URL set — persisting results to PostgreSQL")
    else:
        logger.info("DATABASE_URL not set — results will only be printed, not persisted")
    return CompositeSink(sinks)


def _run_one_cycle(source: str, camera_id: str, pipeline: Pipeline) -> None:
    """Runs the reader+HLS+pipeline loop for one pass over `source`. For a
    video file this ends when the file is exhausted; for RTSP it runs until
    interrupted (RTSPReader reconnects internally, so one "cycle" is the
    whole process lifetime).
    """
    is_rtsp = is_rtsp_source(source)
    reader: FrameSource = RTSPReader(source) if is_rtsp else VideoReader(source, target_fps=PROCESSING_FPS)
    frame_queue = FrameQueue(maxsize=FRAME_QUEUE_MAXSIZE)

    # Fresh store each cycle, matching the fresh reader/HLS output — a client
    # fetching mid-cycle only ever sees this cycle's segments, never a mix.
    # Registered before HlsService.start() so its manifest-readiness poll
    # (see HlsService) always has a store to report into. RTSP gets a
    # bounded store (see SegmentStore) since it never naturally ends.
    store = segment_store.SegmentStore(bounded=is_rtsp)
    segment_store.register(camera_id, store)

    hls_service = HlsService(camera_id=camera_id, source=source)
    hls_service.start()
    # Tracked so a later ffmpeg watchdog restart (RTSP only — see
    # HlsService) can be detected and reacted to below: its segment
    # numbering resets to 0 on relaunch, so the wall-clock epoch and
    # detection store must reset in lockstep with it, not silently drift.
    last_seen_hls_generation = hls_service.generation

    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=_reader_loop, args=(reader, frame_queue, stop_event), daemon=True
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

            if is_rtsp and hls_service.generation != last_seen_hls_generation:
                # ffmpeg was restarted by HlsService's watchdog (crash or
                # stall) — its segment numbering restarted from 0, so
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
                store = segment_store.SegmentStore(bounded=True)
                store.hls_generation = last_seen_hls_generation
                segment_store.register(camera_id, store)
                logger.info(
                    "HLS generation changed to %d for camera %s — reset segment epoch",
                    last_seen_hls_generation, camera_id,
                )

            if frame is None:
                continue

            if store.frame_width is None:
                height, width = frame.image.shape[:2]
                store.set_source_resolution(width, height)

            t0 = time.monotonic()
            results = pipeline.process(frame)
            duration = time.monotonic() - t0

            # video_time is the file decoder's own PTS (see Frame docstring)
            # — None for RTSP, which has no content timeline. Falling back
            # to wall-clock-since-cycle-start makes RTSP segment bucketing
            # possible at all (previously never triggered, so store.add_frame
            # was never called for a live camera), but it's a best-effort
            # approximation: detection (OpenCV) and HLS (ffmpeg) are two
            # independent connections to the camera, so their segment
            # boundaries can drift apart over a long session, unlike the
            # file case where both sides read the exact same deterministic
            # content. Accepted, not considered a bug.
            effective_time = frame.video_time
            if effective_time is None:
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
    # Keep the cameras API's advertised port in sync with what we actually bind.
    config.PUBLIC_PORT = api_port

    # Loaded once and reused across cycles — reloading models on every replay
    # of a video file would be wasteful, and PlateTracker degrades gracefully
    # across a source restart (new track_ids get assigned once motion
    # prediction can no longer match, same as a real camera scene cut).
    pipeline = Pipeline(camera_id=camera_id, sink=_build_sink())

    api_server = ApiServer(host=api_host, port=api_port)
    api_server.start()

    # Models are already loaded above (the slow part) — but the actual
    # video-reading/detection loop waits here until something explicitly
    # asks it to start (e.g. the frontend's Play button hitting
    # POST /api/cameras/{id}/start), instead of always running the moment
    # this process launches regardless of whether anyone is watching.
    controller = pipeline_controller.PipelineController()
    pipeline_controller.register(camera_id, controller)
    is_rtsp = is_rtsp_source(source)

    if is_rtsp:
        # A live camera should start monitoring immediately — there's no
        # "recording" for a viewer to click Play on. POST /start remains
        # valid (a no-op) if a frontend calls it anyway.
        controller.trigger_start()

    # Video files run exactly one cycle per Play click, not an
    # auto-restarting loop. Looping silently reset the HLS output/
    # segment_store to a fresh cycle while the browser's single hls.js
    # instance was still playing through the *previous* cycle's manifest —
    # segment numbers and detection data went out of sync mid-playback
    # (404s, missing boxes) once the backend had already moved on to a
    # later cycle than what the client was still watching. Resetting the
    # controller after each cycle lets a later Play click (POST /start)
    # trigger a fresh, clean cycle instead of looping unattended. RTSP
    # already behaves as one indefinite "cycle" (RTSPReader reconnects
    # internally), so it only ever runs through this once.
    try:
        while True:
            logger.info("Ready — waiting for start signal for camera %s", camera_id)
            controller.wait_for_start()
            logger.info("Start signal received — beginning video processing")
            _run_one_cycle(source, camera_id, pipeline)
            if is_rtsp:
                break
            controller.reset()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ANPR detection pipeline")
    parser.add_argument("--source", required=True, help="Path to a video file or an rtsp:// URL")
    parser.add_argument("--camera-id", required=True, help="Identifier for this camera")
    parser.add_argument("--api-host", default="0.0.0.0", help="Host to bind the API server to")
    parser.add_argument("--api-port", type=int, default=8765, help="Port for the API server")
    args = parser.parse_args()
    run(args.source, args.camera_id, args.api_host, args.api_port)


if __name__ == "__main__":
    main()
