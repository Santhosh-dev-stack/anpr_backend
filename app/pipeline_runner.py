import threading
import time

from app import config
from app.camera.frame_source import FrameSource
from app.camera.video_reader import VideoReader
from app.config import DATABASE_URL
from app.services import pipeline_controller
from app.services.api_server import ApiServer
from app.services.frame_queue import FrameQueue
from app.services.pipeline import Pipeline
from app.services.pipeline_controller import PipelineController
from app.services.result_sink import CompositeSink, PrintSink, ResultSink
from app.utils.logger import get_logger

logger = get_logger(__name__)


def reader_loop(reader: FrameSource, frame_queue: FrameQueue, stop_event: threading.Event) -> None:
    # The isinstance(reader, VideoReader) branch is load-bearing, not
    # incidental: RTSPReader.read() returns None transiently (no frame
    # ready yet — sleep and keep going), while VideoReader.read() returns
    # None permanently at EOF (cv2.VideoCapture.isOpened() stays True past
    # EOF, so is_open can't be used to detect end-of-file instead). Do not
    # split or "simplify" this per source type — that's exactly how a live
    # reader thread would end up dying on the first ordinary inter-frame gap.
    while not stop_event.is_set() and reader.is_open:
        frame = reader.read()
        if frame is None:
            if isinstance(reader, VideoReader):
                break
            time.sleep(0.01)
            continue
        frame_queue.put(frame)
    stop_event.set()


def build_sink() -> ResultSink:
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


def bootstrap(camera_id: str, api_host: str, api_port: int) -> tuple[Pipeline, PipelineController]:
    """Shared process-startup sequence for both the live and static
    entrypoints: sync the advertised API port, load models once (the slow
    part), start the API server, and register a fresh start-gate
    controller. Identical for both source types — kept in one place so a
    future change (e.g. a new CLI-configurable startup step) can't be added
    to one entrypoint and forgotten in the other.
    """
    # Keep the cameras API's advertised port in sync with what we actually bind.
    config.PUBLIC_PORT = api_port

    # Loaded once and reused across cycles — reloading models on every replay
    # of a video file would be wasteful, and PlateTracker degrades gracefully
    # across a source restart (new track_ids get assigned once motion
    # prediction can no longer match, same as a real camera scene cut).
    pipeline = Pipeline(camera_id=camera_id, sink=build_sink())

    api_server = ApiServer(host=api_host, port=api_port)
    api_server.start()

    # Models are already loaded above (the slow part) — but the actual
    # video-reading/detection loop waits here until something explicitly
    # asks it to start (e.g. the frontend's Play button hitting
    # POST /api/cameras/{id}/start), instead of always running the moment
    # this process launches regardless of whether anyone is watching.
    # (The live entrypoint immediately triggers this itself — see
    # app/live/main.py — since a live camera has no "Play button" step.)
    controller = pipeline_controller.PipelineController()
    pipeline_controller.register(camera_id, controller)

    return pipeline, controller
