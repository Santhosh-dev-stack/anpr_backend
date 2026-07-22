import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    camera_id: str
    frame_id: int
    timestamp: float
    # Every tracked vehicle now gets a real track_id (see VehicleTracker) —
    # kept Optional for defensive typing, not because a real untracked-
    # vehicle case exists anymore.
    track_id: int | None
    # Bumped each time this camera's track_id numbering restarts from
    # scratch (Pipeline.reset_for_new_cycle — a static video's Play-button
    # restart; always 0 for a live/RTSP source, which never restarts).
    # Part of the DB upsert key alongside (camera_id, track_id) specifically
    # so a replayed video's reused track_id numbers can't silently overwrite
    # a DIFFERENT vehicle's row from an earlier cycle — see db_sink.py.
    generation: int
    vehicle_type: str
    # No vehicle-detection stage — plates are tracked directly on the full
    # frame, so there's no vehicle box/confidence to report.
    vehicle_bbox: tuple[int, int, int, int] | None
    plate_bbox: tuple[int, int, int, int] | None
    plate: str | None
    # Registration category ("private"/"commercial"/"ev"/"government") read
    # from the plate's background color — India RTO convention, independent
    # of vehicle_type (body shape) and OCR text. None for a plateless
    # vehicle box, same as plate/plate_bbox.
    plate_category: str | None
    vehicle_confidence: float | None
    plate_confidence: float | None
    ocr_confidence: float | None


class ResultSink(ABC):
    """Output boundary for aggregated detection results. DB storage and
    WebSocket broadcast (added in a later pass) are just new subclasses —
    pipeline.py never needs to change.
    """

    @abstractmethod
    def emit(self, result: DetectionResult) -> None:
        ...


class PrintSink(ResultSink):
    def emit(self, result: DetectionResult) -> None:
        print(json.dumps(asdict(result)))


class CallbackSink(ResultSink):
    def __init__(self, callback: Callable[[DetectionResult], None]):
        self._callback = callback

    def emit(self, result: DetectionResult) -> None:
        self._callback(result)


class CompositeSink(ResultSink):
    """Fans a result out to multiple sinks, e.g. PrintSink + DbSink at once."""

    def __init__(self, sinks: list[ResultSink]):
        self._sinks = sinks

    def emit(self, result: DetectionResult) -> None:
        for sink in self._sinks:
            sink.emit(result)
