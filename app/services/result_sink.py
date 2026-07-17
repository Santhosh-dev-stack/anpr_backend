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
    # None for a plateless vehicle box (see PlateTracker.UntrackedVehicle) —
    # it was never matched a track_id since there's no plate center to
    # anchor continuity on; these are display-only and never reach OCR/DB.
    track_id: int | None
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
