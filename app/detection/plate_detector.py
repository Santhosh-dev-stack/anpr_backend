import os

from ultralytics import YOLO

from app.config import DEVICE, PLATE_MODEL_PATH
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PlateDetector:
    """Loads the license-plate YOLO weight for direct full-frame detection.

    COCO pretrained YOLO has no license-plate class, so this expects a
    separately trained weight with a plate class (see PLATE_CLASS_NAME in
    app/config.py — the current weight is a combined vehicle+plate model,
    filtered down to just its plate class by PlateTracker). The underlying
    `model` is used directly by tracking.plate_tracker.PlateTracker rather
    than through a predict-on-a-crop method — plates are detected and
    tracked straight on the full frame, with no separate vehicle-detection
    stage first.
    """

    def __init__(self, model_path: str = PLATE_MODEL_PATH, device: str = DEVICE):
        self._model_path = model_path
        self._device = device
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        return self._ensure_loaded()

    def _ensure_loaded(self) -> YOLO:
        if self._model is None:
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f"Plate detection model not found at '{self._model_path}'. "
                    "Download a pretrained license-plate YOLO weight (e.g. from "
                    "Roboflow Universe or the Ultralytics HF hub) and place it "
                    "at this path, or update PLATE_MODEL_PATH in app/config.py."
                )
            self._model = YOLO(self._model_path)
            self._model.to(self._device)
            logger.info("Loaded plate detector %s on %s", self._model_path, self._device)
        return self._model
