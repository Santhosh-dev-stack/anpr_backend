import re
from dataclasses import dataclass

import cv2
import numpy as np
from paddleocr import PaddleOCR

from app.config import OCR_LANG
from app.utils.logger import get_logger

logger = get_logger(__name__)

_PLATE_TARGET_HEIGHT = 64
_MAX_PLATE_WIDTH = 400  # caps pathological aspect ratios from a bad plate detection
_ALLOWED_CHARS = re.compile(r"[^A-Z0-9]")


@dataclass
class OcrResult:
    text: str
    confidence: float


def _preprocess(plate_crop: np.ndarray) -> np.ndarray:
    h, w = plate_crop.shape[:2]
    scale = _PLATE_TARGET_HEIGHT / max(h, 1)
    # An extreme aspect ratio (e.g. a near-zero-height mis-detection) would
    # otherwise blow this up to a huge width, making the denoise/blur below
    # pathologically slow — cap it rather than trust the detector's box.
    new_w = min(max(int(w * scale), 1), _MAX_PLATE_WIDTH)
    resized = cv2.resize(plate_crop, (new_w, _PLATE_TARGET_HEIGHT))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(equalized, h=10)

    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def _normalize_text(text: str) -> str:
    return _ALLOWED_CHARS.sub("", text.upper().strip())


class PlateReader:
    """PaddleOCR-based plate text recognizer with preprocessing tuned for
    small, often low-quality license-plate crops (resize, CLAHE, denoise,
    sharpen) before recognition.
    """

    def __init__(self, lang: str = OCR_LANG):
        self._ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)
        logger.info("Loaded PaddleOCR (lang=%s)", lang)

    def read(self, plate_crop: np.ndarray) -> OcrResult | None:
        if plate_crop.size == 0:
            return None
        preprocessed = _preprocess(plate_crop)
        result = self._ocr.ocr(preprocessed, cls=False)
        if not result or not result[0]:
            return None

        # A double-line plate (state+RTO code on one line, series+number on
        # the next — common on two-wheelers) is detected by PaddleOCR as
        # *separate* text boxes, one per line. Keeping only the single
        # highest-confidence box (the old behavior) silently discarded
        # whichever line lost that comparison, so a double-line plate could
        # never produce its full number — only ever half of it. Sorting by
        # each box's top y-coordinate and concatenating reconstructs the
        # correct top-to-bottom reading order instead.
        lines = sorted(result[0], key=lambda entry: min(point[1] for point in entry[0]))

        texts: list[str] = []
        confidences: list[float] = []
        for _, (text, conf) in lines:
            normalized = _normalize_text(text)
            if normalized:
                texts.append(normalized)
                confidences.append(conf)

        if not texts:
            return None
        combined_text = "".join(texts)
        combined_confidence = sum(confidences) / len(confidences)
        return OcrResult(text=combined_text, confidence=float(combined_confidence))
