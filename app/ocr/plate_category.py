import cv2
import numpy as np

# India RTO plate-background color convention: private=white, commercial=
# yellow, EV=green, government/diplomatic=blue or red. Classified from the
# plate crop's background color (not OCR text), since body/vehicle_type
# from the detector says nothing about registration category.
_MIN_BACKGROUND_PIXELS = 50

# Text strokes are near-black (low V) and specular glare on the plate's
# laminate is near-white blowout (V close to 255) — both would skew a plain
# average toward the wrong color, so background sampling excludes both ends.
_BACKGROUND_MIN_VALUE = 60
_BACKGROUND_MAX_VALUE = 250

# A white/private plate is achromatic (low saturation) regardless of hue —
# measured directly against real plate crops from this video: under normal
# (non-bright) lighting a genuine white plate's median V often lands well
# below a "clean white" value like 140 (e.g. V=117), so gating "private" on
# saturation AND a high value threshold rejected real white plates whenever
# lighting wasn't ideal. Saturation alone is the reliable signal; V only
# needs to rule out a near-black crop (shadow/mis-detection), not require
# brightness.
_ACHROMATIC_SATURATION_MAX = 60
_MIN_VALUE_FOR_CALL = 50

# Hue ranges in OpenCV's H in [0, 179] scale. Yellow's lower bound is pulled
# down from a textbook ~25 to 8: real Indian commercial (yellow) plates
# measured from this video's footage often read as orange (hue ~10-17)
# under its color grading/compression, not pure yellow — e.g. a plate
# reading "TN.03 D.9766" measured hue=12, sat=235 and was previously
# missed entirely (fell between red's old cutoff of 10 and yellow's old
# start of 18). Red's cutoff is correspondingly tightened to 7 so true red
# and orange-shifted yellow don't overlap.
_YELLOW_HUE_RANGE = (8, 38)
# Upper bound pulled up from a textbook ~85 to 100: a real EV plate in this
# video (dark teal/green background, confirmed visually) measured a stable
# hue of 95-100 across 10 consecutive frames — this video's color grading
# renders green closer to teal/cyan than a textbook green. Genuine blue
# plates measured elsewhere in this footage sit at hue 115-123, well clear
# of this range, so the boundary is moved to 101 rather than the old 95
# without creating a new collision.
_GREEN_HUE_RANGE = (40, 100)
_BLUE_HUE_RANGE = (101, 130)
_RED_HUE_LOW_MAX = 7
_RED_HUE_HIGH_MIN = 170


def classify_plate_category(plate_crop: np.ndarray) -> str:
    """Classifies a plate crop's registration category from its background
    color. Returns one of "private", "commercial", "ev", "government", or
    "unknown" (crop too small/washed-out to call reliably).

    No crop-size floor is applied on purpose: measured directly, a
    genuinely readable small plate and a false-positive plate detection
    (PlateTracker occasionally locks onto a mirror/wheel/helmet, not a real
    plate) land in the same size range, so an area cutoff rejected as many
    real small plates as it filtered out junk. A wrong detection producing
    a wrong color label is a pre-existing tracker/detector limitation (see
    plate_tracker.py), not something color classification can fix.
    """
    if plate_crop is None or plate_crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    background_mask = (val > _BACKGROUND_MIN_VALUE) & (val < _BACKGROUND_MAX_VALUE)
    if int(background_mask.sum()) < _MIN_BACKGROUND_PIXELS:
        return "unknown"

    median_hue = float(np.median(hue[background_mask]))
    median_sat = float(np.median(sat[background_mask]))
    median_val = float(np.median(val[background_mask]))

    if median_sat < _ACHROMATIC_SATURATION_MAX:
        return "private" if median_val > _MIN_VALUE_FOR_CALL else "unknown"
    if _YELLOW_HUE_RANGE[0] <= median_hue <= _YELLOW_HUE_RANGE[1]:
        return "commercial"
    if _GREEN_HUE_RANGE[0] <= median_hue <= _GREEN_HUE_RANGE[1]:
        return "ev"
    if _BLUE_HUE_RANGE[0] <= median_hue <= _BLUE_HUE_RANGE[1]:
        return "government"
    if median_hue <= _RED_HUE_LOW_MAX or median_hue >= _RED_HUE_HIGH_MIN:
        return "government"
    return "unknown"
