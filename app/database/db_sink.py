from app.database.connection import get_connection
from app.services.result_sink import DetectionResult, ResultSink
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Upserts one row per (camera_id, track_id): plate_number/confidences only
# improve when this frame's OCR read is better than what's stored, so a
# track's row converges to its best reading instead of being overwritten by
# a worse or empty later attempt.
_UPSERT_SQL = """
INSERT INTO vehicle_events (
    camera_id, track_id, generation, vehicle_type, plate_number, plate_category,
    vehicle_confidence, plate_confidence, ocr_confidence,
    vehicle_bbox, plate_bbox,
    first_seen_frame, last_seen_frame, first_seen_at, last_seen_at
) VALUES (
    %(camera_id)s, %(track_id)s, %(generation)s, %(vehicle_type)s, %(plate_number)s, %(plate_category)s,
    %(vehicle_confidence)s, %(plate_confidence)s, %(ocr_confidence)s,
    %(vehicle_bbox)s, %(plate_bbox)s,
    %(frame_id)s, %(frame_id)s, %(timestamp)s, %(timestamp)s
)
ON CONFLICT (camera_id, generation, track_id) DO UPDATE SET
    vehicle_type = EXCLUDED.vehicle_type,
    plate_category = COALESCE(EXCLUDED.plate_category, vehicle_events.plate_category),
    vehicle_confidence = GREATEST(vehicle_events.vehicle_confidence, EXCLUDED.vehicle_confidence),
    plate_number = CASE
        WHEN EXCLUDED.ocr_confidence IS NOT NULL
             AND (vehicle_events.ocr_confidence IS NULL
                  OR EXCLUDED.ocr_confidence > vehicle_events.ocr_confidence)
        THEN EXCLUDED.plate_number
        ELSE vehicle_events.plate_number
    END,
    plate_confidence = CASE
        WHEN EXCLUDED.ocr_confidence IS NOT NULL
             AND (vehicle_events.ocr_confidence IS NULL
                  OR EXCLUDED.ocr_confidence > vehicle_events.ocr_confidence)
        THEN EXCLUDED.plate_confidence
        ELSE vehicle_events.plate_confidence
    END,
    ocr_confidence = GREATEST(
        COALESCE(vehicle_events.ocr_confidence, 0), COALESCE(EXCLUDED.ocr_confidence, 0)
    ),
    plate_bbox = COALESCE(EXCLUDED.plate_bbox, vehicle_events.plate_bbox),
    last_seen_frame = EXCLUDED.last_seen_frame,
    last_seen_at = EXCLUDED.last_seen_at,
    updated_at = now();
"""


class DbSink(ResultSink):
    """Persists detection results to PostgreSQL's vehicle_events table.

    One row per (camera_id, track_id), upserted as better OCR reads arrive
    over the track's lifetime — see schema.py for why this isn't one row
    per frame.
    """

    def emit(self, result: DetectionResult) -> None:
        params = {
            "camera_id": result.camera_id,
            "track_id": result.track_id,
            "generation": result.generation,
            "vehicle_type": result.vehicle_type,
            "plate_number": result.plate,
            "plate_category": result.plate_category,
            "vehicle_confidence": result.vehicle_confidence,
            "plate_confidence": result.plate_confidence,
            "ocr_confidence": result.ocr_confidence,
            "vehicle_bbox": list(result.vehicle_bbox) if result.vehicle_bbox else None,
            "plate_bbox": list(result.plate_bbox) if result.plate_bbox else None,
            "frame_id": result.frame_id,
            "timestamp": result.timestamp,
        }
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(_UPSERT_SQL, params)
        except Exception:
            logger.exception(
                "Failed to persist detection result for track_id=%s", result.track_id
            )
