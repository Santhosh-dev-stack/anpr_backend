from app.database.connection import get_connection

# One row per (camera_id, track_id): a vehicle sighting is upserted as better
# plate reads arrive over the track's lifetime, rather than inserting a new
# row per frame (Phase 10/11 emits per-frame results, but persisting every
# frame would bloat the table with near-duplicate rows for the same vehicle).
CREATE_VEHICLE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS vehicle_events (
    id BIGSERIAL PRIMARY KEY,
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    vehicle_type TEXT NOT NULL,
    plate_number TEXT,
    plate_category TEXT,
    vehicle_image TEXT,
    plate_image TEXT,
    vehicle_confidence REAL,
    plate_confidence REAL,
    ocr_confidence REAL,
    vehicle_bbox INTEGER[4],
    plate_bbox INTEGER[4],
    first_seen_frame INTEGER NOT NULL,
    last_seen_frame INTEGER NOT NULL,
    first_seen_at DOUBLE PRECISION NOT NULL,
    last_seen_at DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (camera_id, track_id)
);
"""

CREATE_PLATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_vehicle_events_plate_number
    ON vehicle_events (plate_number);
"""

# CREATE TABLE IF NOT EXISTS above is a no-op against a table that already
# existed before plate_category was added, so add it separately.
ADD_PLATE_CATEGORY_COLUMN = """
ALTER TABLE vehicle_events ADD COLUMN IF NOT EXISTS plate_category TEXT;
"""


def init_schema() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_VEHICLE_EVENTS_TABLE)
            cur.execute(CREATE_PLATE_INDEX)
            cur.execute(ADD_PLATE_CATEGORY_COLUMN)
