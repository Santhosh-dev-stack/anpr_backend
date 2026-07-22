from app.database.connection import get_connection

# One row per (camera_id, generation, track_id): a vehicle sighting is
# upserted as better plate reads arrive over the track's lifetime, rather
# than inserting a new row per frame (Phase 10/11 emits per-frame results,
# but persisting every frame would bloat the table with near-duplicate rows
# for the same vehicle). `generation` is part of the key (not just
# camera_id+track_id) because a static video's Play-button restart
# deliberately reuses track_id numbers from 1 each cycle (see
# Pipeline.reset_for_new_cycle) — without generation in the key, a replayed
# cycle's track_id 1 would silently overwrite a completely different
# vehicle's row from an earlier cycle that happened to also be track_id 1.
CREATE_VEHICLE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS vehicle_events (
    id BIGSERIAL PRIMARY KEY,
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
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
    UNIQUE (camera_id, generation, track_id)
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

# Same reasoning — a table created before `generation` existed needs both
# the column added AND the unique constraint widened from
# (camera_id, track_id) to (camera_id, generation, track_id). Wrapped in a
# DO block since Postgres has no ADD CONSTRAINT IF NOT EXISTS; safe to run
# on every startup (checks pg_constraint before adding).
ADD_GENERATION_COLUMN_AND_CONSTRAINT = """
DO $$
BEGIN
    ALTER TABLE vehicle_events ADD COLUMN IF NOT EXISTS generation INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE vehicle_events DROP CONSTRAINT IF EXISTS vehicle_events_camera_id_track_id_key;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'vehicle_events_camera_id_generation_track_id_key'
    ) THEN
        ALTER TABLE vehicle_events ADD CONSTRAINT vehicle_events_camera_id_generation_track_id_key
            UNIQUE (camera_id, generation, track_id);
    END IF;
END $$;
"""


def init_schema() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_VEHICLE_EVENTS_TABLE)
            cur.execute(CREATE_PLATE_INDEX)
            cur.execute(ADD_PLATE_CATEGORY_COLUMN)
            cur.execute(ADD_GENERATION_COLUMN_AND_CONSTRAINT)
