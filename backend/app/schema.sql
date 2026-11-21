-- RetailFlow database schema (PostgreSQL). Applied at API start-up; idempotent.
--
-- Only anonymous analytics are stored: no images of people, no identities.
-- track_id is an integer that is only meaningful inside one session.

CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    scene       JSONB NOT NULL,               -- line, zones, profile... (scenes/<camera>.json)
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One session = one run of the pipeline on one source (a video file or a live
-- stream). Times inside a session are seconds on THAT session's timeline
-- (video time for files), so sessions are never merged onto a fake clock.
CREATE TABLE IF NOT EXISTS sessions (
    session_id  UUID PRIMARY KEY,
    camera_id   TEXT NOT NULL REFERENCES cameras (camera_id) ON DELETE CASCADE,
    source      TEXT NOT NULL,
    video_fps   REAL,
    run_info    JSONB NOT NULL DEFAULT '{}',
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),   -- when the run started (server clock)
    finished_at TIMESTAMPTZ,
    summary     JSONB
);

CREATE TABLE IF NOT EXISTS events (
    id         BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions (session_id) ON DELETE CASCADE,
    frame      INTEGER NOT NULL,
    ts         DOUBLE PRECISION NOT NULL,             -- seconds on the session timeline
    type       TEXT NOT NULL CHECK (type IN ('line_cross', 'zone_enter', 'zone_exit',
                                             'staff_flag', 'static_flag')),
    track_id   INTEGER,
    direction  TEXT CHECK (direction IN ('in', 'out')),
    zone       TEXT,
    x          REAL,
    y          REAL,
    value      REAL,                                  -- dwell seconds for zone_exit
    staff      BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS events_session_type_ts ON events (session_id, type, ts);
CREATE INDEX IF NOT EXISTS events_flags ON events (session_id, track_id)
    WHERE type IN ('staff_flag', 'static_flag');

-- Tracks flagged as staff or static objects (mannequins). Flags are retroactive:
-- every event of a flagged track is excluded from customer analytics.
CREATE OR REPLACE VIEW customer_events AS
SELECT e.*
FROM events e
WHERE NOT EXISTS (
    SELECT 1 FROM events f
    WHERE f.session_id = e.session_id
      AND f.track_id = e.track_id
      AND f.type IN ('staff_flag', 'static_flag')
);

-- Person-seconds grid of a session, plus a people-free background image.
CREATE TABLE IF NOT EXISTS heatmaps (
    session_id UUID PRIMARY KEY REFERENCES sessions (session_id) ON DELETE CASCADE,
    cell       REAL NOT NULL,                         -- cell size: pixels, or metres if floor
    floor      BOOLEAN NOT NULL DEFAULT FALSE,        -- true = floor-plan grid (homography)
    grid       JSONB NOT NULL,                        -- rows x cols person-seconds
    background BYTEA                                  -- PNG, median of sampled frames
);

-- A store: several calibrated cameras sharing one floor (run_store.py). `config` holds
-- the camera names, their floor positions and the floor bounds, for drawing the map.
CREATE TABLE IF NOT EXISTS stores (
    store_id    TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    config      JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One live run over all of a store's cameras. `state` is the latest store-wide picture
-- (replaced every few seconds): store-wide person numbers, floor positions, camera
-- names and video-timeline seconds only. No images, no appearance features.
CREATE TABLE IF NOT EXISTS store_runs (
    run_id      UUID PRIMARY KEY,
    store_id    TEXT NOT NULL REFERENCES stores (store_id) ON DELETE CASCADE,
    run_info    JSONB NOT NULL DEFAULT '{}',
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    state       JSONB,
    summary     JSONB
);
