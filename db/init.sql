-- PRT-1 套准质检 — PostgreSQL schema (reference; the app also creates these
-- tables automatically via SQLAlchemy metadata on startup).

CREATE TABLE IF NOT EXISTS scans (
    id                      SERIAL PRIMARY KEY,
    run_id                  VARCHAR(32) UNIQUE NOT NULL,
    filename                VARCHAR(255) NOT NULL,
    sha256                  VARCHAR(64) NOT NULL,
    width_px                INTEGER NOT NULL,
    height_px               INTEGER NOT NULL,
    declared_scan_ppi       DOUBLE PRECISION,
    measured_ppi            DOUBLE PRECISION,
    frame_rotation_deg      DOUBLE PRECISION NOT NULL,
    frame_scale             DOUBLE PRECISION NOT NULL,
    frame_quality           DOUBLE PRECISION NOT NULL,
    residual_rotation_deg   DOUBLE PRECISION NOT NULL,
    residual_scale          DOUBLE PRECISION NOT NULL,
    status                  VARCHAR(32) NOT NULL,
    detector_version        VARCHAR(48) NOT NULL,
    provenance              JSONB NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_scans_sha ON scans (sha256);

CREATE TABLE IF NOT EXISTS plate_readings (
    id              SERIAL PRIMARY KEY,
    scan_id         INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    plate           VARCHAR(16) NOT NULL,
    status          VARCHAR(16) NOT NULL,           -- ok | candidate | missing
    dx_um           DOUBLE PRECISION,               -- NULL when missing
    dy_um           DOUBLE PRECISION,
    x_lo_um         DOUBLE PRECISION,
    x_hi_um         DOUBLE PRECISION,
    y_lo_um         DOUBLE PRECISION,
    y_hi_um         DOUBLE PRECISION,
    confidence      DOUBLE PRECISION NOT NULL,
    n_x_readings    INTEGER NOT NULL,
    n_y_readings    INTEGER NOT NULL,
    notes           JSONB NOT NULL DEFAULT '[]',
    UNIQUE (scan_id, plate)
);

-- Machine readings are never overwritten: human decisions live separately.
CREATE TABLE IF NOT EXISTS confirmations (
    id                  SERIAL PRIMARY KEY,
    scan_id             INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    operator_id         VARCHAR(64) NOT NULL,
    decision            VARCHAR(16) NOT NULL CHECK
                        (decision IN ('confirmed','corrected','rejected')),
    corrected_dx_um     DOUBLE PRECISION,
    corrected_dy_um     DOUBLE PRECISION,
    reason              TEXT NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_confirm_scan ON confirmations (scan_id, created_at);
