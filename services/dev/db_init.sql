-- db_init.sql
-- Runs once on first PostgreSQL container start.
-- Skipped automatically if the data directory already exists (Docker behaviour).

-- ============================================================
-- Users
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT        NOT NULL UNIQUE,
    email         TEXT        NOT NULL UNIQUE,
    password_hash TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email    ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

-- Auto-update updated_at on every row change
CREATE OR REPLACE FUNCTION fn_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();

-- ============================================================
-- Sessions  (connect-pg-simple schema)
-- Stores server-side cookie sessions. Rows expire automatically
-- via the session store's pruning job.
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    sid    VARCHAR      NOT NULL COLLATE "default" PRIMARY KEY,
    sess   JSON         NOT NULL,
    expire TIMESTAMP(6) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_expire ON sessions (expire);

-- ============================================================
-- Jobs
-- Server A owns job state for on-demand Server B workers.
-- ============================================================
CREATE TABLE IF NOT EXISTS jobs (
    id             BIGSERIAL PRIMARY KEY,
    type           TEXT        NOT NULL CHECK (type IN ('generation', 'training')),
    status         TEXT        NOT NULL DEFAULT 'queued' CHECK (
        status IN (
            'queued',
            'claimed',
            'running',
            'uploading',
            'completed',
            'failed',
            'cancel_requested',
            'cancelled'
        )
    ),
    priority       INTEGER     NOT NULL DEFAULT 0,
    payload        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    result         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    artifact_uri   TEXT,
    log_uri        TEXT,
    error_message  TEXT,
    claimed_by     TEXT,
    claimed_at     TIMESTAMPTZ,
    heartbeat_at   TIMESTAMPTZ,
    started_at     TIMESTAMPTZ,
    finished_at    TIMESTAMPTZ,
    attempt_count  INTEGER     NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts   INTEGER     NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    created_by     INTEGER     REFERENCES users(id) ON DELETE SET NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_priority_created
    ON jobs (status, priority DESC, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_jobs_claimed_by
    ON jobs (claimed_by);

CREATE INDEX IF NOT EXISTS idx_jobs_heartbeat
    ON jobs (heartbeat_at);

CREATE OR REPLACE TRIGGER trg_jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();

-- ============================================================
-- Seed — test account (password: test1234)
-- ============================================================
INSERT INTO users (username, email, password_hash)
VALUES (
    'testuser',
    'test@test.com',
    '67eb8ad9c110d329032388a788d9c382:4cb0c31056573c14a5f6dc47b6f8e04d3622672029c8960da2862ae77d64ca2d4e5b84f3937be7c7914f7628b3f99db0b61455fea0e93ac53e0c09108a9c074c'
)
ON CONFLICT DO NOTHING;
