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
-- Jobs / workers
-- MVP on-demand AI worker bridge. These tables are also ensured
-- by backend startup because this init script only runs when a new
-- PostgreSQL data directory is created.
-- ============================================================
CREATE TABLE IF NOT EXISTS jobs (
    id               BIGSERIAL PRIMARY KEY,
    job_type         TEXT        NOT NULL CHECK (job_type IN ('generation', 'training')),
    layer            TEXT        NOT NULL DEFAULT 'unknown',
    status           TEXT        NOT NULL DEFAULT 'queued',
    priority         INTEGER     NOT NULL DEFAULT 100,
    created_by       TEXT,
    repo_branch      TEXT,
    command          TEXT,
    params_json      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    output_path      TEXT,
    expected_artifacts JSONB     NOT NULL DEFAULT '[]'::jsonb,
    claimed_by       TEXT,
    lease_until      TIMESTAMPTZ,
    heartbeat_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at       TIMESTAMPTZ,
    completed_at     TIMESTAMPTZ,
    error_message    TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_priority ON jobs (status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_claimed_by      ON jobs (claimed_by);
CREATE INDEX IF NOT EXISTS idx_jobs_lease_until     ON jobs (lease_until);

CREATE TABLE IF NOT EXISTS workers (
    worker_id     TEXT PRIMARY KEY,
    status        TEXT        NOT NULL DEFAULT 'unknown',
    current_job_id BIGINT     REFERENCES jobs(id) ON DELETE SET NULL,
    metadata_json JSONB       NOT NULL DEFAULT '{}'::jsonb,
    heartbeat_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS job_events (
    id            BIGSERIAL PRIMARY KEY,
    job_id        BIGINT      NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    worker_id     TEXT,
    status        TEXT,
    message       TEXT,
    metadata_json JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_job_events_job_id_created_at ON job_events (job_id, created_at);

CREATE TABLE IF NOT EXISTS job_artifacts (
    id            BIGSERIAL PRIMARY KEY,
    job_id        BIGINT      NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kind          TEXT        NOT NULL,
    path          TEXT        NOT NULL,
    metadata_json JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_job_artifacts_job_id ON job_artifacts (job_id);

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
