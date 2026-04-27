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
-- Seed — test account (password: test1234)
-- ============================================================
INSERT INTO users (username, email, password_hash)
VALUES (
    'testuser',
    'test@test.com',
    '67eb8ad9c110d329032388a788d9c382:4cb0c31056573c14a5f6dc47b6f8e04d3622672029c8960da2862ae77d64ca2d4e5b84f3937be7c7914f7628b3f99db0b61455fea0e93ac53e0c09108a9c074c'
)
ON CONFLICT DO NOTHING;
