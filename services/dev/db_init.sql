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
    role          TEXT        NOT NULL DEFAULT 'user',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user';

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

-- Seed — admin accounts
INSERT INTO users (username, email, password_hash, role)
VALUES (
    'admin01',
    'admin01@sonic.lab',
    'ed4b2c0e0ee5ef289ee98362a9927df1:1ba5baa2973abc5ed8c222669d4f7a75a688ed4dd720db696016137a60afd53c8da6a1434681aab463787d59c6586fc85d4a7fd68863c5969984729e49a67815',
    'admin'
), (
    'admin02',
    'admin02@sonic.lab',
    '5c4a0bfc94fa6eb01e6ab39937082407:d39cad53417b119940acfa8b78fd5773bf2a5e6a25d1bac6a81c5c0063b0bcbb16db2ec79dd53cdadb021edb667d0c755206bfaef584cde172a7a482c2e92c08',
    'admin'
), (
    'admin03',
    'admin03@sonic.lab',
    'efad28e12c6b3064e68b7c94da2d4ed8:45add704bafc140119e26751a1d92701f3f5df8ad095c68d9235f3b72642d086990d71288d582d2544dfee03fb6fc273f078b37de557e96572527e8abe3241dd',
    'admin'
)
ON CONFLICT (username) DO UPDATE
SET email = EXCLUDED.email,
    password_hash = EXCLUDED.password_hash,
    role = EXCLUDED.role;
