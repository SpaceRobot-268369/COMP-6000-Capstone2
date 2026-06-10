-- Migration: Model Selection Settings
-- To run manually:
--   psql -d capstone2 -f services/migrations/001_model_configs.sql
-- Or inside the Docker postgres container:
--   docker exec -it services-db-1 psql -U postgres -d capstone2 -f /workspace/services/migrations/001_model_configs.sql

-- ============================================================
-- Helper Function (in case it wasn't created yet)
-- ============================================================
CREATE OR REPLACE FUNCTION fn_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- Model Selection Settings
-- ============================================================
CREATE TABLE IF NOT EXISTS model_configs (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER     REFERENCES users(id) ON DELETE CASCADE,  -- NULL = global
    name        TEXT        NOT NULL DEFAULT 'default',
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_config_slots (
    id          SERIAL PRIMARY KEY,
    config_id   INTEGER NOT NULL REFERENCES model_configs(id) ON DELETE CASCADE,
    slot        TEXT    NOT NULL,   -- 'layer_a' | … | 'layer_e_aggregator'
    attempt_id  TEXT    NOT NULL,
    UNIQUE (config_id, slot)
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_global_config
    ON model_configs (is_active)
    WHERE user_id IS NULL AND is_active;

CREATE OR REPLACE TRIGGER trg_model_configs_updated_at
    BEFORE UPDATE ON model_configs
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();
