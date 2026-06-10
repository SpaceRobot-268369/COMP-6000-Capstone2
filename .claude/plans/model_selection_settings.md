# Plan — Model-Selection Settings Page

Let users choose **which attempt each layer uses** (Generation A/B/C/D +
Analysis E heads) from a Settings page, persist the choice in Postgres, and
have the backend forward it to the AI server on every run.

Status: proposal. Not yet implemented.

---

## 1. Problem & current behaviour

Each layer has multiple attempts in `acoustic_ai/registry.yaml`, but the
*active* one is hard-coded as the per-layer `default:` key. There is no way to
switch attempts at runtime without editing the YAML and rebooting serverB.

Key finding from the current code — **the override path already exists on the
AI side but is dead-ended at the backend:**

| Hop | Generation | Analysis |
|---|---|---|
| AI server accepts overrides? | ✅ `orchestrate_generation(layer_a_attempt … layer_d_attempt)` — `registry.py:532` | ✅ `orchestrate_analysis(ambient_attempt … aggregator_attempt)` — `registry.py:624` |
| FastAPI request model carries them? | ✅ `OrchestratedGenerationRequest` — `server.py:160` | ✅ `Form(...)` fields on `/analysis/run` — `server.py:292` |
| Express forwards them? | ❌ `/api/generation` builds payload without `layer_*_attempt` — `index.js:930` | ❌ `/api/analysis` raw-pipes the upload, no attempt fields — `index.js:1005` |
| Persisted anywhere? | ❌ | ❌ |

So the work is: **(a)** a place to store the selection, **(b)** backend reads
it and injects the override fields, **(c)** a UI to edit it. No new model code;
the AI server's `default:` becomes the *fallback* when a slot is unset.

Per-head Analysis dev page (`DevAnalysisPage.jsx`) already lets a developer
pick attempts ad-hoc, but that's transient UI state, not a saved app setting.

---

## 2. Design decisions

### 2.1 Where the active config lives — **Postgres (backend-owned)**

serverB is a disposable worker that "reboots fresh" (per CLAUDE.md), so config
**cannot** live there. Postgres is the only durable store and already backs
the backend. The AI server's registry `default:` stays as the last-resort
fallback. Backend is the source of truth and injects the selection on every
call.

### 2.2 Scope — **single global config row** (decided)

Auth is currently disabled (`requireAuth` is a pass-through, `index.js:92`).
This is a shared research prototype with one serverB, so a single global
"active model config" is the chosen MVP and matches the framing ("which exact
model we gonna use"). Schema keeps a nullable `user_id` so per-user overrides
can be layered on later without a migration. **Anyone can edit for now** — no
role/auth gate; that comes with the later auth work.

### 2.2a Deployment topology (confirms backend-owned design)

Server A runs Postgres **and** the Express backend; Server B runs the AI
service. Server A calls Server B for inference. The model-selection config
therefore lives in **Server A's Postgres**, and **Server A's backend** reads it
and injects the chosen attempt IDs into the same calls it already makes to
Server B. Server B's `registry.yaml` `default:` stays as the fallback when a
slot is unset. (Local dev mirrors this: Postgres + backend in Docker, AI
reached via the `ai-tunnel` sidecar.)

### 2.3 Validation — **against the live registry**

A saved selection must reference an attempt that (a) exists for that layer in
`GET /layers`, and (b) is `available: true` (checkpoint materialised, not just
a DVC pointer). The backend validates on save and rejects unknown/unavailable
attempts. For Analysis, the chosen attempt's `head` must match the slot
(ambient/weather/events/aggregator).

### 2.4 Slots modelled

| Mode | Slots |
|---|---|
| Generation | `layer_a`, `layer_b`, `layer_c`, `layer_d` |
| Analysis | `layer_e_ambient`, `layer_e_weather`, `layer_e_events`, `layer_e_aggregator` |

A row per (config, slot). Unset slot ⇒ AI-server default.

---

## 3. Database schema

Add to `services/dev/db_init.sql` (and ship as a numbered migration for
existing DBs — see §6).

```sql
-- Active model-selection config. One global row today (user_id NULL);
-- per-user rows can be added later. `is_active` lets us keep history.
CREATE TABLE IF NOT EXISTS model_configs (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER     REFERENCES users(id) ON DELETE CASCADE,  -- NULL = global
    name        TEXT        NOT NULL DEFAULT 'default',
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per layer slot. attempt_id is validated against the live registry
-- at write time, not FK-enforced (the registry is YAML, not a table).
CREATE TABLE IF NOT EXISTS model_config_slots (
    id          SERIAL PRIMARY KEY,
    config_id   INTEGER NOT NULL REFERENCES model_configs(id) ON DELETE CASCADE,
    slot        TEXT    NOT NULL,   -- 'layer_a' | … | 'layer_e_aggregator'
    attempt_id  TEXT    NOT NULL,
    UNIQUE (config_id, slot)
);

-- At most one active global config.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_global_config
    ON model_configs (is_active)
    WHERE user_id IS NULL AND is_active;

CREATE TRIGGER trg_model_configs_updated_at
    BEFORE UPDATE ON model_configs
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();
```

`fn_set_updated_at()` already exists in `db_init.sql`. No seed rows needed —
absence of a slot means "use registry default", so an empty config behaves
exactly like today.

---

## 4. Backend changes (`backend/src/index.js`, + new module)

### 4.1 New module `backend/src/modelConfig.js`
- `getActiveConfig()` → `{ slots: { layer_a: attemptId, … } }` from Postgres
  (global row; per-user later). Cache in-process with a short TTL or bust on
  write.
- `setSlots(slots)` → upsert into `model_config_slots`, after validation.
- `validateSlots(slots)` → fetch `GET /layers` via `fetchAi`, check each
  attempt exists + `available` + (Analysis) head matches slot. Returns
  per-slot errors.

### 4.2 New REST endpoints
- `GET  /api/model-config` → current saved slots **merged with** the live
  registry (so the UI can render dropdowns: per layer, list attempts with
  `label/stage/status/available`, mark the active one, mark the default).
- `PUT  /api/model-config` → `{ slots }`; validates, persists, returns saved
  state. 400 with per-slot detail on invalid selection.

### 4.3 Inject overrides into existing AI calls
- `/api/generation` (`index.js:930`): after building `payload`, spread in the
  saved generation slots:
  ```js
  const { slots } = await getActiveConfig();
  if (slots.layer_a) payload.layer_a_attempt = slots.layer_a;
  // … b, c, d
  ```
- `/api/analysis` (`index.js:1005`): currently raw-pipes multipart, which makes
  appending form fields awkward. **Recommended:** add attempt overrides as
  **query-string params** on the forwarded URL and teach `/analysis/run` to
  read them (query takes precedence over the unset Form fields). This keeps the
  zero-buffer streaming pipe intact. Alternative: introduce `multer`, parse the
  upload, and re-emit a new multipart body with the Form fields — heavier.

### 4.4 Per-attempt direct routes unchanged
`/api/layers/:layer/attempts/:attempt/generate|analyze` already target a
specific attempt by URL — leave them as the dev/manual escape hatch.

---

## 5. AI server changes (`acoustic_ai/server/server.py`)

Minimal. If §4.3 uses query-string overrides for analysis, add optional query
params to `/analysis/run` mirroring the existing `Form` fields and prefer them
when present. Generation needs **no** change — `OrchestratedGenerationRequest`
already carries `layer_*_attempt`. Registry `default:` stays as fallback.

---

## 6. Migrations

No migration runner exists in the repo — migrations are applied **by hand**,
and `db_init.sql` only runs on a fresh DB volume. There are **two** init files
that both need the §3 DDL:
- `services/dev/db_init.sql` (local Docker, includes the test-user seed)
- `services/server-a/db_init.sql` (production Server A, no seed)

Deliverable:
1. Append the §3 DDL (all idempotent `IF NOT EXISTS`) to **both** files so
   fresh volumes get the tables.
2. Add a standalone `services/migrations/00X_model_configs.sql` with the same
   idempotent DDL, to be run by hand against the **existing** Server A
   Postgres (and any existing dev volume) since `db_init.sql` won't re-run.
3. Document the one-liner to apply it (`psql … -f …` against the Server A DB).

---

## 7. Frontend — Settings page

New route `frontend/src/pages/SettingsPage.jsx` (+ nav entry in `App.jsx`):
- Two sections, **Generation** and **Analysis**, one labelled dropdown per slot.
- Populate from `GET /api/model-config`: options show `label · stage · status`;
  unavailable attempts disabled with the `unavailable_reason` tooltip; the
  registry default tagged "(default)"; current selection preselected.
- Save → `PUT /api/model-config`; surface per-slot validation errors inline.
- "Reset to defaults" clears slots (revert to registry defaults).
- Reuse the registry-fetch + attempt-shape logic already in
  `lib/api.js` / `DevAnalysisPage.jsx`.

---

## 8. Work breakdown

1. Schema: `db_init.sql` + migration file (§3, §6).
2. Backend `modelConfig.js` + `GET/PUT /api/model-config` + validation (§4.1–4.2).
3. Backend: inject overrides into `/api/generation` and `/api/analysis` (§4.3).
4. AI server: optional query-param overrides on `/analysis/run` (§5).
5. Frontend `SettingsPage.jsx` + nav (§7).
6. Tests: validation rejects unknown/unavailable/head-mismatch; generation
   round-trips the chosen attempt into `metadata.orchestration.attempts`
   (already echoed back — `registry.py:608`); analysis echoes `model_lineage`.

---

## 9. Resolved decisions

1. **Scope** → **Global** single config row. Per-user-ready schema, but one
   global active config for now.
2. **Topology** → Server A (Postgres + Express backend) owns the config and
   injects overrides into its calls to Server B (AI service). Confirms the
   backend-owned design in §2.1. See §2.2a.
3. **Migrations** → No runner; applied by hand. Append DDL to **both**
   `db_init.sql` files + ship a standalone hand-run migration for existing
   DBs. See §6.
4. **Analysis override transport** → **Query-string** params on `/analysis/run`
   (keeps the zero-buffer streaming pipe; no `multer`). See §4.3 / §5.
5. **Who can edit** → **Anyone** for now (no role gate); auth comes later.
