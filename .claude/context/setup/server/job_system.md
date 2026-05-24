# Server Job System Contract

## Purpose

The job system is the contract between Server A (`spacerobot-268369`) and
Server B (`shinypokemon`).

Server A is the source of truth. It accepts user requests, stores job state in
PostgreSQL, exposes job APIs, records artifacts, and decides retries.

Server B is a disposable GPU worker. It claims jobs through Server A APIs, runs
generation or training work, uploads durable outputs, and reports status back to
Server A.

## MVP Scope

Milestone 2 should implement the smallest reliable job loop:

1. Frontend/backend creates a job.
2. Job enters `queued`.
3. Worker claims one queued job.
4. Worker sends heartbeat while active.
5. Worker updates status through `running`, `uploading`, and final state.
6. Server A records final result or failure.

Server A should not SSH into Server B for job execution in the MVP. Server B
should not connect directly to PostgreSQL.

## Job Types

MVP job types:

| Type | Meaning |
|---|---|
| `generation` | User-facing soundscape generation |
| `training` | Longer-running model training or fine-tuning |

Generation jobs should have priority over training jobs because users wait on
them interactively.

## Job Statuses

| Status | Meaning |
|---|---|
| `queued` | Server A accepted the job; no worker owns it |
| `claimed` | Worker leased the job but has not started heavy work |
| `running` | Worker is executing generation or training |
| `uploading` | Worker finished local work and is syncing durable artifacts |
| `completed` | Artifacts are durable and result metadata is recorded |
| `failed` | Job ended with an error |
| `cancel_requested` | User or system requested cooperative cancellation |
| `cancelled` | Worker stopped cleanly after cancellation |

## Status Transitions

Allowed MVP transitions:

```text
queued -> claimed
claimed -> running
running -> uploading
uploading -> completed

claimed -> failed
running -> failed
uploading -> failed

queued -> cancelled
claimed -> cancel_requested
running -> cancel_requested
uploading -> cancel_requested
cancel_requested -> cancelled
cancel_requested -> failed
```

Retry creates a new attempt by moving an eligible failed or stale job back to
`queued` after incrementing `attempt_count`. The original job row remains the
source of truth for attempt count and final state.

## Jobs Table

Initial PostgreSQL schema should include:

| Column | Type | Purpose |
|---|---|---|
| `id` | `BIGSERIAL PRIMARY KEY` | Job identifier |
| `type` | `TEXT NOT NULL` | `generation` or `training` |
| `status` | `TEXT NOT NULL` | Current job status |
| `priority` | `INTEGER NOT NULL DEFAULT 0` | Higher value claims first |
| `payload` | `JSONB NOT NULL DEFAULT '{}'::jsonb` | User request / model params |
| `result` | `JSONB NOT NULL DEFAULT '{}'::jsonb` | Completion metadata |
| `artifact_uri` | `TEXT` | Primary durable result URI or key |
| `log_uri` | `TEXT` | Durable worker log URI or path |
| `error_message` | `TEXT` | Short failure reason |
| `claimed_by` | `TEXT` | Worker id |
| `claimed_at` | `TIMESTAMPTZ` | Claim timestamp |
| `heartbeat_at` | `TIMESTAMPTZ` | Last worker heartbeat |
| `started_at` | `TIMESTAMPTZ` | Heavy work start time |
| `finished_at` | `TIMESTAMPTZ` | Terminal state timestamp |
| `attempt_count` | `INTEGER NOT NULL DEFAULT 0` | Number of claims/attempts |
| `max_attempts` | `INTEGER NOT NULL DEFAULT 3` | Retry cap |
| `created_by` | `INTEGER REFERENCES users(id)` | User who created the job |
| `created_at` | `TIMESTAMPTZ NOT NULL DEFAULT NOW()` | Creation timestamp |
| `updated_at` | `TIMESTAMPTZ NOT NULL DEFAULT NOW()` | Last update timestamp |

Recommended indexes:

```sql
CREATE INDEX idx_jobs_status_priority_created
    ON jobs (status, priority DESC, created_at ASC);

CREATE INDEX idx_jobs_claimed_by
    ON jobs (claimed_by);

CREATE INDEX idx_jobs_heartbeat
    ON jobs (heartbeat_at);
```

MVP can use SQL init files because the project does not yet have a migration
system. A later milestone should introduce migrations before schema changes
become frequent.

Current schema locations:

```text
services/dev/db_init.sql
services/prod/db_init.sql
```

These SQL init files run only when a PostgreSQL data volume is created for the
first time. Existing dev or production databases need a manual migration,
temporary schema patch, or volume rebuild to receive later schema changes.

## User API

User-facing APIs require normal session auth.

### `POST /api/jobs`

Creates a job.

Request:

```json
{
  "type": "generation",
  "payload": {
    "seed": 42
  }
}
```

Response:

```json
{
  "ok": true,
  "job": {
    "id": 1,
    "type": "generation",
    "status": "queued"
  }
}
```

### `GET /api/jobs/:id`

Returns job state and visible result metadata for the owner.

### `POST /api/jobs/:id/cancel`

Cancels a job owned by the current user.

MVP behavior:

- `queued` jobs move directly to `cancelled` because no worker owns them.
- `claimed`, `running`, and `uploading` jobs move to `cancel_requested`.
- `completed`, `failed`, and `cancelled` jobs cannot be cancelled.

Workers should check for `cancel_requested` between long-running steps and then
report `cancelled` or `failed`.

## Worker API

Worker APIs require a worker token, separate from user session auth.

Header:

```text
Authorization: Bearer <WORKER_API_TOKEN>
```

Server A verifies the token against:

```env
WORKER_API_TOKEN=...
```

### `POST /api/worker/jobs/claim`

Claims one eligible job atomically.

Request:

```json
{
  "worker_id": "shinypokemon-a100-001",
  "types": ["generation"],
  "capabilities": {
    "gpu": "a100",
    "vram_gb": 40
  }
}
```

Response when a job is available:

```json
{
  "ok": true,
  "job": {
    "id": 1,
    "type": "generation",
    "status": "claimed",
    "payload": {}
  }
}
```

Response when no job is available:

```json
{
  "ok": true,
  "job": null
}
```

Claim must be atomic. Use a PostgreSQL transaction and row locking, such as
`FOR UPDATE SKIP LOCKED`, so two workers cannot execute the same job.

Claim side effects:

- `status = 'claimed'`
- `claimed_by = worker_id`
- `claimed_at = NOW()`
- `heartbeat_at = NOW()`
- `attempt_count = attempt_count + 1`

### `POST /api/worker/jobs/:id/heartbeat`

Updates `heartbeat_at` for the claimed worker.

Request:

```json
{
  "worker_id": "shinypokemon-a100-001"
}
```

Server A should reject the heartbeat if the job is not owned by that worker.

### `POST /api/worker/jobs/:id/status`

Updates status and metadata.

Request:

```json
{
  "worker_id": "shinypokemon-a100-001",
  "status": "running",
  "result": {},
  "artifact_uri": null,
  "log_uri": null,
  "error_message": null
}
```

Server A must validate transitions. Workers cannot arbitrarily move jobs across
unrelated states.

## Retry Policy

MVP retry policy:

- Retry is controlled by Server A, not the worker.
- Worker reports `failed` with `error_message` and optional `log_uri`.
- If `attempt_count < max_attempts`, Server A may move the job back to
  `queued`.
- If attempts are exhausted, the job remains `failed`.

Automatic retry can be implemented as an explicit backend helper first. A
scheduled retry sweeper can be added later.

## Heartbeat and Lease Policy

Workers should send heartbeat every 30 seconds while in `claimed`, `running`, or
`uploading`.

Server A should treat a job as stale if:

```text
status IN ('claimed', 'running', 'uploading')
AND heartbeat_at < NOW() - INTERVAL '5 minutes'
```

MVP stale handling:

- If attempts remain, move stale job back to `queued`.
- If attempts are exhausted, mark `failed`.
- Record an error message such as `worker heartbeat expired`.

Do not reclaim jobs in `uploading` too aggressively; a later version may use a
longer upload timeout.

## Server A Smoke Test

Verified on Server A (`spacerobot-268369`) on 2026-05-24:

- Deployment branch was updated to:

```text
656e96d fix: allow http session smoke tests
```

- Existing production PostgreSQL volume was manually patched with the `jobs`
  schema from `services/prod/db_init.sql`.
- `WORKER_API_TOKEN=server-a-worker-test-token` was added to
  `services/prod/.env` for smoke testing.
- `SESSION_COOKIE_SECURE=false` was added temporarily so HTTP-only localhost
  tests can preserve session cookies before HTTPS is configured.
- Backend was rebuilt and restarted with:

```bash
docker compose up -d --build backend
```

- `/api/health` returned `ok: true` and `db: connected`.
- A smoke-test user was registered through `POST /api/register`.
- `POST /api/jobs` created job `1` with status `queued`.
- `GET /api/jobs/1` returned the created job for the owning session.
- Worker claim succeeded:

```text
queued -> claimed
claimed_by = manual-test-worker
attempt_count = 1
```

- Worker heartbeat updated `heartbeat_at`.
- Worker status updates succeeded:

```text
claimed -> running -> uploading -> completed
```

- Final completed metadata:

```text
artifact_uri = s3://test/generated/job-1.wav
log_uri = s3://test/logs/job-1.log
result = {"duration_s":10,"sample_rate":16000}
finished_at = set
```

- Cancel API was later deployed and smoke-tested on Server A at:

```text
1ff3daf feat: add job cancel API
```

- Queued cancel path succeeded:

```text
job 2: queued -> cancelled
finished_at = set
```

- Active cancel path succeeded:

```text
job 3: queued -> claimed -> cancel_requested -> cancelled
claimed_by = manual-cancel-worker
finished_at = set
```

This confirms the Milestone 2 MVP API flow works on Server A through nginx and
PostgreSQL. The test used placeholder artifact URIs and a temporary worker
token; production deployment must replace the token and switch
`SESSION_COOKIE_SECURE=true` after HTTPS is configured.

## Artifact Contract

Workers must upload artifacts before marking a job `completed`.

MVP artifact fields:

| Field | Meaning |
|---|---|
| `artifact_uri` | Primary output, such as S3 key, DVC path, or durable URL |
| `log_uri` | Worker log path or object key |
| `result` | JSON metadata: duration, sample rate, model path, seed, explanation |

Server A stores artifact references. It should not receive large generated
files directly in the MVP.

## Security

- User APIs require session auth.
- Worker APIs require `WORKER_API_TOKEN`.
- Worker token must never be exposed to the browser.
- Server B should use HTTPS once Server A has HTTPS configured.
- Do not allow arbitrary users to claim jobs or update job status.

## MVP Limitations

The first implementation may intentionally skip:

- worker registry table;
- admin dashboard;
- scheduled stale-job sweeper;
- multi-worker capability matching;
- detailed attempt history table;
- artifact signing or temporary download URLs.

These can be added once the single-worker flow is stable.
