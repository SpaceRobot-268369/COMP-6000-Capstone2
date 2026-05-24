# Milestone 1/2 Status

## Branch

Current working branch:

```text
infra/songke/server-a-deployment
```

This branch contains the Server A deployment skeleton and Milestone 2 backend
job-system MVP work. It has been pushed to:

```text
origin/infra/songke/server-a-deployment
```

## Milestone 1 - Server A Deployment

### Completed

- Separated development and production deployment configuration:

```text
services/dev/
services/prod/
```

- Added production Docker setup for:

```text
PostgreSQL
Express backend
Vite frontend build
nginx reverse proxy
```

- Added production Dockerfiles:

```text
backend/Dockerfile.prod
frontend/Dockerfile.prod
```

- Added nginx configuration:

```text
services/prod/nginx/default.conf
frontend/nginx.prod.conf
```

- Added `.env.example` files and removed real `.env` files from git tracking.
- Restored backend session auth middleware.
- Added Server A deployment runbook.
- Installed Docker and Docker Compose on Server A.
- Cloned the deployment branch to Server A:

```text
/home/ubuntu/COMP-6000-Capstone2-app
```

- Started the production compose stack on Server A:

```text
postgres
backend
frontend
nginx
```

- Verified Server A local HTTP checks:

```text
http://localhost/
http://localhost/api/health
```

- Verified SSH local port forwarding:

```bash
ssh -L 8080:localhost:80 spacerobot-268369
```

Then verified from the developer machine:

```text
http://localhost:8080
http://localhost:8080/api/health
```

### Current Blockers

- RONIN/cloud inbound TCP 80 is not open to the public internet.
- RONIN/cloud inbound TCP 443 is not open yet.
- Public domain HTTP access is blocked until RONIN/admin updates network rules.
- HTTPS is not configured yet.
- Server A reports `System restart required`; schedule a reboot before long-term use.

### Temporary Settings

The current Server A `.env` contains smoke-test settings:

```text
SESSION_COOKIE_SECURE=false
WORKER_API_TOKEN=server-a-worker-test-token
```

These are acceptable for HTTP-only smoke testing through localhost or SSH
tunneling. Before production exposure:

- replace `WORKER_API_TOKEN` with a strong secret;
- replace all other placeholder secrets;
- set `SESSION_COOKIE_SECURE=true`;
- set `FRONTEND_URL` to the final HTTPS URL.

## Milestone 2 - Job System

### Completed

- Added Job System contract documentation.
- Added `jobs` table schema to:

```text
services/dev/db_init.sql
services/prod/db_init.sql
```

- Manually patched the existing Server A PostgreSQL volume with the `jobs`
  schema.
- Implemented user-facing APIs:

```text
POST /api/jobs
GET /api/jobs/:id
POST /api/jobs/:id/cancel
```

- Implemented worker APIs:

```text
POST /api/worker/jobs/claim
POST /api/worker/jobs/:id/heartbeat
POST /api/worker/jobs/:id/status
POST /api/worker/jobs/recover-stale
```

- Added `WORKER_API_TOKEN` based worker API auth.
- Added stale job recovery helper for heartbeat expiry.

### Verified On Server A

User job flow:

```text
register user
POST /api/jobs -> queued
GET /api/jobs/:id -> queued
```

Worker completion flow:

```text
queued -> claimed -> running -> uploading -> completed
```

Cancel flow:

```text
queued -> cancelled
queued -> claimed -> cancel_requested -> cancelled
```

Stale recovery flow:

```text
claimed stale job with attempts remaining -> queued
running stale job with attempts exhausted -> failed
```

All tests were performed through Server A nginx against PostgreSQL.

## Not Completed Yet

Milestone 1 remaining:

- public RONIN inbound 80/443;
- HTTPS;
- final production secrets;
- reboot and post-reboot recovery check.

Milestone 2 remaining:

- frontend job UI;
- automated stale-job scheduler;
- richer retry history / attempt audit table;
- worker registry table;
- real artifact storage integration;
- production cleanup of smoke-test data.

Milestone 3 not started:

- GPU worker runtime;
- Python worker loop;
- DVC/S3 worker setup;
- real generation execution;
- artifact upload from Server B.

## Recommended Next Steps

1. Ask RONIN/admin to open inbound TCP 80 and 443 for Server A.
2. Configure HTTPS and switch Server A `.env` back to secure cookie mode.
3. Replace temporary secrets.
4. Start Milestone 3 with a minimal Python worker loop that claims a job,
   sends heartbeats, writes placeholder artifacts, and marks the job completed.
