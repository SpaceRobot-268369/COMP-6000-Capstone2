# Server A Deployment Runbook

## Purpose

Server A (`spacerobot-268369`) is the always-on application server. It owns the
frontend, Express backend, PostgreSQL database, nginx entrypoint, user auth, and
future job orchestration APIs.

Server B (`shinypokemon`) remains the on-demand GPU worker and is not part of
this compose stack.

## Current Production Layout

Production deployment files live under `services/prod/`:

```text
services/prod/
  docker-compose.yml
  .env.example
  db_init.sql
  nginx/default.conf
```

Production image definitions:

```text
backend/Dockerfile.prod
frontend/Dockerfile.prod
frontend/nginx.prod.conf
```

The existing `services/dev/` stack is for local development and should remain
separate from Server A deployment.

## Prerequisites

- Docker and Docker Compose installed on Server A.
- Repository cloned on Server A.
- DNS points the chosen domain to Server A before HTTPS is configured.
- Fresh production secrets generated. Do not reuse any secret that was ever
  committed in a `.env` file.

## First-Time Setup

From the repository root on Server A:

```bash
cd services/prod
cp .env.example .env
```

Edit `services/prod/.env` and set production values:

```env
POSTGRES_USER=capstone_user
POSTGRES_PASSWORD=<new-production-password>
POSTGRES_DB=capstone_prod

BACKEND_PORT=4000
SESSION_SECRET=<new-production-secret>
SESSION_COOKIE_SECURE=true
JWT_SECRET=<new-production-secret>
APP_SECRET=<new-production-secret>
FRONTEND_URL=https://<server-a-domain>

AI_SERVER_URL=http://shinypokemon:8000
VITE_API_URL=
```

Keep `VITE_API_URL` empty for same-origin browser calls through nginx.

## Start Server A

```bash
cd services/prod
docker compose up -d --build
```

Check container state:

```bash
docker compose ps
```

Check logs:

```bash
docker compose logs -f backend
docker compose logs -f nginx
```

## Verification

Before HTTPS is added:

```bash
curl http://localhost/api/health
```

Expected result:

```json
{"ok":true,"db":"connected",...}
```

From a browser or remote machine, verify:

```text
http://<server-a-domain>/
http://<server-a-domain>/api/health
```

## Local Production Smoke Test

Use this before deploying to Server A to verify the production compose skeleton
locally.

```bash
cd services/prod
cp .env.example .env
# edit .env with local-only test secrets
docker compose up -d --build
docker compose ps
curl http://localhost/api/health
curl -I http://localhost/
docker compose down
```

Expected results:

- `postgres` is healthy.
- `backend`, `frontend`, and `nginx` are running.
- `/api/health` returns `ok: true` and `db: connected`.
- `/` returns HTTP 200 through nginx.

Verified locally on 2026-05-23:

- Backend production image built successfully.
- Frontend production image built successfully after adding the Vite `build`
  script.
- nginx served the frontend through `http://localhost/`.
- nginx proxied `http://localhost/api/health` to the backend.
- Backend reached PostgreSQL successfully.
- The compose stack was stopped with `docker compose down` after verification.

## Server A Verification

Verified on Server A (`spacerobot-268369`) on 2026-05-23:

- Host OS is Ubuntu 24.04.2 LTS.
- Docker 29.1.3 installed.
- Docker Compose 2.40.3 installed.
- `ubuntu` user is in the `docker` group and can run Docker without `sudo`.
- Deployment branch `infra/songke/server-a-deployment` was cloned to:

```text
/home/ubuntu/COMP-6000-Capstone2-app
```

- The branch is at:

```text
267f851 infra: add server a production deployment skeleton
```

- Production `.env` was created from `services/prod/.env.example`.
- `docker compose config` passed on Server A.
- `docker compose up -d --build` successfully started:

```text
postgres   healthy
backend    running
frontend   running
nginx      running on 0.0.0.0:80
```

- Server-side checks passed:

```bash
curl -i http://localhost/api/health
curl -I http://localhost/
```

- `/api/health` returned HTTP 200 with `ok: true` and `db: connected`.
- `/` returned HTTP 200 through nginx.

Public HTTP access is not available yet because external TCP connections to
port 80 are blocked by RONIN/cloud network security rules. Server A itself is
listening on port 80 and `ufw` is inactive, so the remaining blocker is outside
the VM.

Temporary access works through SSH local port forwarding from the developer
machine:

```bash
ssh -L 8080:localhost:80 spacerobot-268369
```

Then open:

```text
http://localhost:8080
http://localhost:8080/api/health
```

This only exposes the site to the local developer machine while the SSH session
is open. Public access still requires RONIN/admin configuration for inbound
HTTP 80 and HTTPS 443.

## Production Auth Note

In production, the Express session cookie is marked `secure` by default because
`NODE_ENV=production`. Keep `SESSION_COOKIE_SECURE=true` for HTTPS deployment.

For temporary HTTP-only smoke tests through localhost or SSH port forwarding,
set this in `services/prod/.env` and restart the backend:

```env
SESSION_COOKIE_SECURE=false
```

HTTP-only checks can always verify `/api/health`, but login and authenticated
browser flows need either HTTPS or this temporary smoke-test override.

## Restart and Stop

Restart:

```bash
cd services/prod
docker compose restart
```

Stop:

```bash
cd services/prod
docker compose down
```

Do not remove volumes unless intentionally deleting the production database.

## Current Production Blockers

This stack is a production deployment skeleton, not yet a final release.

- HTTPS is not configured yet.
- Job system APIs are not implemented yet.
- Worker claim, heartbeat, retry, and idle shutdown are not implemented yet.
- `AI_SERVER_URL` is still a placeholder until Server B networking is decided.
- Database schema is initialized with SQL files, not migrations.
- Old committed `.env` secrets must be treated as exposed and rotated.

## Notes

- Server A should not run the GPU AI server locally.
- Server B should update job state through Server A APIs, not by connecting
  directly to PostgreSQL.
- Generated clips, logs, and model artifacts must be durable before Server B
  shuts down.
