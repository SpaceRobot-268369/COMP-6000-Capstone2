# Services & Local Development Setup

## Service topology

| Service | How it runs | URL |
|---------|-------------|-----|
| Frontend | Docker | `http://localhost:5173` |
| Backend | Docker | `http://localhost:4000` |
| PostgreSQL | Docker | `localhost:5432` |
| AI tunnel | Docker sidecar | `ai-tunnel:8000` inside Compose |
| AI server (FastAPI) | Native on serverB | `serverB:127.0.0.1:8000` via SSH tunnel |

The Docker backend reaches serverB through the Compose `ai-tunnel` sidecar.
The FastAPI process itself runs natively on serverB. Keep `.pem` files outside
the repository; see `services/dev/README.md` for the current key-path
convention and manual tunnel diagnostics.

## Docker Compose (postgres + ai-tunnel + backend + frontend)

Config lives at `services/dev/docker-compose.yml`; environment at
`services/dev/.env`.

```bash
docker compose -f services/dev/docker-compose.yml up
```

The Compose stack mounts the serverB pem as a read-only secret into `ai-tunnel`
and waits for the tunnel health check before starting the backend.

## AI server (serverB native)

Run the FastAPI server on serverB from the project venv. For an SSH tunnel,
binding to localhost is enough:

```bash
cd acoustic_ai
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn server.server:app --host 127.0.0.1 --reload --port 8000
```

For manual diagnosis from the local machine:

```bash
cd services/dev
./start-ai-tunnel.sh
```

## Environment variables

| Variable | Service | Description |
|---|---|---|
| `DATABASE_URL` | Backend | PostgreSQL connection string |
| `PORT` | Backend | Port to bind (default 4000) |
| `AI_CONNECTION_MODE` | Backend | `ssh_tunnel` in Compose for serverB diagnostics |
| `AI_SERVER_URL` | Backend | AI FastAPI URL (`http://ai-tunnel:8000` in Compose) |
| `AI_SERVER_LABEL` | Backend / tunnel | Human-readable serverB label in status messages |
| `AI_SSH_USER` | Tunnel | SSH username for serverB (default `ubuntu`) |
| `AI_SSH_HOST` | Tunnel | SSH hostname for serverB |
| `AI_TUNNEL_LOCAL_PORT` | Backend / tunnel | Local tunnel port inside Compose (default `8000`) |
| `AI_TUNNEL_REMOTE_HOST` | Backend / tunnel | Remote FastAPI bind host (default `127.0.0.1`) |
| `AI_TUNNEL_REMOTE_PORT` | Backend / tunnel | Remote FastAPI port (default `8000`) |
| `VITE_API_URL` | Frontend | Backend base URL for Vite proxy |

## Prerequisites

```bash
brew install docker          # Docker Desktop handles compose
# or install Docker Desktop from https://www.docker.com/products/docker-desktop/
```

**DVC** must be installed at user-site (not in the venv) so git hooks can call
it without venv activation:

```bash
pip3 install --user 'dvc[s3]'
```

See [.claude/context/dev/dvc_workflow.md](../../dev/dvc_workflow.md) for the
full DVC + S3 setup (AWS profile, remote config, daily commands).

## Running individual services natively (when not using Docker)

Rare — only when iterating outside Docker. The Docker path is canonical.

**Backend:**
```bash
cd backend
DATABASE_URL=postgresql://capstone_user:<password>@localhost:5432/capstone_dev \
PORT=4000 npm run dev
```

**Frontend:**
```bash
cd frontend
VITE_API_URL=http://localhost:4000 npm run dev
```
