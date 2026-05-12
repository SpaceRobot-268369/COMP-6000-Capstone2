# Services & Local Development Setup

## Service topology

| Service | How it runs | URL |
|---------|-------------|-----|
| Frontend | Docker | `http://localhost:5173` |
| Backend | Docker | `http://localhost:4000` |
| PostgreSQL | Docker | `localhost:5432` |
| AI server (FastAPI) | **Native only** (GPU/MPS) | `http://localhost:8000` |

Docker cannot access macOS MPS — the AI server **must** run natively.

## Docker Compose (postgres + backend + frontend)

Config lives at `services/dev/docker-compose.yml`; environment at
`services/dev/.env`.

```bash
docker compose -f services/dev/docker-compose.yml up
```

## AI server (native)

Run from the project venv. **Never** use system / Homebrew `python3`, `pip`,
`accelerate`, or `uvicorn` for AI training/inference — they load incompatible
torch/torchaudio builds.

```bash
cd acoustic_ai
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn server.server:app --reload --port 8000
```

Equivalent no-activation form (preferred for one-shot commands):

```bash
./acoustic_ai/.venv/bin/python -m uvicorn server.server:app --reload --port 8000
```

## Environment variables

| Variable | Service | Description |
|---|---|---|
| `DATABASE_URL` | Backend | PostgreSQL connection string |
| `PORT` | Backend | Port to bind (default 4000) |
| `AI_SERVER_URL` | Backend | AI FastAPI URL (default `http://localhost:8000`) |
| `VITE_API_URL` | Frontend | Backend base URL for Vite proxy |

## Prerequisites

```bash
brew install docker          # Docker Desktop handles compose
# or install Docker Desktop from https://www.docker.com/products/docker-desktop/
```

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
