# Project Services & Scripts

## Docker Compose Services

Config: `services/dev/docker-compose.yml`
Env: `services/dev/.env`

Start all:
```bash
docker compose -f services/dev/docker-compose.yml up
```

| Service | Source | Port | Purpose | Status |
|---------|--------|------|---------|--------|
| `frontend` | `frontend/` (Vite + React) | 5173 | UI — Analysis, Generation, Transformation pages | Scaffold done |
| `backend` | `backend/` (Express.js) | 4000 | REST API — auth + future AI endpoints | Auth done |
| `postgres` | postgres:16-alpine | 5432 | Primary database (users, metadata, experiments) | Running |
| `acoustic_ai` | python:3.11-slim | — | Python AI container, workspace at `/workspace` | Placeholder only |

---

## Data Scripts (`script/`)

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `fetch_recordings.py` | A2O API | — | Discover available recordings for site 257 |
| `sample_mvp_dataset.py` | `site_257_all_items.csv` | `site_257_filtered_items.csv` | Apply diel stratified sampling policy (seed=42) |
| `download_site_257_originals.py` | filtered CSV | `resources/.../originals/` | Download full FLAC files (~2 hr each) |
| `download_site_257_clips.py` | filtered CSV | `resources/.../clips/` | Download short audio clips |
| `download_site_257_annotations.py` | filtered CSV | `resources/.../downloaded_annotations/` | Download annotation CSVs (species labels) |
| `fetch_nasa_env_data.py` | filtered CSV dates | `site_257_env_data.csv` | Fetch NASA POWER hourly/daily env vars + optional MODIS NDVI |

---

## Resource Files (`resources/site_257_bowra-dry-a/`)

| File | Git tracked | Description |
|------|-------------|-------------|
| `site_257_all_items.csv` | No (gitignored) | Full archive — 12,251 recordings |
| `site_257_filtered_items.csv` | Yes | 287-recording MVP sample (diel stratified) |
| `site_257_env_data.csv` | Yes | NASA POWER env data aligned to MVP sample dates |
| `downloaded_annotations/` | No | Annotation CSVs per recording |
