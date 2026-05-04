# COMP-6000 Capstone 2 — Speculative Soundscape Generation

## Project Overview

A **research-oriented prototype** that explores AI-driven speculative soundscape generation using ecoacoustic recordings and environmental data. The system learns relationships between environmental conditions and soundscape structure, then generates plausible soundscapes under altered conditions.

**Sits at the intersection of:**
- Ecoacoustics
- Environmental data science
- AI-based audio modeling
- Creative sound practice

**Core concept:** `Acoustic recordings + Environmental variables + AI modeling → Speculative soundscape generation`

---

## System Architecture

### Components

| # | Component | Tech | Status |
|---|-----------|------|--------|
| 1 | **Frontend** | React + Vite (`frontend/`) | UI scaffold done |
| 2 | **Backend** | Express.js + PostgreSQL (`backend/`) | Auth endpoints done |
| 3 | **Environmental Data Module** | Python (`acoustic_ai/`) | Not yet started |
| 4 | **Acoustic Data Module** | Python (`acoustic_ai/`) | Not yet started |
| 5 | **AI Modeling Module** | Python/PyTorch (`acoustic_ai/`) | Not yet started |
| 6 | **LLM Interface** (optional) | TBD | Not yet started |
| 7 | **Metadata Database** (optional) | PostgreSQL | Not yet started |

### Three Processing Pipelines

**1. Soundscape Analysis Pipeline**
```
Input Audio → Spectrogram → Feature Extraction / Encoder → Soundscape Embedding
→ Environmental Data Alignment → acoustic features, correlations, summary
```

**2. Speculative Soundscape Generation Pipeline**
```
Environmental Conditions + Site Info + Optional Reference Audio
→ Condition Encoding → Generative Model → Spectrogram → Audio Reconstruction → Output
```

**3. Soundscape Transformation Pipeline**
```
Input Audio → Embedding → User-defined Environmental Changes
→ Conditioned Generative Model → New Soundscape → Audio Output
```

### AI Modeling Approach — Layered Soundscape System

A soundscape is treated as a layered composition, not a single generated waveform:

```
speculative soundscape = ambient site bed (Module A)
                       + weather layer    (Module B)
                       + event layer      (Module C)
                       + final mix        (Module D)
```

**Generation pipeline:**
```
env conditions → Module A: ambient retrieval (NN search in latent_clips.npy)
              → Module B: weather asset mixing (wind/rain → gain/EQ)
              → Module C: event scheduling (annotation/BirdNET snippets)
              → Module D: layer combiner → WAV + spectrogram + explanation JSON
```

**Analysis pipeline:**
```
uploaded audio → Module E: ambient similarity (VAE latent NN)
              → Module E: weather detector (spectral heuristics / classifier)
              → Module E: event detector (BirdNET / annotation lookup)
              → analysis report (estimated conditions + layer breakdown)
```

### AI Module Details

| Module | Role | Status | Code location |
|---|---|---|---|
| A — Ambient | VAE encoder + NN retrieval for ambient bed | VAE trained (30 epochs) | `acoustic_ai/modules/ambient/` |
| B — Weather | Curated wind/rain assets + parameter mixing | Placeholder | `acoustic_ai/modules/weather/` |
| C — Events | Annotation audit + event snippets + scheduler | Placeholder | `acoustic_ai/modules/events/` |
| D — Mixer | Combine A+B+C → WAV + explanation JSON | Placeholder | `acoustic_ai/modules/mixer/` |
| E — Analysis | Ambient similarity + weather + event detectors | Partial (A working) | `acoustic_ai/modules/analysis/` |

**Vocoder:** Ecoacoustic HiFi-GAN trained on Site 257 audio (128-bin, 22,050 Hz).
Checkpoint: `acoustic_ai/checkpoints/vocoder/best.pt` (DVC-tracked).

**VAE checkpoint:** `acoustic_ai/checkpoints/ambient/best.pt` (DVC-tracked).

> Full details: `.claude/context/ai/architecture.md`
> Pipeline design: `.claude/context/ai/pipeline_design.md`
> Decision log: `.claude/context/ai/logs/mvp_decision_log.md`

### Environmental Variables
temperature, humidity, wind speed/direction, rainfall, time of day, season, geographic site

### Data Sources
- Australian Acoustic Observatory
- TERN Ecoacoustic datasets
- Bureau of Meteorology
- TERN EcoPlots

---

## Implementation Plan & Timeline

### Stage 1 — Designing (2 weeks)
**Goal:** Define system architecture, AI pipeline, and feature scope for three modes.

Deliverables:
- System architecture design document
- AI pipeline design (Raw Audio → Mel-spectrogram → Embedding → Output)
- Model selection and technical specification
- Dataset requirements and data collection plan
- Development tools/framework selection (PyTorch, Librosa, etc.)

### Stage 2 — Pilot (3 weeks)
**Goal:** Train AI with a small dataset to verify feasibility.

Deliverables:
- Prototype AI models for audio embedding and feature extraction
- Small experimental dataset prepared and preprocessed
- Working prototype of the audio processing pipeline
- Preliminary results for generation and analysis
- Feasibility evaluation report

### Stage 3 — Implementation (5 weeks)
**Goal:** Full system — trained models with larger datasets, three AI modes integrated.

Deliverables:
- Fully trained AI models for audio generation and analysis
- All three modes: Analysis, Generation, Transformation
- Backend processing pipeline and APIs
- Model training and optimization results
- Integrated system prototype

### Stage 4 — Interaction Refinement (2 weeks)
**Goal:** Improve usability and system performance.

Deliverables:
- Refined UI for audio input and editing
- Improved natural language interaction for modifying audio parameters
- System performance optimization (latency and quality)
- User testing and feedback report
- Final system version ready for deployment

---

## Key Design Principles

1. **Multimodal Integration** — acoustic data + environmental variables are central
2. **Modular Architecture** — loosely coupled so models can be swapped independently
3. **Research-Oriented Flexibility** — supports experimentation and methodology comparison
4. **Ethical Awareness** — responsible use of ecoacoustic data; transparency about model limits

---

## Hardware Requirements

### Minimum
- OS: macOS 12+
- CPU: 4-core (Intel Core i5 10th gen / AMD Ryzen 5)
- GPU: NVIDIA GTX 1660 / RTX 2060, 6 GB VRAM
- RAM: 32 GB
- Storage: 1 TB SSD

### Recommended
- OS: macOS 13+
- CPU: 8-core (Intel Core i7 / Apple Silicon M2 Pro)
- GPU: NVIDIA RTX 3080 / RTX 4070 Ti, 12–16 GB VRAM
- RAM: 128 GB
- Storage: 2 TB NVMe SSD

---

## Development Setup (macOS)

### Prerequisites
```bash
brew install docker          # Docker Desktop handles compose
# or install Docker Desktop from https://www.docker.com/products/docker-desktop/
```

### Running with Docker Compose

Docker runs PostgreSQL, backend, and frontend only.
**The AI server must run natively** — Docker cannot access the macOS GPU (MPS).

```bash
# Start postgres + backend + frontend
docker compose -f services/dev/docker-compose.yml up
```

Services started:
- **Frontend** → http://localhost:5173
- **Backend API** → http://localhost:4000
- **PostgreSQL** → localhost:5432

### Running the AI server natively (required for GPU)

```bash
cd acoustic_ai
pip install -r requirements.txt
uvicorn server.server:app --reload --port 8000
```

The backend connects to the AI server at `http://localhost:8000`.

### Running other services natively

**PostgreSQL** (via Docker only):
```bash
docker compose -f services/dev/docker-compose.yml up postgres -d
```

**Backend:**
```bash
cd backend
DATABASE_URL=postgresql://capstone_user:<password>@localhost:5432/capstone_dev PORT=4000 npm run dev
```

**Frontend:**
```bash
cd frontend
VITE_API_URL=http://localhost:4000 npm run dev
```

---

## Project File Structure

```
COMP-6000-Capstone2/
├── frontend/                    # React + Vite UI
├── backend/                     # Express.js API
├── acoustic_ai/                 # Python AI module (runs natively for GPU)
│   ├── modules/
│   │   ├── ambient/             # Module A: VAE + retrieval
│   │   ├── weather/             # Module B: weather assets + mixing
│   │   ├── events/              # Module C: annotation + event scheduling
│   │   ├── mixer/               # Module D: layer combiner
│   │   └── analysis/            # Module E: analysis explainer
│   ├── precompute/              # One-off data prep scripts
│   ├── data/                    # DVC-tracked pipeline artifacts
│   │   ├── shared/              # Shared wavs + spectrograms
│   │   ├── ambient/latents/     # Latent clip database
│   │   ├── weather/             # Weather assets + asset_index.csv
│   │   ├── events/              # Event index + snippets
│   │   └── analysis/            # Analysis module data
│   ├── checkpoints/
│   │   ├── ambient/best.pt      # VAE checkpoint (DVC)
│   │   └── vocoder/best.pt      # HiFi-GAN checkpoint (DVC)
│   └── server/                  # FastAPI server
│       ├── server.py            # FastAPI entry point
│       └── inference.py         # Inference helpers
├── resources/                   # Raw source data (DVC-tracked)
│   └── site_257_bowra-dry-a/
│       ├── site_257_filtered_items.csv    (git)
│       ├── site_257_env_data.csv          (git)
│       ├── site_257_training_manifest.csv (git)
│       ├── downloaded_clips/              (DVC, 125 GB)
│       └── downloaded_annotations/        (DVC)
├── services/dev/                # Docker Compose (postgres + backend + frontend only)
├── script/                      # Data download scripts
├── dvc.yaml                     # DVC pipeline stages
├── params.yaml                  # Tracked hyperparameters
├── Makefile                     # git+dvc convenience commands
└── .claude/                     # Claude Code context and settings
    └── context/
        ├── ai_module_architecture.md
        ├── generation_layers.md
        ├── analysis_components.md
        └── ai_mvp_decision_log_and_new_architecture.md
```

### Storage Rule

> **All Claude-related context files must live under `.claude/`** — not the project root.
>
> | Type | Location |
> |------|----------|
> | Architecture and design docs | `.claude/context/` |
> | Known issues, decisions, notes | `.claude/context/` |
> | Claude Code settings | `.claude/settings.local.json` |

---

## Git + DVC Collaboration

Git tracks code and small text files. DVC tracks large binary artifacts (audio, spectrograms, model weights, latent databases). They work together so every git branch carries a complete, reproducible snapshot of both code and data.

### How it works

```
git commit  →  .dvc pointer files committed (tiny text, ~100 bytes each)
               actual binary data stored in local cache (~/.dvc-store/capstone2)

git checkout <branch>  →  post-checkout hook fires
                           dvc checkout runs automatically
                           binary files swapped to match the branch's .dvc pointers
```

Each `.dvc` file in the repo is a pointer — it stores the content hash and size of the real artifact. The actual bytes live in the cache, never in git.

### Automatic git hooks

All four hooks were installed by `dvc install` and fire without any manual step:

| Git action | Hook | DVC action |
|---|---|---|
| `git checkout <branch>` / `git switch` | `post-checkout` | `dvc checkout` — swaps data files to match the new branch |
| `git pull` / `git merge` | `post-merge` | `dvc checkout` — syncs data after incoming commits change `.dvc` files |
| `git commit` | `pre-commit` | warns if tracked data was modified but not staged with `dvc add` |
| `git push` | `pre-push` | `dvc push` — copies new/changed artifacts into local cache before code push |

### Local cache

All data is stored outside the repo:

```
~/.dvc-store/capstone2/   ← local DVC cache (set in .dvc/config)
```

DVC deduplicates by content hash — a file used on two branches is stored once. Branches share the cache.

### Typical branch workflow

```bash
# Start a new experiment
git checkout -b experiment/beta-kl-0.05
# post-checkout fires → dvc checkout syncs data for this branch (same as main initially)

# Change a hyperparameter and re-run the pipeline
vim params.yaml
python3 -m dvc repro          # only re-runs stages whose inputs changed
git add .
git commit -m "experiment: higher beta KL"
# pre-commit fires → warns if any DVC-tracked file is dirty
git push
# pre-push fires → dvc push copies new checkpoint to local cache

# Switch back — everything restores automatically
git checkout main
# post-checkout fires → dvc checkout restores main's best.pt, latents, etc.
```

### DVC pipeline (`dvc.yaml`)

Defines reproducible stages. `dvc repro` re-runs only stages whose deps or params changed.

| Stage | Command | Key outputs |
|---|---|---|
| `precompute_spectrograms` | `precompute/precompute_spectrograms.py` | `data/shared/wavs/`, `data/shared/spectrograms/` |
| `train_vae` | `modules/ambient/train.py` | `checkpoints/ambient/best.pt` |
| `precompute_latents` | `precompute/precompute_latents.py` | `data/ambient/latents/latent_clips.npy`, `latent_templates.npy` |
| `train_vocoder` | `modules/ambient/train_vocoder.py` | `checkpoints/vocoder/best.pt` |

Hyperparameters that affect which stages re-run are tracked in `params.yaml`.
Compare params between branches: `python3 -m dvc params diff main`.

### Makefile shortcuts

The `Makefile` wraps the most common combined git+dvc operations:

```bash
make branch b=<name>   # git checkout <name> + dvc checkout
make push              # git push + dvc push
make pull              # git pull + dvc pull
make repro             # dvc repro (re-run changed pipeline stages)
make diff              # git diff + dvc params diff
make status            # git status + dvc status
make ai                # start AI server locally on port 8000
```

### What is tracked where

| Artifact | Tracked by | Location |
|---|---|---|
| Source code, scripts, configs | git | everywhere in repo |
| Small CSVs (manifests, env data) | git | `resources/site_257_bowra-dry-a/*.csv` |
| `.dvc` pointer files | git | alongside DVC-tracked artifacts |
| `dvc.yaml`, `params.yaml`, `dvc.lock` | git | project root |
| Model checkpoints (`best.pt`) | DVC | `acoustic_ai/checkpoints/*/` |
| Latent databases (`.npy`) | DVC | `acoustic_ai/data/ambient/latents/` |
| Shared wavs + spectrograms | DVC | `acoustic_ai/data/shared/` |
| Weather assets | DVC | `acoustic_ai/data/weather/weather_assets/` |
| Event snippets | DVC | `acoustic_ai/data/events/event_snippets/` |
| Raw audio clips (125 GB) | DVC | `resources/site_257_bowra-dry-a/downloaded_clips/` |
| Raw annotations | DVC | `resources/site_257_bowra-dry-a/downloaded_annotations/` |

### Fresh clone setup

On a new machine, after `git clone`:

```bash
# 1. Install DVC
pip3 install dvc

# 2. Configure the local cache path (must match where data was pushed)
python3 -m dvc remote add local_cache /path/to/your/dvc-store/capstone2
python3 -m dvc remote default local_cache

# 3. Pull all tracked data
python3 -m dvc pull

# 4. Re-install git hooks (hooks live in .git/, not committed)
python3 -m dvc install
# Fix hooks to use python3 -m dvc (if dvc is not on PATH)
sed -i '' 's/exec dvc /exec python3 -m dvc /g' .git/hooks/pre-commit .git/hooks/pre-push .git/hooks/post-checkout
# Add post-merge hook manually
echo '#!/bin/sh\npython3 -m dvc git-hook post-checkout $@' > .git/hooks/post-merge
chmod +x .git/hooks/post-merge
```

> **Note:** `dvc` may not be on `PATH` on macOS when installed via `pip3`. All commands in this project use `python3 -m dvc` explicitly. The git hooks are patched to do the same (see above).

### Pre-Commit File Audit

Before every commit, check whether any unintended files are being tracked by git.

1. Run `git status` and inspect the staged and untracked file lists.
2. If any file appears that should not be committed (large binaries, generated outputs, credentials, editor artefacts, OS files, etc.), **do not commit yet**.
3. Add the offending path(s) to `.gitignore` (and run `git rm --cached <path>` if the file is already tracked).
4. Verify `git status` is clean of unintended files before proceeding with the commit.

This check is mandatory — never skip it, even for "quick" commits.

---

## Backend API

Current endpoints:
- `GET  /api/health` — DB connectivity check
- `POST /api/register` — user registration
- `POST /api/login` — user login

Planned (to be added in Stage 3):
- `POST /api/analysis` — submit audio for soundscape analysis
- `POST /api/generation` — generate soundscape from environmental params
- `POST /api/transformation` — transform audio with new environmental conditions

---

## MVP Dataset — `site_257_filtered_items.csv`

Generated by `script/sample_mvp_dataset.py` using the complete-day diel sampling policy (seed=42).
Tracked at `resources/site_257_bowra-dry-a/site_257_filtered_items.csv`.

### Contents

- **287 recordings** sampled from 12,251 in the full archive
- **73 unique local dates** (AEST), spanning **2019-08-14 to 2025-06-20**
- **38 year-months** covered (of 72 possible) — gaps mainly 2021–2022
- **~510 hours** of audio, **~43.8 GB** total download size

### Diel Bin Distribution

| Bin | Count | Window (AEST) |
|-----|-------|--------------|
| Dawn | 72 | 05:00–07:00 |
| Morning | 72 | 08:00–10:00 |
| Afternoon | 72 | 13:00–15:00 |
| Night | 71 | 22:00–00:00 |

### Year Coverage

| Year | Recordings | Notes |
|------|-----------|-------|
| 2019 | 40 | Aug–Dec (site start) |
| 2020 | 42 | Jan–Apr, Aug–Sep |
| 2021 | 1 | Mar only — heavy archive gap |
| 2022 | 0 | No recordings in archive |
| 2023 | 60 | Apr–Dec |
| 2024 | 96 | Full year |
| 2025 | 48 | Jan–Jun |

### Audio Properties

- Format: FLAC, mono, 22,050 Hz — all 287 files `ready`
- Duration: avg ~1.8 hrs, max ~2 hrs, min 76 s

### Notable Gap
2021–2022 is almost entirely absent from the source archive (1 recording only). This is a data availability gap in the A2O source, not a sampling artifact — the model will have limited coverage of those years.

### Downloading the Sample
```bash
python3 script/download_site_257_originals.py \
  --csv-path resources/site_257_bowra-dry-a/site_257_filtered_items.csv \
  --start-item 1 --end-item 9999 --workers 6
```

---

## Data Commands

### Download annotations for MVP sample
```bash
python3 script/download_site_257_annotations.py \
  --csv-path resources/site_257_bowra-dry-a/site_257_filtered_items.csv \
  --start-item 1 \
  --end-item 287 \
  --workers 6
```

### Fetch environmental data for MVP sample
```bash
python3 script/fetch_nasa_env_data.py
```
Output: `resources/site_257_bowra-dry-a/site_257_env_data.csv`

### Download clips for MVP sample
```bash
python3 script/download_site_257_clips.py \
  --csv-path resources/site_257_bowra-dry-a/site_257_filtered_items.csv \
  --start-item 1 \
  --end-item 287 \
  --workers 6
```

---

## ⚠️ Known Data Issues

> Full details in `.claude/context/issues/known_issues.md`

### Unrecoverable Clips — DO NOT RE-DOWNLOAD

**12 clips permanently fail with `422 Unprocessable Entity` from the A2O API.**
These are corrupted/missing on the server side. Retrying will always fail.

| CSV Count | Recording ID | Clip |
|-----------|-------------|------|
| 216 | 1678484 | 021 |
| 219 | 1678513 | 006 |
| 222 | 1681319 | 009 |
| 248 | 1679394 | 024 |
| 249 | 1676521 | 005 |
| 252 | 1676441 | 021 |
| 254 | 1676444 | 018 |
| 256 | 1670355 | 024 |
| 266 | 1672094 | 011 |
| 268 | 1681455 | 001 |
| 270 | 1676142 | 023 |
| 281 | 1672466 | 011 |

**Impact:** 12 / 6,160 clips (0.2%) — negligible for training. Each affected recording still has the majority of its clips. **Exclude these from the training pipeline.**

---

## Notes

- Avoid excessive filtering/denoising of audio — anthropogenic noise is authentic soundscape data
- Data representations should be **learned** (spectrogram → encoder → embedding), not hand-crafted parameters
- Prototype stage may use pre-trained models and reduced datasets
