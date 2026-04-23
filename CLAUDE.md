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

### AI Modeling Approach
```
Raw Audio → Mel-Spectrogram → CNN Encoder → Audio Embedding
→ Combine with Environmental Variables → Multimodal Model
→ Latent Soundscape Representation → Conditional Generator
→ Spectrogram → Neural Vocoder → Generated Audio
```

Model types to consider: CNN encoders, Transformers, Conditional diffusion models, GAN/VAE, Neural vocoders.

### AI Module Architecture — A / B / C Pattern

The AI system is split into three independent modules. This separation allows each to be trained, frozen, and extended without disrupting the others.

**Module A — Environmental Autoencoder** (`acoustic_ai/model.py`, current)
- Trained unsupervised on audio + environmental features (no annotations needed)
- Learns: environment ↔ soundscape structure relationship
- Output: latent representation `z` (256-dim)
- Status: pilot training complete (Stage 2)

**Module B — Ecological Classifier** (Stage 3, future)
- Built on top of frozen Module A encoder
- Trained on annotated clips only (sparse annotation coverage is fine)
- Learns: `z → species presence vector` (e.g. probability per species)
- Can be added later without retraining Module A

**Module C — Conditioned Generator** (Stage 3, future)
- Combines Module A's environmental latent + Module B's species signal
- Input: target env conditions + optional target species
- Output: generated spectrogram matching both signals
- Implemented as an extended decoder: `[z, species_vector] → spectrogram`

**Build order for Stage 3:**
1. Freeze Module A weights (`model.requires_grad_(False)`)
2. Train Module B (small MLP head) on annotated clips
3. Build Module C — extend decoder to accept `[z, species_vector]`
4. Fine-tune Module C while keeping A and B frozen

This is the **frozen backbone + task head** pattern — standard ML practice for extending a pretrained base model without losing learned representations.

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

### Running with Docker Compose (recommended)
```bash
# From project root
docker compose -f services/dev/docker-compose.yml up
```

Services started:
- **Frontend** → http://localhost:5173
- **Backend API** → http://localhost:4000
- **PostgreSQL** → localhost:5432
- **acoustic_ai** — Python container (workspace at `/workspace`)

### Running natively (faster iteration)

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

**Acoustic AI (Python):**
```bash
cd acoustic_ai
pip install -r requirements.txt
# then run individual scripts/notebooks
```

### Dev Container (VS Code)
Open the project in VS Code → "Reopen in Container". All services start automatically.

---

## Project File Structure

```
COMP-6000-Capstone2/
├── frontend/               # React + Vite UI (Analysis, Generation, Transformation)
│   └── src/pages/          # HomePage, GenerationPage, TransformationPage, Auth
├── backend/                # Express.js API (auth + future AI endpoints)
│   └── src/index.js        # Entry point, DB schema, API routes
├── acoustic_ai/            # Python AI module
│   └── requirements.txt    # Python dependencies
├── services/dev/           # Docker Compose + PostgreSQL config
│   ├── docker-compose.yml
│   └── .env                # Local secrets (not committed)
├── script/                 # Data download scripts
│   ├── download_site_257_clips.py
│   ├── download_site_257_originals.py
│   ├── download_site_257_annotations.py
│   ├── sample_mvp_dataset.py
│   ├── fetch_nasa_env_data.py
│   └── fetch_recordings.py
├── resources/              # Downloaded audio data (gitignored)
├── .claude/                # All AI-related files (skills, context, settings)
│   ├── skills/             # Sampling policies, workflow skill docs
│   ├── context/            # Known issues, decisions, project notes
│   └── settings.local.json
└── .devcontainer/          # VS Code dev container config
```

### Storage Rule

> **All AI-related files must live under `.claude/`** — not the project root.
>
> | Type | Location |
> |------|----------|
> | Skill / workflow docs | `.claude/skills/` |
> | Known issues, decisions, notes | `.claude/context/` |
> | Claude Code settings | `.claude/settings.local.json` |

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

> Full details in `.claude/context/known_issues.md`

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
