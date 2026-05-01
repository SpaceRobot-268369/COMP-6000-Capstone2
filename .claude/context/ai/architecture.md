# AI Module Architecture — Layered Soundscape System

## Overview

The AI pipeline is organised as five modules (A–E) arranged into two modes:
**Generation** (A+B+C+D) and **Analysis** (A+B+C via E).
Each module owns its code, its derived data, and its checkpoints.

```
acoustic_ai/
├── modules/
│   ├── ambient/     — Module A: ambient bed (VAE encoder + retrieval)
│   ├── weather/     — Module B: weather sound engine (asset mixing)
│   ├── events/      — Module C: species/event layer (annotation + scheduler)
│   ├── mixer/       — Module D: layer combiner + explanation output
│   └── analysis/    — Module E: analysis explainer (detectors)
├── precompute/      — One-off preprocessing scripts
├── data/            — DVC-tracked pipeline artifacts (per-module)
├── checkpoints/     — DVC-tracked model weights
├── server.py        — FastAPI entry point (runs locally, port 8000)
└── inference.py     — Inference helpers (used by server.py)
```

---

## Generation Mode

```
User env request
    └── Module A: retrieve ambient bed clips (NN search in latent_clips.npy)
    └── Module B: select + mix weather assets (wind/rain intensity → gain/EQ)
    └── Module C: schedule event snippets (season/time/env → event timeline)
    └── Module D: combine layers → WAV + spectrogram + explanation JSON
```

## Analysis Mode

```
Uploaded audio clip
    └── Module A (via E): encode → latent NN → ambient context + env estimate
    └── Module B (via E): spectral heuristics → wind/rain intensity
    └── Module C (via E): BirdNET / annotation lookup → detected events
    └── Module E: assemble analysis report
```

---

## Module Details

### Module A — Ambient Representation (`modules/ambient/`)

| File | Role |
|---|---|
| `model.py` | VAE (CNN encoder → FusionMLP → reparameterise → MelDecoder) |
| `dataset.py` | SoundscapeDataset, env feature encoding (29-dim) |
| `train.py` | VAE training loop |
| `preprocess.py` | Mel spectrogram config + audio loading |
| `train_vocoder.py` | Ecoacoustic HiFi-GAN training |
| `retrieval.py` | NN retrieval logic [PLACEHOLDER] |

**Checkpoint:** `checkpoints/ambient/best.pt` (213 MB, DVC-tracked)
**Vocoder:** `checkpoints/vocoder/best.pt` (11 MB, DVC-tracked)
**Latents:** `data/ambient/latents/latent_clips.npy` — 5,318 per-clip latents + env vectors (DVC-tracked)

**Current status:** VAE trained 30 epochs on event-contaminated full clips, best val loss ≈ 0.003580, KL ≈ 0.05/element. The original generation path (NN latent grounding + controlled noise + decoder + vocoder) is being **replaced** for Layer A by direct retrieval over a cleaned ambient-only segment pool — see `pipeline_design.md` § Layer A. The trained VAE is retained for transformation mode and Module E analysis only; it is not on the Layer A generation path.

**Layer A data dependency:** the cleaned segment pool (`data/ambient/ambient_segments/` + `ambient_index.csv`) must be built by `precompute/build_ambient_index.py` before retrieval can run. Cleaning is **audio-only and content-agnostic** — events are an open class (birds, vehicles, frogs, helicopters, voices, unknown), but ambient is locally stationary, so the gate flags frames that deviate > 3·MAD from a per-clip rolling-median baseline of mel/RMS/centroid/flatness/flux/ZCR features. After ±0.5 s dilation, contiguous unmasked spans ≥ 20 s are kept and sliced into 20–60 s segments (target 30 s) so runtime crossfades are minimal. BirdNET and A2O annotations are run as **post-hoc audits** over retained segments, not as gates. Retrieval matches on `diel_bin` + `season` + (`hour`, `month`) cyclic encoding only — temp/humidity/wind/rain are excluded because they belong to Layers B and C.

### Module B — Weather Sound Engine (`modules/weather/`)

| File | Role |
|---|---|
| `asset_index.py` | Weather asset library loader [PLACEHOLDER] |
| `mixer.py` | Parameter → gain/EQ/density mixing [PLACEHOLDER] |

**Data:** `data/weather/weather_assets/wind/{none,light,moderate,strong}/` and `rain/{none,light,moderate,heavy}/` (DVC-tracked)
**Asset index:** `data/weather/asset_index.csv` (git-tracked, headers only for now)

Intensity mapping:
- wind: none <2 m/s, light 2–6, moderate 6–10, strong >10
- rain: none 0 mm, light 0–2, moderate 2–5, heavy >5

### Module C — Species/Event Layer (`modules/events/`)

| File | Role |
|---|---|
| `annotation_audit.py` | Audit A2O annotation CSVs, produce event index [PLACEHOLDER] |
| `event_index.py` | Extract event snippets from clips [PLACEHOLDER] |
| `scheduler.py` | Timeline event placement [PLACEHOLDER] |

**Data:** `data/events/annotation_event_index.csv`, `event_snippets/`, `birdnet_labels/` (DVC-tracked)
**Pre-condition:** annotation_audit.py must run before any Module C training.

### Module D — Mixer (`modules/mixer/`)

| File | Role |
|---|---|
| `audio_mixer.py` | Combine A+B+C → WAV + spectrogram + explanation JSON [PLACEHOLDER] |

No training data. Pure algorithmic combiner.

### Module E — Analysis Explainer (`modules/analysis/`)

| File | Role |
|---|---|
| `weather_detector.py` | Detect wind/rain intensity from mel spectrogram [PLACEHOLDER] |
| `event_detector.py` | Detect species/events from audio [PLACEHOLDER] |

No dedicated training data. Uses Module A latents, Module B asset index, Module C event index.

---

## Data Ownership

| Module | Reads | Produces |
|---|---|---|
| A | `resources/downloaded_clips/`, `data/shared/spectrograms/` | `data/ambient/latents/` |
| B | `resources/downloaded_clips/` (curation), `data/shared/wavs/` | `data/weather/weather_assets/` |
| C | `resources/downloaded_annotations/`, `resources/downloaded_clips/` | `data/events/annotation_event_index.csv`, `event_snippets/` |
| D | `data/ambient/latents/`, `data/weather/weather_assets/`, `data/events/event_snippets/` | ephemeral WAV + JSON per request |
| E | `data/ambient/latents/`, `data/weather/asset_index.csv`, `data/events/annotation_event_index.csv` | ephemeral analysis report |

---

## Running the AI Server

The server must run natively (not in Docker) to use the Apple GPU (MPS):

```bash
cd acoustic_ai
uvicorn server:app --reload --port 8000
```

Backend connects to `http://localhost:8000` (or `http://host.docker.internal:8000` from Docker).
