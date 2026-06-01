# AI Module Architecture — Layered Soundscape System

> **Layout note (post-restructure):** code lives under
> `acoustic_ai/layers/layer_<X>/attempts/<member>__<stage>__<slug>/`.
> Available attempts are declared in `acoustic_ai/registry.yaml` and
> served via `GET /layers` for the frontend dropdown. Naming rules:
> [../conventions.md](../conventions.md). Per-layer
> "Module" sections describe the *role*; concrete implementations live
> across one or more attempts per layer.



## Overview

The AI pipeline is organised as five modules (A–E) arranged into two modes:
**Generation** (A+B+C+D) and **Analysis** (A+B+C via E).
Each module owns its code, its derived data, and its checkpoints.

```
acoustic_ai/
├── server/                                # registry-driven FastAPI app on :8000
│   ├── server.py                          #   GET /layers, POST /layers/<l>/attempts/<id>/generate
│   └── registry.py                        #   reads acoustic_ai/registry.yaml
├── registry.yaml                          # declares which attempts the server exposes
├── layers/                                # per-layer attempts (role per layer below)
│   ├── layer_a/attempts/<member>__<stage>__<slug>/   # Layer A — Ambient bed
│   ├── layer_b/attempts/…                            # Layer B — Weather (placeholder)
│   ├── layer_c/attempts/…                            # Layer C — Events
│   ├── layer_d/attempts/…                            # Layer D — Mixer (placeholder)
│   └── layer_e/attempts/…                            # Layer E — Analysis (partial)
├── scripts/                               # AI-module utilities
│   ├── extract_expected_samples.py        #   populate <attempt>/expected/ from source recordings
│   └── regenerate_samples.py              #   populate <attempt>/showcase/ from the handler
├── requirements.txt
└── .venv/                                 # gitignored — the ONLY interpreter for AI work
```

Each attempt is **self-contained** with `code/` (handler + train + sample),
`data/` (DVC), and the artifact tiers `expected/` / `showcase/` /
`dev-artifacts-self-testing/`. See [conventions.md](../conventions.md) for
the full per-attempt layout and artifact tier rules. Checkpoints live in `model/candidates/<member>/<stage>__<slug>/` and
are paired with attempts by name; promotion to `model/production/<role>/`
happens only after explicit sign-off.

---

## Generation Mode

```
User env request
    └── Module A: retrieve ambient bed clips (NN search in latent_clips.npy)
    └── Module B: select + mix weather assets (wind/rain intensity → gain/EQ)
    └── Module C: generate event clips with AudioGen LoRA (per-species/context LoRAs, conditioned by env/time → event timeline)
    └── Module D: combine layers → WAV + spectrogram + explanation JSON
```

## Analysis Mode

Three detector heads (E-A ambient, E-B weather, E-C events) run in parallel
on the raw mixture — no decomposer, no shared stem. Layer E aggregates.

Full design rationale (why no decomposer, per-head model options, report
schema, MVP build order) lives in
[pipeline_design.md § Analysis Mode](pipeline_design.md#analysis-mode--component-design).

---

## Module Details

### Layer A — Ambient Representation (layers/layer_a/)

| File | Role |
|---|---|
| `model.py` | VAE (CNN encoder → FusionMLP → reparameterise → MelDecoder) |
| `dataset.py` | SoundscapeDataset, env feature encoding (29-dim) |
| `train.py` | VAE training loop |
| `preprocess.py` | Mel spectrogram config + audio loading |
| `train_vocoder.py` | Ecoacoustic HiFi-GAN training |
| `retrieval.py` | NN retrieval logic [PLACEHOLDER] |

**Checkpoint:** `model/candidates/lucas/vae-site257-30epoch/best.pt` (213 MB, DVC-tracked)
**Vocoder:** `model/candidates/lucas/vocoder-hifigan-site257/best.pt` (11 MB, DVC-tracked)
**Latents:** `data/ambient/latents/latent_clips.npy` — 5,318 per-clip latents + env vectors (DVC-tracked)

**Current status:** Layer A generation is validated with AudioLDM2 LoRA on this branch — smoke checkpoint `model/candidates/lucas/layer-a-audioldm2-raw-smoke` (base `cvssp/audioldm2`) passes for quiet, stationary ambient beds. The trained VAE is retained for transformation mode and Module E analysis only — not on the Layer A generation path.

Generation algorithm, dev-path seed contract, the cleaned-segment-pool data dependency, and the legacy retrieval-first design all live in
[pipeline_design.md § Layer A](pipeline_design.md#layer-a--ambient-site-bed).

### Layer B — Weather Sound Engine (layers/layer_b/)

| File | Role |
|---|---|
| `asset_index.py` | Weather asset library loader [PLACEHOLDER] |
| `mixer.py` | Parameter → gain/EQ/density mixing [PLACEHOLDER] |

**Data:** `data/weather/weather_assets/wind/{none,light,moderate,strong}/` and `rain/{none,light,moderate,heavy}/` (DVC-tracked)
**Asset index:** `data/weather/asset_index.csv` (git-tracked, headers only for now)

Intensity mapping:
- wind: none <2 m/s, light 2–6, moderate 6–10, strong >10
- rain: none 0 mm, light 0–2, moderate 2–5, heavy >5

### Layer C — Species/Event Layer (layers/layer_c/)

**Approach:** generative AudioGen LoRA per species (and optionally per diel/seasonal context) on top of `facebook/audiogen-medium` (1.5B params, 16 kHz mono).

| File | Role |
|---|---|
| `annotation_audit.py` | Audit A2O annotation CSVs → per-species training manifests [PLACEHOLDER] |
| `dataset.py` | Manifest → 16 kHz mono clips + captions for AudioGen training [PLACEHOLDER] |
| `train_audiogen.py` | LoRA fine-tune AudioGen on a per-species manifest [PLACEHOLDER] |
| `sample_audiogen.py` | Generate event clips from a LoRA + prompt + duration + seed [PLACEHOLDER] |
| `scheduler.py` | Timeline event placement (which LoRAs fire, when, at what density) [PLACEHOLDER] |

**Checkpoints:** `model/candidates/<member>/layer-c-audiogen-<species>-<context>/` per LoRA (DVC-tracked)
**Training data:** `data/events/<species>/manifest.csv` + extracted snippets per species (DVC-tracked)

Why AudioGen over AudioLDM2 here, per-species selection policy, smoke-test hyperparameters, sample-rate boundary, tooling environment, and prompt style:
[pipeline_design.md § Layer C](pipeline_design.md#layer-c--species-and-annotated-event-layer).

### Layer D — Mixer (layers/layer_d/)

| File | Role |
|---|---|
| `audio_mixer.py` | Combine A+B+C → WAV + spectrogram + explanation JSON [PLACEHOLDER] |

No training data. Pure algorithmic combiner.

### Layer E — Analysis Explainer (layers/layer_e/)

Three detector heads run in parallel on the raw mixture (no decomposition
step), plus an aggregator that assembles the report JSON. See
[pipeline_design.md § Analysis Mode](pipeline_design.md#analysis-mode--component-design)
for the full per-head design, pre-trained model options, and report schema.

| File | Role |
|---|---|
| `ambient_similarity.py` | E-A: embed clip → k-NN against `ambient_index.csv` → context estimate [PLACEHOLDER] |
| `weather_detector.py` | E-B: wind/rain intensity from mel + tagger probs [PLACEHOLDER] |
| `event_detector.py` | E-C: species/event detection + onsets on the raw mixture [PLACEHOLDER] |
| `aggregator.py` | Assemble per-head outputs into the report JSON [PLACEHOLDER] |

Reuse-from-generation table (CLAP, EnCodec, what does and doesn't transfer):
[pipeline_design.md § Reuse from generation models](pipeline_design.md#reuse-from-generation-models).

No dedicated training data of its own. Reads Module A latents/index, Module B
asset index, Module C event index.

---

## Generative Model Strategy

### Current stage (MVP and smoke tests)

Layers A and C use frozen large base models with LoRA adapters:

| Layer | Base model | Adapter | Status |
|-------|-----------|---------|--------|
| A — Ambient | `cvssp/audioldm2` (~1.5B params, latent diffusion) | LoRA fine-tuned on ~50 Bowra clips | Smoke test 1 passed (spring night); smoke test 2 in progress (insect/cicada) |
| C — Events | `facebook/audiogen-medium` (~1.5B params, autoregressive transformer + EnCodec) | Per-species LoRA (40–80 clips each) | Smoke test pending (Southern Boobook nocturnal) |

LoRA adds ~0.1–0.5% extra parameters on top of frozen base weights. The full base model is loaded at inference time (6–8 GB VRAM each).

### Future product-level consideration

Migration from base model + LoRA to **distilled own models** is under consideration for a future production deployment. The goal is to reduce inference VRAM footprint and latency. This is not pursued during the MVP or research prototype stages.

Distillation approaches per layer:
- **Layer A:** consistency distillation (LCM) or progressive distillation — compresses 100+ DDIM steps → 4 steps (~25× speedup); optionally shrinks the U-Net backbone.
- **Layer C:** sequence-level knowledge distillation into a smaller transformer; EnCodec codec can be reused.

Pursue distillation only when all three conditions are met:
1. The LoRA path is proven to produce high-quality results across multiple species and seasonal contexts (not just smoke tests).
2. Deployment latency or VRAM cost is a demonstrated user-facing bottleneck.
3. The team has sufficient ecoacoustic data and ML research capacity without stalling product work.

If latency becomes a bottleneck before full distillation is feasible, the preferred intermediate step is **LCM step-reduction on Layer A only** — Layer C per-species LoRAs should remain as-is, since adding a species by retraining a generalist student is not cost-effective.

> Full risk and trade-off analysis: `.claude/context/ai/distillation_strategy.md`

---

## Data Ownership

| Module | Reads | Produces |
|---|---|---|
| A | `resources/downloaded_clips/`, `data/shared/spectrograms/` | `data/ambient/latents/` |
| B | `resources/downloaded_clips/` (curation), `data/shared/wavs/` | `data/weather/weather_assets/` |
| C | `resources/downloaded_annotations/`, `resources/downloaded_clips/` | `data/events/<species>/manifest.csv` + extracted snippets, `model/candidates/<member>/layer-c-audiogen-<species>-<context>/` |
| D | `data/ambient/latents/`, `data/weather/weather_assets/`, AudioGen LoRA outputs (resampled 16 → 22.05 kHz) | ephemeral WAV + JSON per request |
| E | `data/ambient/latents/`, `data/weather/asset_index.csv`, `data/events/annotation_event_index.csv` | ephemeral analysis report |

---

## Running the AI Server

The server must run natively (not in Docker) to use the Apple GPU (MPS):

```bash
cd acoustic_ai
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn server.server:app --reload --port 8000
```

Backend connects to `http://localhost:8000` (or `http://host.docker.internal:8000` from Docker).
