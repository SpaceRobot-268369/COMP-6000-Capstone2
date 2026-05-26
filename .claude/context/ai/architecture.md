# AI Module Architecture — Layered Soundscape System

> **Layout note (post-restructure):** code lives under
> `acoustic_ai/layers/layer_<X>/attempts/<member>__<stage>__<slug>/`.
> Available attempts are declared in `acoustic_ai/registry.yaml` and
> served via `GET /layers` for the frontend dropdown. Naming rules:
> [../dev/attempt_naming.md](../dev/attempt_naming.md). Per-layer
> "Module" sections describe the *role*; concrete implementations live
> across one or more attempts per layer.



## Overview

The AI pipeline is organised as five modules (A–E) arranged into two modes:
**Generation** (A+B+C+D) and **Analysis** (A+B+C via E).
Each module owns its code, its derived data, and its checkpoints.

```
acoustic_ai/
├── modules/
│   ├── ambient/     — Module A: ambient bed (VAE encoder + retrieval)
│   ├── weather/     — Module B: weather sound engine (asset mixing)
│   ├── events/      — Module C: species/event layer (AudioGen LoRA generative)
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
    └── Module C: generate event clips with AudioGen LoRA (per-species/context LoRAs, conditioned by env/time → event timeline)
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

**Current status:** Layer A generation is being validated with AudioLDM2 LoRA on this branch. This is one attempted Layer A implementation and has succeeded for the smoke test. The current user-validated smoke checkpoint is `model/candidates/lucas/layer-a-audioldm2-raw-smoke`, based on `cvssp/audioldm2`; it works well for quiet, environmental-like ambient beds with only minor issues. Keep output low-volume and mostly stationary. The dev path is fixed-prompt because the smoke dataset is tiny: the frontend should expose only a non-negative integer seed, the backend should forward only `{ seed }`, and FastAPI owns the prompt/checkpoint/settings. Different seeds produce different variations from the same model/prompt/settings; the same seed should reproduce effectively the same audio on the same code path. Seed is not temperature, and temperature is not exposed here. The failed high-RMS checkpoint `model/candidates/lucas/layer-a-audioldm2-rms005-smoke` is deprecated because it produced pulsing/machine-like artifacts after over-amplifying quiet field recordings. If this branch is merged into `main`, update the broader docs so AudioLDM2 LoRA is described consistently as the main Layer A path. The trained VAE is retained for transformation mode and Module E analysis only; it is not on the Layer A generation path. CLI and frontend Layer A spectrograms should use the shared log-mel renderer in `modules.ambient.diffusion.layer_a_visualization`.

**Layer A data dependency:** the cleaned segment pool (`data/ambient/ambient_segments/` + `ambient_index.csv`) must be built by `precompute/build_ambient_index.py` before retrieval can run. Cleaning is **audio-only and content-agnostic** — events are an open class (birds, vehicles, frogs, helicopters, voices, unknown), but ambient is locally stationary, so the gate flags frames that deviate > 3·MAD from a per-clip rolling-median baseline of mel/RMS/centroid/flatness/flux/ZCR features. After ±0.5 s dilation, contiguous unmasked spans ≥ 20 s are kept and sliced into 20–60 s segments (target 30 s) so runtime crossfades are minimal. BirdNET and A2O annotations are run as **post-hoc audits** over retained segments, not as gates. Retrieval matches on `diel_bin` + `season` + (`hour`, `month`) cyclic encoding only — temp/humidity/wind/rain are excluded because they belong to Layers B and C.

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

**Approach:** **Generative**, using **AudioGen LoRA** fine-tuned per species (and optionally per diel/seasonal context) on top of the `facebook/audiogen-medium` base model. AudioGen is chosen over AudioLDM2 for this layer because:
- Token-based (EnCodec) representation preserves transients better than mel→HiFi-GAN
- Trained on AudioSet's environmental/animal labels — the base model already has owl, songbird, insect priors
- Native short-clip operating range (1–10 s) matches Layer C event durations
- LoRA fine-tuning is supported via PEFT on the transformer attention layers

| File | Role |
|---|---|
| `annotation_audit.py` | Audit A2O annotation CSVs → per-species training manifests [PLACEHOLDER] |
| `dataset.py` | Manifest → 16 kHz mono clips + captions for AudioGen training [PLACEHOLDER] |
| `train_audiogen.py` | LoRA fine-tune AudioGen on a per-species manifest [PLACEHOLDER] |
| `sample_audiogen.py` | Generate event clips from a LoRA + prompt + duration + seed [PLACEHOLDER] |
| `scheduler.py` | Timeline event placement (which LoRAs fire, when, at what density) [PLACEHOLDER] |

**Base model:** `facebook/audiogen-medium` (1.5B params, 16 kHz mono)
**Checkpoints:** `model/candidates/<member>/layer-c-audiogen-<species>-<context>/` per LoRA (DVC-tracked)
**Training data:** `data/events/<species>/manifest.csv` + extracted snippets per species (DVC-tracked)

**Pre-condition:** annotation_audit.py must run before any Module C training, to produce per-species manifests filtered by score, duration, and diel context.

**Smoke test:** Single LoRA on Southern Boobook nocturnal calls — see `pipeline_design.md` Layer C section for the smoke-test selection policy and hyperparameters.

**Sample-rate boundary:** AudioGen output is 16 kHz mono. Module D mixer must resample 16 kHz → 22,050 Hz at the layer boundary before overlaying on the ambient bed.

**Tooling note:** AudioGen lives in Meta's `audiocraft` repo (not HuggingFace `diffusers`). Use a separate Python environment (`acoustic_ai/.venv-audiogen`) to avoid torch/torchaudio conflicts with the AudioLDM2 stack.

### Layer D — Mixer (layers/layer_d/)

| File | Role |
|---|---|
| `audio_mixer.py` | Combine A+B+C → WAV + spectrogram + explanation JSON [PLACEHOLDER] |

No training data. Pure algorithmic combiner.

### Layer E — Analysis Explainer (layers/layer_e/)

| File | Role |
|---|---|
| `weather_detector.py` | Detect wind/rain intensity from mel spectrogram [PLACEHOLDER] |
| `event_detector.py` | Detect species/events from audio [PLACEHOLDER] |

No dedicated training data. Uses Module A latents, Module B asset index, Module C event index.

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
