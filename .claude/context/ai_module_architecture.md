# AI Module Architecture — A / B / C Pattern

## Overview

The AI system is split into **three independent modules**. This separation allows each to be
trained, frozen, and extended without disrupting the others.

```
Module A                Module B                Module C
Environmental           Ecological              Conditioned
VAE                     Classifier              Generator
───────────             ──────────              ───────────
Audio + Env             Latent z                z + species
    ↓                       ↓                       ↓
Latent z (256)          Species probs           Generated mel
mu, log_var             (per species)           spectrogram
```

---

## Data Flow

```
                        ┌─────────────────┐
  Audio input ─────────►│    Module A     │
  Env features ────────►│     (VAE)       │──── latent z (256) ─────────────┐
                        └─────────────────┘                                  │
                                                                              │
                        ┌─────────────────┐                                  │
  latent z ────────────►│    Module B     │──── species probs ──────────┐    │
  (from Module A)       │  (Classifier)  │     [bird_A: 0.8,           │    │
                        └─────────────────┘      wind: 0.3, ...]       │    │
                                                                        │    │
                        ┌─────────────────┐                            │    │
  target env ──────────►│    Module C     │◄───────────────────────────┘    │
  target species ───────►│  (Generator)   │◄────────────────────────────────┘
                        └────────┬────────┘
                                 │
                            mel spectrogram
                                 │
                     ecoacoustic HiFi-GAN vocoder
                     (trained on site 257 audio)
                                 │
                           generated .wav
```

---

## Module Descriptions

### Module A — Environmental VAE (`acoustic_ai/model.py`)

- **Trained on:** 5,318 clips + 29-dim env feature vectors (unsupervised — no labels needed)
- **Architecture:** CNN encoder → FusionMLP (audio + env) → reparameterise → MelDecoder
  - Encoder: 4-block CNN → (512,) audio embedding
  - FusionMLP: concat(audio_emb, env) → shared trunk → separate mu and log_var heads (256-dim each)
  - Decoder: latent (256,) → project to (256, 4, 64) spatial seed → 5× TransposeBlock upsampling → (1, 128, 2048) → crop to target frames
- **Loss:** `MSE(recon, input) + β × KL(q(z|x) ‖ N(0,I))`, β = 0.01
- **Status:** Stage 3 complete — 30 epochs, best val loss ≈ 0.003580, KL ≈ 0.05/element
- **Key improvement over Stage 2:** VAE regularises the latent space so decoded outputs have
  temporal structure (4×64 decoder seed vs old 4×4 + bilinear stretch that caused flat spectrograms)

### Vocoder — Ecoacoustic HiFi-GAN (`acoustic_ai/train_vocoder.py`)

- **Trained on:** up to 500 site 257 `.webm.wav` clips (real ecoacoustic recordings)
- **Architecture:** HiFi-GAN V1 generator with 128 mel bins at 22,050 Hz
  - Upsample rates: [8, 8, 4, 2] — product = 512 = hop_length ✓
  - No mel interpolation, no sample rate resampling
- **Loss:** multi-resolution STFT loss + L1 mel reconstruction loss (no discriminator)
- **Status:** Stage 3 complete — 100 epochs, best val loss ≈ 31.89
- **Replaces:** `microsoft/speecht5_hifigan` (speech-domain, 80-bin, 16 kHz) — no longer used

### Generation Pipeline (current)

```
env conditions (29 features)
  → cosine similarity vs 5,318 per-clip env vectors (latent_clips.npy)
  → top-10 nearest clips → average their latents → add Gaussian noise (std=0.3)
  → Module A decoder → mel-spectrogram (128 × 1,291)
  → ecoacoustic HiFi-GAN → .wav
```

All 29 env features affect which clips are retrieved, so changing temperature, humidity,
wind, etc. produces meaningfully different output — not just season × sample_bin.

### Module B — Ecological Classifier (Stage 3, future)

- **Trained on:** annotated clips only — sparse annotation coverage is fine
- **Architecture:** small MLP head on top of **frozen** Module A encoder
- **Input:** latent `z` from Module A
- **Output:** species presence probability vector (e.g. probability per species in BirdNET taxonomy)
- **Key point:** Module A weights are frozen — B is a lightweight head that reuses learned representations

### Module C — Conditioned Generator (Stage 3, future)

- **Trained on:** all clips using both env conditions and Module B species signal
- **Architecture:** extended decoder accepting `[z (256), species_vector (N)]` as input
- **Input:** target env conditions + optional target species probabilities
- **Output:** generated mel spectrogram matching both env and species signals
- **Key point:** Modules A and B stay frozen — C is an extended decoder only

---

## Why Three Separate Modules

| Module | Trains On | Requires Labels? | Can Retrain Independently? |
|--------|-----------|-----------------|---------------------------|
| A | All clips + env features | No (unsupervised) | Yes |
| B | Annotated clips only | Yes (species labels) | Yes — A stays frozen |
| C | All clips (env + species signal) | No | Yes — A and B stay frozen |

**Pattern:** frozen backbone + task head — standard ML practice for extending a pretrained
base model without losing learned representations.

---

## Build Order

```
Stage 2 (complete)
  └─ Module A — plain autoencoder, MSE only
     └─ generation via 16 mean latent templates (season × sample_bin)
     └─ vocoder: SpeechT5 HiFi-GAN (speech-domain, 128→80 interpolation workaround)

Stage 3 — Module A improvements (complete)
  ├─ Retrained as VAE (β·KL loss, β=0.01 — regularised latent space)
  ├─ Fixed decoder architecture (4×64 seed → native 2048 frames, no bilinear stretch)
  ├─ Switched to nearest-neighbour generation (all 29 env features used)
  └─ Trained ecoacoustic HiFi-GAN vocoder on site 257 audio (128-bin, 22 kHz)

Stage 3 — remaining
  ├─ Build Module B (classifier head, frozen A)
  └─ Build Module C (extended decoder, frozen A + B)

Stage 4
  └─ Fine-tune C end-to-end
  └─ Optional: latent diffusion for temporal coherence
```

---

## File Locations

| Component | Files |
|-----------|-------|
| Module A model | `acoustic_ai/model.py` |
| Module A training | `acoustic_ai/train.py`, `acoustic_ai/dataset.py`, `acoustic_ai/preprocess.py` |
| Module A inference | `acoustic_ai/inference.py`, `acoustic_ai/server.py` |
| Module A checkpoints | `acoustic_ai/checkpoints/best.pt` |
| Latent databases | `acoustic_ai/latent_templates.npy` (16 group means), `acoustic_ai/latent_clips.npy` (5,318 per-clip latents + env vectors) |
| Vocoder training | `acoustic_ai/train_vocoder.py` |
| Vocoder checkpoint | `acoustic_ai/vocoder_checkpoints/best.pt` |
| Module B | Not yet created — will be a classifier head in `acoustic_ai/model_b.py` |
| Module C | Not yet created — will extend `MelDecoder` to accept species vector |

---

## Current Limitations

- **Latent space partially regularised** — β=0.01 KL is stronger than Stage 2 (β=0.001) but
  reconstruction loss is still ~6× stronger; pure N(0,1) sampling would still produce OOD output.
  Workaround: nearest-neighbour grounding keeps generation within the trained latent region.
- **No temporal/ecological narrative** — 30s crops with no sequential context; generated clips
  have no dawn-chorus build-up or call spacing. Fix: latent diffusion (Stage 4).
- **Vocoder trained without discriminator** — outputs are smoother than GAN-trained vocoders
  but may lack fine harmonic detail. Fix: add discriminator loss in Stage 4.
- **No species conditioning yet** — Modules B and C not built; generation is env-only.
