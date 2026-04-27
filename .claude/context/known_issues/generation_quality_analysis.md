# Generation Quality Analysis — Root Causes & Resolution Status

## Summary

Four compounding problems caused the original "machine noise" audio. Three have been resolved
in Stage 3; one remains open.

| # | Issue | Status |
|---|-------|--------|
| 1 | Wrong model type — plain autoencoder, mean latent decodes to garbage | ✅ **Resolved** — retrained as VAE (β=0.01 KL) |
| 2 | MSE loss → blurry spectrograms, flat temporal output | ✅ **Resolved** — decoder architecture fixed; partially remains (MSE still dominant) |
| 3 | Wrong vocoder domain — speech HiFi-GAN on ecoacoustic mels | ✅ **Resolved** — trained ecoacoustic HiFi-GAN on site 257 audio |
| 4 | No temporal/ecological structure — clips have no narrative | 🔲 **Open** — Stage 4 work (latent diffusion) |

---

## Problem 1 — Wrong Model Type ✅ RESOLVED

**Was:** plain autoencoder — mean latent `z̄` decoded to garbage because the decoder had
never seen that point during training.

**Fix applied:** retrained as VAE with KL divergence loss (β=0.01, 30 epochs).
- FusionMLP now outputs `(mu, log_var)` instead of a single `z`
- Reparameterisation trick: `z = mu + ε·σ` during training
- KL term regularises the latent space toward N(0,1)
- Result: KL ≈ 0.05/element (vs 0.20 before with β=0.001)

**Remaining caveat:** with β=0.01 the reconstruction loss is still ~6× stronger than KL, so
the latent space is not fully unit-Gaussian. Pure N(0,1) sampling still goes slightly OOD.
Workaround: nearest-neighbour generation keeps sampling within the trained latent region.

---

## Problem 2 — Flat Temporal Output ✅ RESOLVED (decoder fix)

**Was:** decoder used a 4×4 spatial seed → bilinear-interpolated 10× in time (128 → 1,290
frames). Every generated spectrogram looked identical in the time axis — a smooth gradient.

**Fix applied:** decoder spatial seed changed from `(256, 4, 4)` to `(256, 4, 64)`.
- 5 TransposeBlock upsampling steps → native output `(1, 128, 2048)`
- Crop to target frames — no bilinear stretch
- Result: decoder now generates temporal variation natively

**Partially remaining:** MSE reconstruction loss still encourages blurry, averaged outputs.
Sharp harmonic peaks and precise bird call onsets are softened. Fix: perceptual/adversarial
loss (Stage 4).

---

## Problem 3 — Wrong Vocoder Domain ✅ RESOLVED

**Was:** `microsoft/speecht5_hifigan` (LJSpeech speech, 80 mel bins, 16 kHz). Required
128→80 mel interpolation and 16→22 kHz resampling. Produced buzzing, robotic artefacts.

**Fix applied:** trained a dedicated ecoacoustic HiFi-GAN on site 257 audio (`train_vocoder.py`).
- Input: 128-bin mel at 22,050 Hz — matches our model exactly
- Trained on up to 500 real site 257 clips, 100 epochs
- Loss: multi-resolution STFT + L1 mel (no discriminator)
- No mel interpolation, no resampling — end-to-end native
- Best val loss: 31.89

**Remaining:** no discriminator means outputs are smoother than GAN-trained vocoders.
Fine harmonic detail may be softened. Fix: add adversarial loss in Stage 4.

---

## Problem 4 — No Temporal/Ecological Structure 🔲 OPEN

**Is:** model trained on random 30 s crops with no sequential context. Latent `z` encodes
a snapshot, not a narrative. Generated clips have no dawn-chorus build-up, no temporal
spacing of calls, no ecological arc.

**Workaround in place:** nearest-neighbour generation retrieves real training clip latents,
so the latent is grounded in actual ecoacoustic structure. But the decoder still generates
from a single static latent — no temporal evolution.

**Fix (Stage 4):** latent diffusion — train a small diffusion model that learns `p(z|env)`
via iterative denoising conditioned on env features.

---

## Current Generation Pipeline

```
env conditions (29 features)
  → cosine similarity vs 5,318 per-clip env vectors (latent_clips.npy)
  → top-10 nearest clips → average their latents → add noise (std=0.3)
  → VAE decoder (4×64 seed → native 2048 frames → crop to 1,291)
  → ecoacoustic HiFi-GAN (128-bin, 22,050 Hz, trained on site 257)
  → .wav
```

---

## Realistic Quality Expectations

| Approach | Sounds Natural? | Ecoacoustic? | Status |
|----------|----------------|-------------|--------|
| Stage 2 (mean latent + speech HiFi-GAN) | No | No | Superseded |
| **Stage 3 (VAE + NN latents + eco vocoder)** | **Somewhat** | **Somewhat** | **Current** |
| + perceptual/adversarial loss | Yes | Mostly | Stage 4 |
| + latent diffusion for temporal structure | Yes | Yes | Stage 4 |
| Full conditional diffusion (e.g. AudioLDM) | Best | Yes | Research-level |

---

## Remaining Issues Summary

| # | Issue | Impact | Fix | Stage |
|---|-------|--------|-----|-------|
| 2b | MSE loss → blurry spectrograms | Medium — softened harmonics | Multi-scale spectral loss or GAN discriminator | 4 |
| 3b | No discriminator in vocoder | Low-medium — smooth output | Add adversarial loss to `train_vocoder.py` | 4 |
| 4 | No temporal narrative in generated clips | Medium — no ecological arc | Latent diffusion conditioned on env | 4 |
| — | Latent space partially regularised (β too small) | Low — NN workaround covers it | Increase β or use β-annealing schedule | optional |

---

## Next Steps (Stage 4)

1. **Multi-scale spectral loss** for Module A — replace/supplement MSE with STFT losses at
   multiple resolutions to encourage sharper spectrograms
2. **Vocoder discriminator** — add a waveform discriminator to `train_vocoder.py` for
   finer harmonic detail
3. **Latent diffusion** — train a small DDPM on the VAE latent space conditioned on env
   features — generates diverse, temporally coherent soundscapes
