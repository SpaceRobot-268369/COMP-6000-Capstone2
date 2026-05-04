# Layer A — Method 3: Latent Diffusion + Env Conditioning

Branch: `model/lucas/layer-a-ambient-method-3`
Started: 2026-05-04

## Goal

Replace the MVP retrieval-only Layer A with a **generative** model that produces
novel ambient bed audio conditioned on environmental context, while structurally
avoiding the machine-noise failure mode of attempts 1 and 2 (see
`.claude/context/ai/logs/mvp_decision_log.md` § Attempts 1–2).

## Why latent diffusion (over the alternatives in the comparison table)

- **Diffusion does not collapse to the mean** — it predicts noise residuals, not
  targets. The MSE-mean-collapse pathology that produced "smooth wash → metallic
  vocoder" output in attempts 1–2 cannot occur structurally.
- **Reuses the existing stack.** Frozen VAE encoder gives us a 256-dim latent;
  trained vocoder turns decoded mel back into audio. The new component is a
  small UNet/MLP over a 256-dim vector, not a full audio model.
- **Conditioning is well-typed.** AdaLN on the env vector is exactly the
  right injection point for low-dim continuous + categorical conditioning.
- **Cheap to train.** A 256-dim 1D diffusion model is tiny — a few hundred K
  parameters — vs. millions for a 2D mel-spectrogram diffusion. Trainable in
  hours on one GPU once latents are precomputed.

## Architectural decisions and trade-offs

### Decision 1: VAE stays frozen
We do **not** re-train the VAE on cleaned segments. The diffusion model learns
the distribution of cleaned-segment latents as the existing encoder happens to
project them. This is principled: the VAE is now used as a generic feature
extractor, not as a generator. If diffusion underfits, we revisit.

**Risk:** the existing VAE's latent space was shaped by event-contaminated
training. Cleaned ambient may all cluster into a narrow region of the latent
manifold, in which case diffusion has little to model. **Mitigation:** the
precompute step reports latent statistics (mean, std, per-dim activity) over
the cleaned pool. If activity is collapsed, fall back to fine-tuning the VAE
encoder on cleaned segments only.

### Decision 2: Conditioning = 12-dim ambient-relevant vector
The VAE's `FusionMLP` requires 29-dim env at encode time (architectural
constraint we cannot avoid without re-training). We feed it the full 29-dim
vector as before during precompute.

But the diffusion model only sees a **12-dim conditioning vector**:

```
[hour_sin, hour_cos, month_sin, month_cos,           # 4 cyclic
 season_onehot(4),                                   # spring/summer/autumn/winter
 diel_onehot(4)]                                     # dawn/morning/afternoon/night
```

This matches Layer A's retrieval-key philosophy (`pipeline_design.md:53`) —
weather and temperature belong to Layers B and C respectively. Including them
in A's diffusion conditioning would either double-count or pull in irrelevant
similarity.

### Decision 3: 1D diffusion, not 2D
Because the VAE latent is `(B, 256)` (not a feature map), the diffusion UNet is
a 1D residual MLP with timestep embedding + AdaLN, not a 2D conv UNet. Smaller,
faster, and the right shape for the data.

### Decision 4: v-prediction parameterisation
Train the model to predict `v = α·ε - σ·x_0` rather than `ε` directly.
v-prediction gives more stable training at low-T and high-T regimes, and is
standard in current latent diffusion work (Imagen, Stable Diffusion 2.x).

### Decision 5: DDIM sampler with 50 steps at inference
Faster than DDPM, deterministic given a seed (good for reproducible demos),
and 50 steps is well within budget for a 256-dim latent.

## Pipeline

```
Offline (precompute):
  cleaned ambient segment.wav
    → mel spectrogram
    → frozen VAE encoder (with full 29-dim env)
    → 256-dim latent mu
    → save to data/ambient/latents/ambient_latents.npy
       + ambient_latents_index.csv  (segment_id, env_12dim, latent_idx)

Training:
  for each (latent z_0, env_12) in batch:
    sample t ~ Uniform(0, T)
    sample ε ~ N(0, I)
    z_t = α_t·z_0 + σ_t·ε
    v_target = α_t·ε - σ_t·z_0
    v_pred = UNet(z_t, t, env_12)
    loss = MSE(v_pred, v_target)

Inference (per request):
  build env_12 from user request
  sample z_T ~ N(0, I)            # 256-dim
  for t in DDIM_schedule(50):
    v = UNet(z_t, t, env_12)
    z_{t-1} = ddim_step(z_t, v, t)
  mel = VAE.decoder(z_0)
  wav = vocoder(mel)
  return wav
```

## File layout

```
acoustic_ai/
├── modules/ambient/diffusion/
│   ├── __init__.py
│   ├── model.py        — env-conditioned 1D UNet + AdaLN
│   ├── schedule.py     — α_t, σ_t, v-prediction utils, DDIM sampler
│   ├── dataset.py      — loads precomputed latents + 12-dim env
│   ├── train.py        — training loop
│   └── sample.py       — inference: env → latent → mel → wav
├── precompute/
│   └── precompute_ambient_latents.py  — encode cleaned segments via frozen VAE
└── data/ambient/latents/
    ├── ambient_latents.npy            — (N, 256) float32
    └── ambient_latents_index.csv      — segment_id, env_12 columns, latent_idx
```

## Hyperparameters (initial guess; tune after first run)

```yaml
diffusion:
  latent_dim: 256
  cond_dim: 12
  hidden_dim: 512
  num_blocks: 6
  num_train_timesteps: 1000
  beta_start: 1.0e-4
  beta_end: 2.0e-2
  schedule: "cosine"        # cosine schedule has better behaviour at endpoints
  prediction: "v"
  num_inference_steps: 50
  epochs: 200
  batch_size: 64
  lr: 2.0e-4
  weight_decay: 1.0e-4
  ema_decay: 0.999
  cond_dropout_p: 0.1       # for classifier-free guidance
  cfg_scale: 1.5            # at inference
```

## Build order

1. ☐ Dev log + plan (this file)
2. ☐ Scaffold `modules/ambient/diffusion/` with empty stubs
3. ☐ Add `diffusion:` section to `params.yaml`
4. ☐ Write `precompute/precompute_ambient_latents.py`
5. ☐ Implement `model.py` (1D UNet w/ AdaLN)
6. ☐ Implement `schedule.py` (cosine schedule, v-pred, DDIM)
7. ☐ Implement `dataset.py` (latent + env loader)
8. ☐ Implement `train.py` (training loop)
9. ☐ Implement `sample.py` (inference)
10. ☐ Smoke-test end to end on a tiny subset
11. ☐ Promote durable insights to `.claude/context/ai/`, delete this folder

## Open questions

- **Does the existing VAE collapse cleaned-ambient latents?** Answer comes from
  the precompute step's stats report. If yes → fine-tune the VAE encoder first.
- **How many cleaned segments survive after the audit?** Currently 1,982. For a
  tiny conditioning task that's plenty; for diffusion it's borderline. Heavy
  augmentation may be needed (noise injection on latents during training).
- **Vocoder mismatch.** The HiFi-GAN was trained on full-clip mel statistics.
  If diffusion-decoded mels look different from real mels, the vocoder may
  squeak. Audit by vocoding a real-mel vs. a diffusion-mel side by side once
  end-to-end runs.
