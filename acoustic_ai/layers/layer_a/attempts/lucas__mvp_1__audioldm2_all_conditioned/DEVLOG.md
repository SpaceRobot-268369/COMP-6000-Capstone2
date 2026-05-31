# DEVLOG — lucas__mvp_1__audioldm2_all_conditioned

## Hypothesis

A single LoRA on top of `cvssp/audioldm2`, trained on the full Bowra dry-A
ambient pool with captions encoding `(season, diel, temp, humidity, wind,
date)`, will reproduce smoke-test-quality ambient beds across all 16 (season,
diel) cells. Success = changing the caption's condition fields produces
audibly different ambient beds AND smoke_1 / smoke_2 scenes remain
reproducible.

## Setup

- Branch / commit at training: `model/lucas/layer-a-mvp-1-all-conditioned` @ `3f1b1ec`
- Dataset: `resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/clips.dvc`
  - md5 `6d033475c1e9d203454fb076a5142ba3.dir`, 1,078 clips after filter+cap
  - Train/val: 971 / 107 (stratified per cell, seed 42)
- Environment: AWS Ubuntu 22.04, Tesla T4 16 GB, driver 580 / CUDA runtime 13,
  Python 3.10.12, torch 2.12.0+cu130, accelerate single-GPU fp16
  (T4 sm_75 has no native bf16)
- Code freeze: `code/` matches the smoke_1 script + val-loss addition
  (per IMPLEMENTATION_PLAN §7.0)

## Run log

### 2026-05-30 10:51 — training run #1 (full §7.2)

- Command:
  ```
  ./.venv/bin/accelerate launch --mixed_precision fp16 \
    layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/code/train_audioldm2.py \
    --manifest_path ../resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest.csv \
    --output_dir ../model/candidates/lucas/mvp_1__audioldm2_all_conditioned \
    --batch_size 4 --num_epochs 3 --learning_rate 1e-5
  ```
- Hyperparams: r=8, alpha=32, dropout=0.1, target_modules `[to_q, to_k, to_v,
  to_out.0]`. 729 optimization steps total.
- Epoch losses:

  | epoch | train_loss | val_loss | wall-clock |
  |---|---:|---:|---|
  | 1/3 | 0.1473 | 0.1540 | ~5:40 |
  | 2/3 | 0.1232 | 0.1362 | ~5:40 |
  | 3/3 | 0.1192 | 0.1316 | ~5:40 |

- Total wall-clock: 16 min 50 s
- GPU peak: 8.3 GB / 16 GB (headroom for batch=8 if wanted)
- Anomalies: one diffusers warning
  `Expected types for language_model: (GPT2LMHeadModel,), got GPT2Model` —
  known cosmetic for `cvssp/audioldm2`, doesn't affect training.

### 2026-05-30 10:48 — environment fixes prior to run

Logged in case future attempts hit the same:

- `torchaudio 2.11` routes `.load()` through `torchcodec` → requires system
  FFmpeg 4 libs. Patched `audioldm2_dataset.py` to use `soundfile.read()`
  instead (commit `3f1b1ec`). `soundfile` was already a dependency.
- A 2-step dry-run with a `/tmp/` stub manifest failed because the training
  script derives `project_root = manifest_path.parent.parent.parent.parent`,
  which became `/` for a `/tmp` manifest. **Lesson:** the stub manifest must
  live inside the repo (at the same depth as the real one) for path
  resolution. Use `resources/.../dryrun_manifest.csv`.

### 2026-05-30 11:08 — checkpoint saved

`adapter_model.safetensors` 11.3 MB → `model/candidates/lucas/mvp_1__audioldm2_all_conditioned/`.
Note: the training script auto-wrote `README.md` over the authored checkpoint
card. Restore from git before any DVC commit.

### 2026-05-30 11:20 — showcase generation (3 seeds)

| seed | prompt cell | dvc/local artifact |
|---|---|---|
| 42 | night spring, mild 18C, moderate humidity, still, 2024-09-15 | `dev-artifacts-self-testing/seed_42_baseline/` |
| 43 | afternoon summer, hot 35C, dry air, light breeze, 2024-01-20 | `dev-artifacts-self-testing/seed_43_summer_afternoon/` |
| 44 | night autumn, mild 16C, dry air, still, 2024-04-10 | `dev-artifacts-self-testing/seed_44_autumn_night/` |

Inference defaults: 200 steps, default guidance, `output_target_rms` per
script defaults, `audio_length_in_s=10`. ~30 s per sample on T4.

### 2026-05-30 11:30 — listening verdict

Below the smoke_1 / smoke_2 quality bar. Per-cell character is present but
washed out — feels like "average ambient with a season-shaped EQ on top"
rather than the texture sharpness smoke achieved on a single scene.

## Comparison vs prior attempts

| attempt | scene coverage | per-scene quality | net |
|---|---|---|---|
| `lucas__smoke_1__audioldm2_spring_night` | 1 (spring night) | high (reference) | ✓ shipped as smoke baseline |
| `lucas__smoke_2__audioldm2_insects` | 1 (summer afternoon insects) | high | ✓ shipped as smoke baseline |
| **`lucas__mvp_1__audioldm2_all_conditioned`** | **16 cells** | **medium / blurred** | **breadth ✓, depth ✗** |

## Retrospective

### What we expected vs what happened

Expected: a single shared LoRA, fed conditioned captions, would learn cell-
specific texture and reproduce smoke quality across the matrix. Reality:
texture is blurred per cell. Conditioning IS working (cells sound different
from each other) but the resolution is insufficient for the demo bar.

### Suspected contributors, ranked

1. **Shared LoRA capacity dilution.** ~2.9 M trainable parameters spread
   across 16+ scenes vs all 2.9 M dedicated to one scene in smoke.
2. **Per-scene step exposure too low.** Strong cells saw ~75 step-views,
   thin cells ~15. Smoke_1 saw 250 step-views of its one scene.
3. **Under-converged.** Losses still descending at epoch 3; epoch 5 would
   have squeezed a bit more, but probably not enough alone.
4. **Date token noise.** ~140 unique date tokens act as ID labels; weak
   signal; useless at inference. Logged in `caption_schema_log.md`.
5. **Caption schema too rich for data scale.** 6 axes × 1,082 clips is
   sparse coverage.

### Decisions made

1. MVP-2 path adopted as production architecture: per-cell LoRAs + router.
   Logged in `mvp_decision_log.md` (2026-05-30 entry).
2. Caption v3 (drop date) for all forward attempts. Logged in
   `caption_schema_log.md`.
3. Thin cells expanded via filter-relax + augmentation, never dropped.
4. Phase 1A `lucas__mvp_1_1__spring_night_replica` runs first as the data-
   pipeline sanity check — if smoke_1 quality is not reproducible on that
   subset, the data pipeline (rule 5 / content filters) is the blocker, not
   the architecture.

### Things to never do again

- Saving the training-script README into the same path as the authored
  checkpoint card. Fix: training script should write `MODEL_CARD.md` or
  `adapter_card.md`, not `README.md`. Filed against the next code freeze.
- Using a `/tmp/` stub manifest for dry-runs (path resolution breaks).

### Follow-ups spawned

- `lucas__mvp_1_1__spring_night_replica` (Phase 1A diagnostic)
- `lucas__mvp_1_2__shared_lora_maxed` (Phase 1B reference ceiling)
- `lucas__mvp_2__per_cell_loras` (production candidate, gated on 1A pass)

## Status

Closed 2026-05-30. Checkpoint kept at `model/candidates/lucas/mvp_1__audioldm2_all_conditioned/`
as a roll-backable historical reference. Will not be promoted to
`model/production/`.
