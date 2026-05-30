# DEVLOG — lucas__mvp_1_1__spring_night_replica

## Hypothesis

Smoke-style training (r=8, 5 epochs, single scene) on the spring-night subset
of the MVP-1 dataset reproduces `lucas__smoke_1` quality. Confirms the MVP-1
data pipeline is not the cause of MVP-1's per-cell quality blur.

## Setup

- Branch / commit at training: `model/lucas/layer-a-mvp-1-all-conditioned`
  (commit recorded at tag `attempt/lucas__mvp_1_1__spring_night_replica`).
- Parent dataset: `resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/clips.dvc`
  (same DVC blob as MVP-1; no rebuild).
- Subset manifest: `spring_night_subset.csv` — 100 rows where
  `(season=spring, diel_bin=night)`, preserving the parent manifest's
  `split` column (90 train + 10 val).
- Build command (run once on server before training):
  ```
  M=resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset
  head -1 $M/manifest.csv > $M/spring_night_subset.csv
  awk -F, 'NR>1 && $5=="night" && $6=="spring"' $M/manifest.csv >> $M/spring_night_subset.csv
  ```
- Environment: AWS EC2 Tesla T4 (host `shinypokemon`), torch 2.12.0+cu130,
  fp16 single-GPU, same venv as MVP-1.

## Run log

### 2026-05-30 14:18 — training run #1

Command (from `~/lucano/COMP-6000-Capstone2/acoustic_ai/`):

```
./.venv/bin/accelerate launch --mixed_precision fp16 \
  layers/layer_a/attempts/lucas__mvp_1_1__spring_night_replica/code/train_audioldm2.py \
  --manifest_path ../resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/spring_night_subset.csv \
  --output_dir ../model/candidates/lucas/mvp_1_1__spring_night_replica \
  --batch_size 4 --num_epochs 5 --learning_rate 1e-5
```

- Total optimisation steps: **115** (90 train ÷ 4 batch × 5 epochs, single GPU).
- Wall-clock: **~2 min 36 s** (14:18:12 → 14:20:48). Much faster than the
  ~14 min projection — projection assumed the MVP-1 step rate, but the
  smaller subset never blocks on dataloader prefetch.
- Audio RMS normalisation: **off** (parity with smoke_1 / MVP-1 negative result).
- First batch sanity (post-preprocess): rms ≈ 2.3e-3, peak ≈ 7e-3 — quiet
  ambient as expected, no clipping.

Epoch losses:

| epoch | train_loss | val_loss |
|---|---|---|
| 1 | 0.1749 | 0.0586 |
| 2 | 0.1258 | 0.0578 |
| 3 | 0.2062 | 0.0580 |
| 4 | 0.1399 | 0.0573 |
| 5 | 0.1649 | 0.0565 |

Notes:
- Train loss is **noisy step-mean** (tiny dataset, no smoothing); the single
  spike at epoch 3 (0.2062) reflects a couple of high-loss steps and is not
  a divergence — val keeps falling.
- Val loss is **~2.3× lower** than MVP-1's final val (0.0565 vs 0.1316).
  Not directly comparable (10-clip val vs 107, single cell vs 16 cells),
  but the directionality matches expectations: with capacity not spread
  across 16 cells, the LoRA fits the one cell tightly.

Checkpoint: `model/candidates/lucas/mvp_1_1__spring_night_replica/adapter_model.safetensors`
(11.87 MB — same shape as MVP-1's r=8 adapter, as designed).

### 2026-05-30 — showcase generation

3 seeds generated on shinypokemon, rsync'd to local
`dev-artifacts-self-testing/`:

| folder | seed | prompt source | purpose |
|---|---|---|---|
| `seed_42_v2_caption/` | 42 | MVP-1.1 v2 (with date + env fields) | native caption |
| `seed_42_smoke1_prompt/` | 42 | smoke_1's verbatim prompt | A/B regression vs smoke_1 (same seed, different LoRA) |
| `seed_43_v2_caption/` | 43 | MVP-1.1 v2 | variation seed |

Inference settings (from `params.yaml`): 200 steps, guidance 2.0, audio
length 10 s, output_target_rms 0.0015, highpass 80 Hz.

### 2026-05-30 — listening verdict

**Quality matches smoke_1.** The `seed_42_smoke1_prompt` A/B is the cleanest
single-variable comparison (same prompt, same seed, only difference is which
LoRA's adapter is merged) and it sounds equivalent to smoke_1 — same
spring-night ambient bed character, no obvious degradation.

The v2-caption seeds (42, 43) are also in-distribution spring-night ambient
beds. The extra caption fields (temperature, humidity, wind, date) did not
visibly hurt the single-cell fit.

## Decision gate

| outcome | next step | actual |
|---|---|---|
| Quality matches smoke_1 | data pipeline confirmed sound → proceed to Phase 1B + Phase 2 (per-cell) | **✓ this outcome** |
| Quality close but not matching | acceptable — proceed but flag for Phase 4 inference sweep | — |
| Quality clearly worse than smoke_1 | halt — investigate data pipeline | — |

**Decision:** PROCEED. The MVP-1 data filter pipeline (rules 1–7 plus §6.1
content filters) is sound. MVP-1's per-cell quality blur is a **capacity
problem**, not a data problem — confirmed by removing the per-cell capacity
dilution (this attempt) and recovering smoke_1 quality.

## Comparison vs prior attempts

| attempt | scope | val_loss (final) | quality |
|---|---|---|---|
| `lucas__smoke_1__audioldm2_spring_night` | spring-night only, raw smoke pool | n/a (no metrics.json) | reference quality ★ |
| `lucas__mvp_1__audioldm2_all_conditioned` | 16 (season,diel) cells, ~1,082 clips, r=8, 3 epochs | 0.1316 | below smoke per cell (blurred) |
| `lucas__mvp_1_1__spring_night_replica` (this) | spring-night only, MVP-1 filter pipeline, r=8, 5 epochs | **0.0565** | **matches smoke_1** ✓ |

Single-variable contrast vs MVP-1: same code path, same hyperparameters
(except 5 vs 3 epochs), same caption schema (v2), same source segment pool,
same content filters — the only knob changed is **how many (season, diel)
cells the LoRA is asked to cover**. With 1 cell → smoke quality. With 16
cells (at r=8) → blurred. Capacity dilution confirmed.

## Retrospective

What worked
- Single-variable design. Keeping caption schema v2 (with date) deliberately
  preserved the comparison's interpretability. If quality had dropped, we'd
  have known either pipeline OR caption was wrong; instead we have a clean
  isolation of "pipeline is fine".
- Reusing the same DVC blob (no `clips.dvc` rebuild) eliminated any chance
  that segment extraction had drifted between MVP-1 and this attempt.
- tmux for SSH-disconnect survival (overkill at 2 min wall-clock, but the
  pattern is needed for Phase 2's 16 LoRAs).

What I'd do differently
- The 14-min ETA was wrong because I extrapolated from MVP-1's step rate
  without re-checking dataloader behaviour on the smaller subset.
  Mostly cosmetic, but it cost a single re-check.
- `train_audioldm2.py` still overwrites `model/candidates/.../README.md`
  on save. Not blocking, but should be fixed to write `MODEL_CARD.md`
  before Phase 2 (16 attempts will hit the same overwrite).

Open questions surfaced
- Does the v2 caption's date field carry any conditioning signal in the
  single-cell case? Phase 1B (caption v3, date-dropped, shared LoRA r=64)
  will answer this directly — if val loss / showcase quality holds without
  the date, v3 becomes the MVP-2 default.
- Val loss collapsed from 0.0586 → 0.0565 between epoch 1 and 5 — small
  margin. Is 5 epochs over-trained for a 90-clip set? Phase 1B at 8 epochs
  on the full pool will give a longer learning curve to inspect.

## Closeout

- Status: **closed — passed decision gate**.
- Tag: `attempt/lucas__mvp_1_1__spring_night_replica` (created at commit
  recording this DEVLOG).
- Successor: `lucas__mvp_1_2__shared_lora_maxed` (Phase 1B — caption v3,
  shared LoRA, r=64, 8 epochs).
