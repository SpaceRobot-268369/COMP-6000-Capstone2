# DEVLOG — lucas__mvp_1_1__spring_night_replica

## Hypothesis

Smoke-style training (r=8, 5 epochs, single scene) on the spring-night subset
of the MVP-1 dataset reproduces `lucas__smoke_1` quality. Confirms the MVP-1
data pipeline is not the cause of MVP-1's per-cell quality blur.

## Setup

- Branch / commit at training: TBD (filled in pre-run from `git rev-parse HEAD`).
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
- Environment (planned): AWS Tesla T4, torch 2.12.0+cu130, fp16 single-GPU,
  same venv as MVP-1.

## Run log

### TBD — training run #1

To execute (from `acoustic_ai/`):

```
./.venv/bin/accelerate launch --mixed_precision fp16 \
  layers/layer_a/attempts/lucas__mvp_1_1__spring_night_replica/code/train_audioldm2.py \
  --manifest_path ../resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/spring_night_subset.csv \
  --output_dir ../model/candidates/lucas/mvp_1_1__spring_night_replica \
  --batch_size 4 --num_epochs 5 --learning_rate 1e-5
```

Expected: ~125 steps/epoch × 5 = ~625 steps × ~1.3 s = ~14 min total wall-clock,
~25-30 % of MVP-1's runtime. GPU peak ≈ 8 GB.

(Epoch losses + anomalies appended here after the run.)

### TBD — showcase generation

3 seeds against the inference prompt above:
- seed 42 baseline — exactly the v2 caption from `params.yaml`.
- seed 42 vs smoke_1 — same caption but with smoke_1's prompt verbatim, for
  the regression A/B.
- seed 43 — variation, different temperature bucket.

Artifacts → `dev-artifacts-self-testing/`. After listening, promote the
keepers into `showcase/` and DVC-add.

### TBD — listening verdict

(Filled in after generation.)

## Decision gate

| outcome | next step |
|---|---|
| Quality matches smoke_1 | data pipeline confirmed sound → proceed to Phase 1B + Phase 2 (per-cell) |
| Quality close but not matching | acceptable — proceed but flag for Phase 4 inference sweep |
| Quality clearly worse than smoke_1 | halt — investigate data pipeline (rule 5 strictness, content filter thresholds, ambient segment extraction). Do NOT proceed to per-cell architecture work until resolved. |

## Comparison vs prior attempts

(Filled in after listening.)

## Retrospective

(Written when attempt is closed.)
