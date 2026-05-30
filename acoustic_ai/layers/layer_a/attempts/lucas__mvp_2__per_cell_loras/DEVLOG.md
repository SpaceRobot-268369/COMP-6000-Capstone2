# DEVLOG — lucas__mvp_2__per_cell_loras

## Hypothesis

16 independent per-cell LoRAs (one per (season, diel), each the proven mvp_1_1
recipe: r=8, α=32, 5 epochs) eliminate the gradient interference that blurred
the shared-LoRA attempts (mvp_1, mvp_1_2). Each cell should reach mvp_1_1-level
quality, and different cells should sound audibly distinct.

## Why we're here (evidence chain)

| attempt | arch | result | what it ruled out |
|---|---|---|---|
| mvp_1_1 | r=8, 1 cell | smoke quality ✓ | data pipeline is fine |
| mvp_1 | r=8, 16 cells shared | blurred | — |
| mvp_1_2 | r=64, 16 cells shared, 8 ep | marginally better, still blurred | capacity is not the main bottleneck |

Conclusion: the bottleneck is **shared-weight gradient interference** across
cells. Per-cell LoRAs remove it by construction.

## Setup

- Branch / commit at training: `model/lucas/layer-a-mvp-1-all-conditioned`
  (recorded at tag `attempt/lucas__mvp_2__per_cell_loras`).
- Dataset: `resources/site_257_bowra-dry-a/mvp2_per_cell_dataset/`
  (Phase 2 Step 0 build — cap 200, wind ≤5.5, rain 0.1, caption v2).
- Per-cell manifests: `cell_<season>_<diel>.csv` via
  `code/split_cell_manifests.py`.
- Orchestration: `code/train_all_cells.sh` (tmux session `mvp2`).
- Environment (planned): AWS Tesla T4, torch 2.12.0+cu130, fp16 single-GPU.

## Run log

### TBD — training run #1 (16 cells)

To execute on server (from `~/lucano/COMP-6000-Capstone2/acoustic_ai/`, in
tmux `mvp2`):

```
git pull --ff-only
dvc pull resources/site_257_bowra-dry-a/mvp2_per_cell_dataset/clips.dvc
bash layers/layer_a/attempts/lucas__mvp_2__per_cell_loras/code/train_all_cells.sh
```

Per-cell final-epoch losses (filled after run):

| cell | clips (tr/val) | train_loss | val_loss | notes |
|---|---|---|---|---|
| spring_night | 180/20 | | | |
| autumn_night | 180/20 | | | |
| autumn_afternoon | 163/18 | | | |
| summer_night | 148/16 | | | |
| spring_dawn | 93/10 | | | |
| summer_dawn | 87/10 | | | |
| winter_night | 87/10 | | | |
| spring_afternoon | 86/10 | | | |
| spring_morning | 84/9 | | | |
| winter_dawn | 67/8 | | | |
| winter_afternoon | 59/7 | | | |
| autumn_dawn | 53/6 | | | |
| summer_afternoon | 53/6 | | | |
| autumn_morning | 34/4 | | | thin |
| summer_morning | 30/3 | | | thin |
| winter_morning | 20/2 | | | thin |

### TBD — showcase generation

3-cell A/B (seed 42), prompts from `params.yaml inference.showcase_prompts`:
spring_night, summer_afternoon, autumn_night. Each generated with its OWN
cell adapter. Artifacts → `dev-artifacts-self-testing/`.

### TBD — listening verdict

(Filled after generation. Per-cell quality + cross-cell distinctness.)

## Decision gate

| outcome | next step |
|---|---|
| ≥12 cells smoke-quality + distinct | ship the bank → Phase 3 router; treat any stragglers in 2.5 |
| 6–11 cells good | Phase 2.5 (augmentation for thin/failing cells) before shipping |
| <6 cells good | per-cell recipe itself is suspect — halt and investigate (unlikely given mvp_1_1) |

## Comparison vs prior attempts

(Filled after listening.)

## Retrospective

(Written when attempt is closed.)
