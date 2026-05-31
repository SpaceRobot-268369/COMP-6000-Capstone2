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

### 2026-05-30 — training run #1 (16 cells)

Executed on shinypokemon (tmux `mvp2`) via `code/train_all_cells.sh`. All 16
cells trained cleanly in one pass (resumable loop; no failures). Recipe per
cell: r=8, α=32, 5 epochs, batch 4, lr 1e-5, fp16, RMS-norm off.

Per-cell final-epoch (5/5) losses:

| cell | clips (tr/val) | train_loss | val_loss | notes |
|---|---|---|---|---|
| spring_night | 180/20 | 0.1413 | 0.1004 | strong |
| autumn_night | 180/20 | 0.1412 | 0.1003 | strong |
| autumn_afternoon | 163/18 | 0.1747 | 0.1191 | strong |
| summer_night | 148/16 | 0.1622 | 0.0938 | strong |
| spring_dawn | 93/10 | 0.1261 | 0.0577 | healthy |
| summer_dawn | 87/10 | 0.1345 | 0.0589 | healthy |
| winter_night | 87/10 | 0.1330 | 0.0589 | healthy |
| spring_afternoon | 86/10 | 0.1688 | 0.0580 | healthy |
| spring_morning | 84/9 | 0.1634 | 0.0576 | healthy |
| winter_dawn | 67/8 | 0.1738 | 0.0860 | healthy |
| winter_afternoon | 59/7 | 0.1854 | 0.1032 | healthy |
| autumn_dawn | 53/6 | 0.1062 | **0.1443** | watch — low train/high val (overfit signature), 6-clip val noisy |
| summer_afternoon | 53/6 | 0.1090 | **0.1473** | watch — overfit signature; Phase 2.5 candidate |
| autumn_morning | 34/4 | 0.2061 | 0.0604 | thin |
| summer_morning | 30/3 | 0.2647 | 0.0697 | thin |
| winter_morning | 20/2 | 0.1705 | 0.0993 | thin |

Reference: `mvp_1_1` final val_loss = 0.0565. Most cells land at or below
~0.10; the two `watch` cells stand out on val but have tiny (6-clip) val sets.

### 2026-05-30 — showcase generation

Two passes, each cell generated with its OWN adapter, seed 42, 200 steps,
guidance 2.0, RMS 0.0015, highpass 80:
1. 3-cell A/B (spring_night / summer_afternoon / autumn_night) from
   `params.yaml inference.showcase_prompts`.
2. All-16 sweep — each cell with a representative caption pulled from its own
   `cell_<cell>.csv` manifest (in-distribution env values).
Artifacts → `dev-artifacts-self-testing/` (and `…/all_cells/`).

### 2026-05-30 — listening verdict

**PASSED overall** (developer audit). Per-cell quality holds across the bank;
cells are audibly distinct by season/time-of-day. Residual issues noted by
the reviewer: (1) no long-form generation yet, (2) minor quality variance on
some cells (consistent with the two `watch` cells above). Neither blocks the
decision gate — they become the Phase 2.5 / long-form backlog.

## Decision gate

| outcome | next step | actual |
|---|---|---|
| ≥12 cells smoke-quality + distinct | ship the bank → Phase 3 router; treat any stragglers in 2.5 | **✓ this outcome** |
| 6–11 cells good | Phase 2.5 before shipping | — |
| <6 cells good | per-cell recipe suspect — halt | — |

**Decision: PROCEED to Phase 3 (inference router).** Per-cell architecture
validated — it resolves the shared-LoRA interference that capped mvp_1 /
mvp_1_2. `summer_afternoon` and `autumn_dawn` go on the Phase 2.5 backlog;
long-form generation is a separate capability track. **Not promoted** —
promotion waits for the router (servability) + a validation sign-off.

## Comparison vs prior attempts

| attempt | arch | result |
|---|---|---|
| mvp_1 | shared LoRA r=8, 16 cells | blurred (interference) |
| mvp_1_1 | LoRA r=8, 1 cell | smoke quality ✓ (proved recipe + data) |
| mvp_1_2 | shared LoRA r=64, 16 cells, 8 ep | marginally better, still blurred (ruled out capacity) |
| **mvp_2** | **16 per-cell LoRAs r=8** | **passed — per-cell quality + distinct cells** |

## Retrospective

What worked
- The diagnostic ladder (mvp_1_1 isolating data, mvp_1_2 isolating capacity)
  meant we reached Phase 2 *knowing* interference was the cause, not guessing.
  The per-cell bank then worked first try.
- Pre-build dry-runs (Step 0) caught that §6.1 content filters were dropping
  nothing and wind was the real thin-cell lever — saved wasted effort.
- Resumable per-cell loop made the 16-train run a single unattended pass.

What I'd do differently / open items
- 2 `watch` cells (summer_afternoon, autumn_dawn) trained on 53 clips show an
  overfit signature; consider fewer epochs or augmentation for sub-60-clip
  cells in Phase 2.5.
- Training script still overwrites `README.md` per cell (not retained in the
  candidate freeze); fix to `MODEL_CARD.md` before any rerun.
- Long-form (10-min) generation needs the tiled+crossfade generator — the
  base model only does ~10 s coherently.

## Closeout

- Status: **closed — passed gate; candidate frozen, not promoted.**
- Tag: `attempt/lucas__mvp_2__per_cell_loras`.
- Checkpoints: 16 adapters DVC-tracked under
  `model/candidates/lucas/mvp_2__per_cell_loras/<cell>/` + pushed to S3.
- Successor: Phase 3 inference router; Phase 2.5 augmentation; long-form bed.
