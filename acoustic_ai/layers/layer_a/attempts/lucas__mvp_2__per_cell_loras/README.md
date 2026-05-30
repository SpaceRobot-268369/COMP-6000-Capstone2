# mvp_2__per_cell_loras

## Summary

- Owner: Lucas
- Layer / role: Layer A — ambient bed (per-cell LoRA bank)
- Status: candidate (Phase 2)
- Base model: cvssp/audioldm2 (frozen)
- Trained at: _pending_
- Parent attempts: `lucas__mvp_1__audioldm2_all_conditioned`,
  `lucas__mvp_1_1__spring_night_replica`,
  `lucas__mvp_1_2__shared_lora_maxed`

## Purpose / hypothesis

The MVP-1 and MVP-1.2 attempts proved that **one shared LoRA cannot cover
all 16 (season, diel) cells** without per-cell quality blur — and that the
cause is **gradient interference between cells**, not capacity (r=64 barely
helped) or data (mvp_1_1 hit smoke quality on a single cell).

This attempt trains **16 independent LoRAs**, one per cell, each with the
exact recipe that worked for `mvp_1_1` (r=8, α=32, 5 epochs). No shared
weights ⇒ no interference. At inference (Phase 3) a router selects the
adapter matching the requested (season, diel) and merges it onto the base.

> Success = each cell sounds like its own `mvp_1_1`-quality ambient bed, and
> different cells sound audibly distinct.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/mvp2_per_cell_dataset/`
  (1,583 clips; cap 200, wind ≤5.5 m/s, rain strict 0.1mm, caption v2).
  Separate from the MVP-1 dataset, which stays frozen.
- Per-cell manifests: `cell_<season>_<diel>.csv`, derived from `manifest.csv`
  by `code/split_cell_manifests.py` (written alongside `manifest.csv` so the
  training script's project_root derivation stays correct).
- Clip counts and tiers per cell: see [params.yaml](./params.yaml). 13 cells
  ≥59 clips; 3 morning cells are source-scarce (autumn 38, summer 33, winter
  22) and flagged for Phase 2.5 augmentation.

## Training context

- Orchestrator: `code/train_all_cells.sh` — splits the manifest, then trains
  each cell (resumable; skips cells whose adapter already exists).
- Per-cell command (what the loop runs, from `acoustic_ai/`):
  ```bash
  ./.venv/bin/accelerate launch --mixed_precision fp16 \
    layers/layer_a/attempts/lucas__mvp_2__per_cell_loras/code/train_audioldm2.py \
    --manifest_path ../resources/site_257_bowra-dry-a/mvp2_per_cell_dataset/cell_<cell>.csv \
    --output_dir ../model/candidates/lucas/mvp_2__per_cell_loras/<cell> \
    --batch_size 4 --num_epochs 5 --learning_rate 1e-5 --lora_r 8 --lora_alpha 32
  ```
- Hardware: AWS EC2 Tesla T4 (`shinypokemon`).
- Expected runtime: ~50–60 min for all 16 (strong cells ~5 min, thin ~1–2 min).

## Artifacts

- Checkpoints: `model/candidates/lucas/mvp_2__per_cell_loras/<cell>/adapter_model.safetensors`
  (16 adapters, each DVC-tracked).
- Metrics: per-cell final-epoch losses recorded in DEVLOG; `metrics.json`
  aggregated after the run.
- Sample outputs: 3-cell showcase in `dev-artifacts-self-testing/`, curated
  keepers into `showcase/`.

## Results / metrics

Pending — see DEVLOG.

## Results analysis / audit

Pending — written in the DEVLOG retrospective.

## Known limitations

- 16 adapters to store/route (vs one) — handled by the Phase 3 router; storage
  is ~16 × 12 MB ≈ 190 MB, all DVC-tracked.
- 3 thin morning cells may overfit at 22–38 clips; treated reactively in
  Phase 2.5 (augmentation) only if they fail the listening audit.
- Caption v2 retains the date field; whether date helps per-cell is untested
  but low-stakes (each LoRA sees one cell).

## Follow-up actions

- Train the bank → record per-cell losses → 3-cell showcase → listen.
- Decision gate in [DEVLOG.md](./DEVLOG.md).
- Phase 3: inference router (`handler.py`) keyed on (season, diel).
