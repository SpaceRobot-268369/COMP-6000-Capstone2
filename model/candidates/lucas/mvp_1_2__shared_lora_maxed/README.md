# mvp_1_2__shared_lora_maxed

## Summary

- Owner: Lucas
- Layer / role: Layer A — ambient bed (capacity diagnostic)
- Status: candidate (Phase 1B)
- Base model: cvssp/audioldm2 (frozen)
- Trained at: _pending_
- Parent: `mvp_1__audioldm2_all_conditioned`,
  `mvp_1_1__spring_night_replica`

## Purpose / hypothesis

Phase 1B capacity diagnostic. Tests whether one shared LoRA at r=64 (8×
MVP-1's r=8) trained for 8 epochs can cover all 16 (season, diel) cells
of the MVP-1 manifest without per-cell quality blur. Caption v3 (date
dropped) is the secondary variable.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/`
  (~1,082 clips; same DVC blob as MVP-1).
- Training manifest: `manifest_v3.csv` — `manifest.csv` with `recorded
  YYYY-MM-DD` stripped from the `caption` column. `recording_date` column
  preserved for logging.

## Training context

See [params.yaml](./params.yaml). Train command in the attempt DEVLOG.
Hardware: AWS EC2 Tesla T4 (`shinypokemon`).

## Artifacts

- Checkpoint: `adapter_model.safetensors` (DVC after training).
- Pointer: `adapter_model.safetensors.dvc`.
- Params snapshot: [params.yaml](./params.yaml).
- Metrics: `metrics.json` (after training).

## Results / metrics

Pending.

## Results analysis / audit

Pending.

## Known limitations

- Caption v3 strips date — degradation could mean lost date signal OR
  capacity-bound. Anchored against MVP-1.1 (caption v2, single cell).
- Thin cells (summer afternoon, 29 clips) unchanged — Phase 2.5 expands.

## Follow-up actions

- See attempt
  `acoustic_ai/layers/layer_a/attempts/lucas__mvp_1_2__shared_lora_maxed/DEVLOG.md`
  for run log + decision gate.
