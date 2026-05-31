# mvp_1_1__spring_night_replica

## Summary

- Owner: Lucas
- Layer / role: Layer A — ambient bed (diagnostic candidate)
- Status: candidate (Phase 1A diagnostic)
- Base model: cvssp/audioldm2 (frozen)
- Trained at: _pending first training run_
- Parent attempt: `lucas__mvp_1__audioldm2_all_conditioned`

## Purpose / hypothesis

Diagnostic for the Layer A quality-push roadmap. Trains a smoke-style single-
scene LoRA (r=8, 5 epochs) on the spring-night subset of the MVP-1 manifest to
test whether the MVP-1 data filter pipeline can support smoke-level quality
when capacity isn't spread across 16 cells.

> If a smoke-replica on MVP-1 data matches `lucas__smoke_1` quality, the data
> pipeline is fine and MVP-1's blur is purely a capacity problem. If not, the
> data pipeline needs fixing before any architecture changes.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/spring_night_subset.csv`
  (100 rows: 90 train + 10 val, filtered from the MVP-1 manifest by
  `season=spring AND diel_bin=night`)
- Source binaries unchanged — same `clips.dvc` as MVP-1.
- Filtering: identical to MVP-1 (rules 1–7 + §6.1 content filters).
- Caption schema: v2 (date INCLUDED) — same as MVP-1, intentionally so the
  comparison is single-variable.

## Training context

- Command (from `acoustic_ai/`):
  ```bash
  ./.venv/bin/accelerate launch --mixed_precision fp16 \
    layers/layer_a/attempts/lucas__mvp_1_1__spring_night_replica/code/train_audioldm2.py \
    --manifest_path ../resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/spring_night_subset.csv \
    --output_dir ../model/candidates/lucas/mvp_1_1__spring_night_replica \
    --batch_size 4 --num_epochs 5 --learning_rate 1e-5
  ```
- Hardware: AWS EC2 Tesla T4 (host `shinypokemon`).
- Expected runtime: ~14 min wall-clock (~125 steps/epoch × 5 × 1.3 s/step).
- Code branch / commit: `model/lucas/layer-a-mvp-1-all-conditioned` (TBD sha).
- Important settings: see [params.yaml](./params.yaml).

## Artifacts

- Checkpoint: `adapter_model.safetensors` (DVC after training).
- Pointer: `adapter_model.safetensors.dvc`.
- Metrics: `metrics.json` (TBD — epoch-by-epoch losses, runtime, GPU peak).
- Sample outputs:
  - Showcase: at the attempt folder `acoustic_ai/.../lucas__mvp_1_1__spring_night_replica/showcase/`.
  - DEVLOG with listening notes: same folder, `DEVLOG.md`.

## Results / metrics

Pending — see DEVLOG.

## Results analysis / audit

Pending — written in the DEVLOG retrospective.

## Known limitations

- Single (season, diel) cell only — does not test conditioning across cells.
- Caption v2 (with date) is preserved deliberately; date-drop is a separate
  decision tested in Phase 1B / MVP-2.

## Follow-up actions

- Train → write `metrics.json` → generate showcase → fill DEVLOG retrospective.
- Decision gate documented in [DEVLOG.md](../../../acoustic_ai/layers/layer_a/attempts/lucas__mvp_1_1__spring_night_replica/DEVLOG.md).
