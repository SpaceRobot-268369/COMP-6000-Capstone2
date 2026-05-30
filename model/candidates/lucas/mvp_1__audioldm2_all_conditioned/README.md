# mvp_1__audioldm2_all_conditioned

## Summary

- Owner: Lucas
- Layer / role: Layer A — ambient bed
- Status: candidate
- Base model: cvssp/audioldm2 (frozen)
- Trained at: _pending first training run_

## Purpose / hypothesis

First MVP-stage Layer A model. The smoke tests showed that AudioLDM2 + LoRA
can reproduce a single narrow scene (smoke_1: spring night; smoke_2: summer
afternoon insects). MVP-1 keeps the **same method** and asks the next
question:

> Can one LoRA adapter cover the whole clean ambient pool from site 257 if
> captions encode (season, diel_bin, temperature, humidity, wind, date)?

Success = the model produces audibly different ambient beds when the caption
condition fields change (e.g. a "spring night" caption sounds night-like; an
"autumn afternoon, warm, light breeze" caption sounds afternoon-like) without
regressing the smoke_1/smoke_2 scenes.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/`
  (~1,082 clips after `--per-cell-cap 100` balancing)
- Source clips / manifests:
  - Source segment pool: `acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_segments/` (DVC)
  - Built by `script/dataset/build_mvp1_all_conditioned_dataset.py`
- Filtering:
  - `wind_speed < 4.5 m/s` AND `wind_max < 8 m/s`
  - `precipitation < 0.1 mm`
  - `duration ≥ 10 s`
  - No overlap with annotated events (`downloaded_annotations/`)
  - Per-cell cap: 100 clips per (season, diel_bin) cell
- Known data caveats: site 257 only (no second site online yet); summer-afternoon
  cell is undersized (29 raw clips). Annotation-overlap exclusion is approximate
  (rec-id schemes between annotation files and env data may not fully align —
  see `script/dataset/build_mvp1_all_conditioned_dataset.py` for the join logic).

## Training or promotion context

- Training command (CUDA, from `acoustic_ai/`):
  ```bash
  ./.venv/bin/accelerate launch \
    layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/code/train_audioldm2.py \
    --manifest_path ../resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest.csv \
    --output_dir ../model/candidates/lucas/mvp_1__audioldm2_all_conditioned \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-5
  ```
- Code branch / commit: `model/lucas/layer-a-mvp-1-all-conditioned`
- Hardware: CUDA GPU (host TBD; trained manually, no Server A worker)
- Runtime: _pending_
- Important settings: see [params.yaml](./params.yaml). Raw audio levels
  preserved (no RMS normalisation) — see smoke_1 negative-result note.

## Artifacts

- Checkpoint binaries: `adapter_model.safetensors` (DVC, after training)
- DVC pointer files: `adapter_model.safetensors.dvc` (after training)
- Params: [params.yaml](./params.yaml)
- Metrics: _Not evaluated yet_
- Sample outputs:
  - Expected: `<attempt>/expected/real_<clip_id>/` — TBD
  - Showcase: `<attempt>/showcase/seed_<N>_<label>/` — TBD
- Related runbook or log: _runbook to be added after first successful training._

## Results / metrics

Not evaluated yet.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

- Untrained.
- Tentative: with ~1k clips across 16 (season, diel) cells, the LoRA may
  not have enough capacity / data per cell to clearly differentiate cells.
  If first results show "average ambient" output insensitive to caption
  conditions, fall back to a smaller subset (e.g. spring-night + autumn-night
  only) or split into per-cell LoRAs.

## Follow-up actions

- Run first training on CUDA host.
- Generate showcase samples across ≥3 condition combinations (e.g. spring
  night / summer afternoon insects / autumn night) and compare against the
  smoke checkpoints.
- If conditioning works, update `code/handler.py` to accept condition fields
  from the dev endpoint (currently only `seed` is exposed — see
  [CLAUDE.md § Layer A dev-generation contract](../../../../CLAUDE.md)).
- Add `metrics.json` once an evaluation protocol exists.
