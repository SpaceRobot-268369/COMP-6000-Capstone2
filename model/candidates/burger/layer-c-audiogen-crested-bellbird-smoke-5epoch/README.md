# layer-c-audiogen-crested-bellbird-smoke-5epoch

## Summary

- Owner: burger
- Layer / role: Layer C event generation
- Status: candidate
- Base model: `facebook/audiogen-medium`
- Trained at: 2026-05-24 21:05:13 ACST

## Purpose / hypothesis

Train a Crested Bellbird LoRA adapter as the third species candidate for the
Layer C bird-call generation path. This run tests whether the relaxed quality
pool plus manual audit can provide enough clean foreground clips for a stable
species-specific AudioGen LoRA.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/`
- Source clips / manifests:
  `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/train_manifest_crested_bellbird_pass_relaxed_top50.csv`
- Filtering or preprocessing: relaxed Layer C quality filter, then manual audit;
  only `Pass` rows from the top-50 Bellbird candidate sheet were included.
- Known data caveats: source clips are short event crops from BirdNET-labelled
  annotations; generated quality still requires fixed-seed human audit.

## Training or promotion context

- Training command:

```bash
acoustic_ai/.venv-audiogen/bin/python -m acoustic_ai.modules.events.train_audiogen_lora \
  --manifest_path resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/train_manifest_crested_bellbird_pass_relaxed_top50.csv \
  --output_dir model/candidates/burger/layer-c-audiogen-crested-bellbird-smoke-5epoch \
  --batch_size 1 \
  --num_epochs 5 \
  --learning_rate 1e-5 \
  --device cpu \
  --max_duration_s 10.0 \
  --seed 42
```

- Code branch / commit: local working tree
- Hardware: CPU
- Runtime: about 22 minutes after model load
- Important settings: LoRA rank 8, alpha 32, dropout 0.1; target modules
  `out_proj`, `linear1`, `linear2`

## Artifacts

- Checkpoint binaries: `adapter_model.safetensors`
- DVC pointer files: not added yet
- Params: `params.yaml`
- Metrics: `training_metadata.json`
- Sample outputs:
  `debug/layer_c/audiogen/samples/layer_c_audiogen_crested_bellbird_smoke_5epoch/`
- Generated sample audit:
  `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/generated_lora_crested_bellbird_5epoch_50seed_sample_audit.csv`
- Generated sample auto-eval:
  `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/generated_lora_crested_bellbird_5epoch_50seed_auto_eval.csv`
- Generated-vs-training distribution eval:
  `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/generated_lora_crested_bellbird_5epoch_50seed_distribution_eval.csv`
- BirdNET embedding similarity:
  `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/generated_lora_crested_bellbird_5epoch_50seed_birdnet_similarity.csv`
- Related runbook or log:
  `.claude/context/branches/layer-c-smoke-eval/layer_c_2_3_species_smoke_plan.md`

## Results / metrics

- Manual input audit: 38 `Pass`, 12 `Fail` from top 50 relaxed Bellbird clips
- Training rows: 38 manually audited `Pass` clips
- Epochs: 5
- Steps: 190
- Final displayed training loss: 3.33
- Fixed-seed generated samples: 50 seeds (`100`-`149`)
- Initial 10-seed audit: 10 `Pass`, 0 `Borderline`, 0 `Fail`
- Automatic sanity check: 48 `pass_auto`, 2 `review`; review seeds were
  `119` and `139` due to low 1-8 kHz energy / high low-frequency energy.
- Generated-vs-training distribution check: 35 `in_distribution`, 15 `review`;
  most review rows had narrower spectral bandwidth than the training reference.
- BirdNET embedding similarity: target species top-1 for 5/50 samples (10.0%);
  target species rank <= 2 for 14/50 samples (28.0%); mean target-centroid
  cosine similarity 0.587440.

## Results analysis / audit

50-seed audit is in progress. The first 10 seeds passed manual audit.
Automatic checks are diagnostic only; manual audit remains the pass/fail
criterion for species correctness.
Distribution review rows should be prioritized for listening, not treated as
automatic failures.
BirdNET embedding similarity is weak for Crested Bellbird; generated samples
often sit closer to the Robin or Fairywren training centroids than the Bellbird
centroid.

## Known limitations

- This is a candidate smoke model, not a production checkpoint.
- The final generated-audio success rate has not been measured yet.
- Training uses a relaxed-filter candidate pool, so the manual `Pass` gate is
  important for avoiding background or off-species events.

## Follow-up actions

- Complete the generated-sample audit sheet.
- If the generated clean pass rate is low, inspect failed seeds before changing
  the model or filter.
