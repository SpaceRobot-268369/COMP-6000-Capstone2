# layer-c-audiogen-red-capped-robin-smoke-5epoch

## Summary

- Owner: burger
- Layer / role: Layer C event generation
- Status: candidate
- Base model: `facebook/audiogen-medium`
- Trained at: 2026-05-24 18:46:16 ACST

## Purpose / hypothesis

Train a Red-capped Robin LoRA adapter as the second species candidate for the
Layer C bird-call generation path. This run tests whether strict-filtered,
manually audited foreground event clips are sufficient for species-specific
AudioGen LoRA generation.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/`
- Source clips / manifests:
  `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/train_manifest_red_capped_robin_pass_strict.csv`
- Filtering or preprocessing: strict Layer C quality filter, then manual audit;
  only `Pass` rows were included.
- Known data caveats: 24 clips is a small smoke-scale dataset, so generated
  stability must be judged by fixed-seed audio audit rather than training loss.

## Training or promotion context

- Training command:

```bash
acoustic_ai/.venv-audiogen/bin/python -m acoustic_ai.modules.events.train_audiogen_lora \
  --manifest_path resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/train_manifest_red_capped_robin_pass_strict.csv \
  --output_dir model/candidates/burger/layer-c-audiogen-red-capped-robin-smoke-5epoch \
  --batch_size 1 \
  --num_epochs 5 \
  --learning_rate 1e-5 \
  --device cpu \
  --max_duration_s 10.0 \
  --seed 42
```

- Code branch / commit: local working tree
- Hardware: CPU
- Runtime: about 8 minutes after model load
- Important settings: LoRA rank 8, alpha 32, dropout 0.1; target modules
  `out_proj`, `linear1`, `linear2`

## Artifacts

- Checkpoint binaries: `adapter_model.safetensors`
- DVC pointer files: not added yet
- Params: `params.yaml`
- Metrics: `training_metadata.json`
- Sample outputs:
  `debug/layer_c/audiogen/samples/layer_c_audiogen_red_capped_robin_smoke_5epoch/`
- Generated sample auto-eval:
  `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/generated_lora_red_capped_robin_5epoch_50seed_auto_eval.csv`
- Generated-vs-training distribution eval:
  `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/generated_lora_red_capped_robin_5epoch_50seed_distribution_eval.csv`
- BirdNET embedding similarity:
  `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/generated_lora_red_capped_robin_5epoch_50seed_birdnet_similarity.csv`
- Related runbook or log:
  `.claude/context/branches/layer-c-smoke-eval/layer_c_2_3_species_smoke_plan.md`

## Results / metrics

- Training rows: 24 manually audited `Pass` clips
- Epochs: 5
- Steps: 120
- Final displayed training loss: 2.76
- Generated sample audit sheet:
  `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/generated_lora_red_capped_robin_5epoch_10seed_sample_audit.csv`
- 10-seed generated sample audit:
  - Samples audited: 10
  - Pass: 9
  - Borderline: 0
  - Fail: 1
  - Clean pass rate: 90.0%
  - Usable rate: 90.0%
- 50-seed generated sample audit:
  - Audit sheet:
    `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/generated_lora_red_capped_robin_5epoch_50seed_sample_audit.csv`
  - Samples audited: 50
  - Pass: 44
  - Borderline: 0
  - Fail: 6
  - Clean pass rate: 88.0%
  - Usable rate: 88.0%
- Automatic sanity check: 50 `pass_auto`, 0 `review`
- Generated-vs-training distribution check: 14 `in_distribution`, 36 `review`;
  most review rows had narrower spectral bandwidth than the training reference.
- BirdNET embedding similarity: target species top-1 for 28/50 samples (56.0%);
  target species rank <= 2 for 50/50 samples (100.0%); mean target-centroid
  cosine similarity 0.629232.

## Results analysis / audit

Manual audit remains the pass/fail criterion for species correctness.
Automatic checks are diagnostic only. The 50-seed sanity check found no obvious
silence, clipping, or low-energy failures, but the distribution check suggests
the generated Robin samples often occupy a narrower spectral shape than the
strict training clips.

## Known limitations

- This is a candidate smoke model, not a production checkpoint.
- The training set is small and species success must be validated by generated
  sample audit.
- Borderline audit rows were excluded to avoid teaching noisy or ambiguous
  events.

## Follow-up actions

- Add a small number of audited relaxed Robin candidates or test a slightly
  different sampling setting if the project requires a strict 90% 50-seed pass
  threshold.
