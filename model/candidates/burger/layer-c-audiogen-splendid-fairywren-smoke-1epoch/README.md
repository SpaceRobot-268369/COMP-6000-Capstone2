# Layer C AudioGen Splendid Fairywren Smoke 1 Epoch

## Purpose

Smoke-test LoRA for Layer C bird-call generation using audited Splendid
Fairywren snippets from site 257 Bowra dry woodland.

This is not a production model. It is a fast smoke candidate trained within the
1.5-day Layer C reset plan.

## Data

- Source manifest:
  `resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/train_manifest_splendid_fairywren_pass.csv`
- Species: Splendid Fairywren / `Malurus splendens`
- Training rows: 24 manually audited `Pass` clips
- Source audio: S3 shared `downloaded_clips/`, exact event windows extracted
  into `resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/segments/`
- Sample rate: 16 kHz mono WAV for AudioGen

## Training

```bash
acoustic_ai/.venv-audiogen/bin/python acoustic_ai/modules/events/train_audiogen_lora.py \
  --manifest_path resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/train_manifest_splendid_fairywren_pass.csv \
  --output_dir model/candidates/burger/layer-c-audiogen-splendid-fairywren-smoke-1epoch \
  --num_epochs 1 \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --max_duration_s 10 \
  --device cpu
```

Training completed on CPU: 24 steps, 1 epoch.

## Artifacts

- `adapter_model.safetensors` - LoRA adapter weights
- `adapter_config.json` - PEFT adapter config
- `training_metadata.json` - training summary
- `params.yaml` - human-readable run params

## Smoke Samples

Generated samples:

```text
debug/layer_c/audiogen/samples/layer_c_audiogen_splendid_fairywren_smoke_1epoch/
```

Seeds generated so far:

- `42`, 5 seconds
- `43`, 3 seconds

Audit sheet:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_sample_audit.csv
```

## Current Status

Layer C smoke has a reliable retrieval baseline:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/smoke_retrieval_pass_set.csv
```

That set contains 42 manually audited `Pass` clips across the three smoke
species. This LoRA is the first generative smoke attempt and should be judged
by listening to the generated sample audit sheet before further training.
