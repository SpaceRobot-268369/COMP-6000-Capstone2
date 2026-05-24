# Layer C AudioGen Southern Boobook Smoke 5 Epoch

## Purpose

Smoke-test `facebook/audiogen-medium` + LoRA on a second Layer C bird-call
species after the Splendid Fairywren smoke pass. This checks whether the same
model route transfers beyond one species.

This is a smoke candidate, not a production model.

## Data

- Source manifest:
  `resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/train_manifest_boobook_pass.csv`
- Species: Southern Boobook / `Ninox boobook`
- Training rows: 24 manually audited `Pass` clips
- Source audio: shared S3 `downloaded_clips/`, exact event windows extracted
  into `resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/segments/`
- Sample rate: 16 kHz mono WAV for AudioGen

## Training

```bash
MPLCONFIGDIR=/private/tmp/capstone_matplotlib \
acoustic_ai/.venv-audiogen/bin/python acoustic_ai/modules/events/train_audiogen_lora.py \
  --manifest_path resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/train_manifest_boobook_pass.csv \
  --output_dir model/candidates/burger/layer-c-audiogen-boobook-smoke-5epoch \
  --num_epochs 5 \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --max_duration_s 10 \
  --device cpu
```

Training completed on CPU: 120 steps, 5 epochs, 24 training clips.

## Artifacts

- `adapter_model.safetensors` - LoRA adapter weights
- `adapter_config.json` - PEFT adapter config
- `training_metadata.json` - training summary
- `params.yaml` - human-readable run params

## Smoke Evaluation

Generate fixed-seed samples and audit them manually:

```bash
MPLCONFIGDIR=/private/tmp/capstone_matplotlib \
acoustic_ai/.venv-audiogen/bin/python acoustic_ai/modules/events/sample_audiogen_lora.py \
  --lora_dir model/candidates/burger/layer-c-audiogen-boobook-smoke-5epoch \
  --prompt "Southern Boobook owl call, nocturnal Bowra woodland" \
  --seeds 200,201,202,203,204,205,206,207 \
  --duration 3 \
  --guidance_scale 3.0 \
  --temperature 1.0 \
  --top_k 250 \
  --output_dir debug/layer_c/audiogen/samples \
  --device cpu
```

The smoke decision should be based on the generated sample audit sheet, not on
training loss alone.

Manual audit result:

```text
Samples audited: 8
Pass: 0
Borderline: 0
Fail: 8
Clean pass rate: 0.0%
Usable rate (Pass + Borderline): 0.0%
```

Conclusion: this Boobook LoRA run is a negative smoke result. The training
pipeline and artifact creation worked, but the generated samples were not
usable Southern Boobook calls. Do not expand this run to more seeds without a
separate prompt, duration, or data strategy change.
