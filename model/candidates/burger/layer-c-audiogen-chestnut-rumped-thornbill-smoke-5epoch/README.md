# Layer C AudioGen Chestnut-Rumped Thornbill Smoke 5 Epoch

## Purpose

Smoke-test `facebook/audiogen-medium` + LoRA on Chestnut-rumped Thornbill as a
candidate second successful Layer C species after the Boobook negative result.

This is a smoke candidate, not a production model.

## Data

- Source manifest:
  `resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/train_manifest_chestnut_rumped_thornbill_pass.csv`
- Species: Chestnut-rumped Thornbill / `Acanthiza uropygialis`
- Training rows: 25 manually audited `Pass` clips
- Source audio: shared S3 `downloaded_clips/`, exact event windows extracted
  into `resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/segments/`
- Sample rate: 16 kHz mono WAV for AudioGen

## Training

```bash
MPLCONFIGDIR=/private/tmp/capstone_matplotlib \
acoustic_ai/.venv-audiogen/bin/python acoustic_ai/modules/events/train_audiogen_lora.py \
  --manifest_path resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/train_manifest_chestnut_rumped_thornbill_pass.csv \
  --output_dir model/candidates/burger/layer-c-audiogen-chestnut-rumped-thornbill-smoke-5epoch \
  --num_epochs 5 \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --max_duration_s 10 \
  --device cpu
```

Training completed on CPU: 125 steps, 5 epochs, 25 training clips.

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
  --lora_dir model/candidates/burger/layer-c-audiogen-chestnut-rumped-thornbill-smoke-5epoch \
  --prompt "Chestnut-rumped Thornbill bird call, Bowra dry woodland" \
  --seeds 300,301,302,303,304,305,306,307 \
  --duration 3 \
  --guidance_scale 3.0 \
  --temperature 1.0 \
  --top_k 250 \
  --output_dir debug/layer_c/audiogen/samples \
  --device cpu
```

The training loss was noisier than the successful Fairywren run, so judge this
run by generated sample audit rather than training loss alone.

Manual audit result after expanding to 30 fixed seeds (`300`-`329`):

```text
Samples audited: 30
Pass: 16
Borderline: 0
Fail: 14
Clean pass rate: 53.3%
Usable rate (Pass + Borderline): 53.3%
```

Conclusion: this run is a partial but below-threshold smoke result. It can
generate some usable Chestnut-rumped Thornbill-like calls, but the 30-sample
success rate is far below the 90% Layer C smoke target. Do not report this as
a second successful species without a new data, filtering, or training
strategy.
