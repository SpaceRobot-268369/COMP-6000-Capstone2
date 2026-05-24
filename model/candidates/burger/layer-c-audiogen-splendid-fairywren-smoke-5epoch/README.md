# Layer C AudioGen Splendid Fairywren Smoke 5 Epoch

## Purpose

Smoke-test `facebook/audiogen-medium` + LoRA as the selected Layer C
bird-call generation route. This run follows the 1-epoch sanity check but uses
5 epochs so the model choice can be judged by generated audio quality rather
than by pipeline execution alone.

This is a smoke candidate, not a production model.

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
MPLCONFIGDIR=/private/tmp/capstone_matplotlib \
acoustic_ai/.venv-audiogen/bin/python acoustic_ai/modules/events/train_audiogen_lora.py \
  --manifest_path resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/train_manifest_splendid_fairywren_pass.csv \
  --output_dir model/candidates/burger/layer-c-audiogen-splendid-fairywren-smoke-5epoch \
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

Generate several fixed-seed samples and audit them manually:

```bash
MPLCONFIGDIR=/private/tmp/capstone_matplotlib \
acoustic_ai/.venv-audiogen/bin/python acoustic_ai/modules/events/sample_audiogen_lora.py \
  --lora_dir model/candidates/burger/layer-c-audiogen-splendid-fairywren-smoke-5epoch \
  --prompt "Splendid Fairywren bird call, Bowra dry woodland, dawn" \
  --seeds 100,101,102,103,104,105,106,107 \
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
Pass: 7
Borderline: 1
Fail: 0
Clean pass rate: 87.5%
Usable rate (Pass + Borderline): 100.0%
```

Smoke conclusion: this run supports `facebook/audiogen-medium + LoRA` as a
feasible Layer C smoke model route for Splendid Fairywren event generation.
The adapter is good enough for smoke proof, while broader MVP/product work
should add more species, more clean pass clips, and larger generated audits.

## 50-Seed Stability Audit

The same 5-epoch adapter was expanded to 50 fixed-seed samples (`100`-`149`):

```text
debug/layer_c/audiogen/samples/layer_c_audiogen_splendid_fairywren_smoke_5epoch/
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_5epoch_50seed_sample_audit.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_splendid_fairywren_5epoch_50seed_auto_eval.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_splendid_fairywren_5epoch_50seed_distribution_eval.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_splendid_fairywren_5epoch_50seed_birdnet_similarity.csv
```

Final 50-seed audit result:

```text
Samples generated: 50
Samples audited: 50
Pass: 45
Borderline: 5
Fail: 0
Clean pass rate: 90.0%
Usable rate (Pass + Borderline): 100.0%
```

Automatic diagnostic checks:

```text
Sanity check: 50 pass_auto, 0 review
Generated-vs-training distribution check: 9 in_distribution, 41 review
Main distribution review reason: generated samples often have narrower
spectral bandwidth / lower spectral rolloff than the training reference clips.
BirdNET embedding similarity: target species top-1 for 32/50 samples (64.0%);
target species rank <= 2 for 49/50 samples (98.0%); mean target-centroid
cosine similarity 0.647521.
```

Conclusion: the 50-seed audit meets the Layer C smoke target. The selected
`facebook/audiogen-medium + LoRA` route is stable enough to report as the
Layer C generative smoke proof for Splendid Fairywren.

Automatic checks are diagnostic only; manual audit remains the pass/fail
criterion for species correctness. The distribution review rows should be
interpreted as a prompt to inspect spectral shape, not as automatic failures.
