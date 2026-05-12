---
base_model: /Users/lucas/.cache/huggingface/hub/models--cvssp--audioldm2/snapshots/c8e7e189d324425c05c4c2f81214041ef4107983/unet
library_name: peft
tags:
- lora
- audioldm2
- ecoacoustics
- insect-soundscape
- soundscape-generation
---

# AudioLDM2 LoRA Insects Smoke Checkpoint

This checkpoint is the Layer A AudioLDM2 LoRA smoke-test-2 adapter for generating
summer afternoon insect/cicada ambient texture from Bowra dry woodland field
recordings. It is a narrow validation checkpoint, not a general seasonal,
diurnal, event, or weather-conditioned soundscape model.

## Model Details

- **Base model:** `cvssp/audioldm2`
- **Adapter type:** PEFT LoRA on the AudioLDM2 UNet attention projections
- **Checkpoint path:** `acoustic_ai/checkpoints/audioldm2-lora-insects-smoke`
- **Project role:** Layer A insect/cicada ambient smoke test
- **Validated use:** hot summer afternoon insect-rich ambience with no birds,
  foreground events, music, machinery, or strong wind
- **Status:** smoke-test checkpoint for the separate insect/cicada Layer A path

## Training Dataset

The adapter was trained from:

```text
resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/manifest.csv
```

Dataset coverage:

- **Site / scene:** Bowra dry woodland, Australia
- **Source type:** manually audited ecoacoustic field-recording clips
- **Clip count:** 35 clips
- **Total duration:** 415.936 seconds, about 6.93 minutes
- **Clip duration range:** 10.008-18.622 seconds
- **Season:** summer only
- **Time of day / diel bin:** afternoon only
- **Scene focus:** insect-rich ambient soundscape, cicada and insect texture
- **Weather/context in captions:** hot dry air, light breeze or moderate wind
- **Caption temperatures represented:** 36C, 37C, 40C, 41C, 42C, and 44C
- **Recording dates represented:** 2019-12-11, 2020-01-28, 2020-02-01,
  2020-02-15, 2024-02-16, 2024-12-12, 2024-12-17, 2025-01-01, 2025-02-13,
  and 2025-02-19
- **Selection metadata:** rows include `insect_score`, `high_ratio`,
  `low_ratio`, `high_cv`, and `rms` audit features

This checkpoint should therefore be described as a **summer-afternoon Bowra
insect/cicada ambient smoke model**. It has no training examples for spring,
autumn, winter, night, dawn, morning, evening, wet-season/rain scenes, other
sites, or non-insect ambient beds.

## Scene Coverage And Prompt Scope

Use prompts that stay inside the observed scene distribution:

```text
summer afternoon insect-rich ambient soundscape, cicada and insect texture,
Bowra dry woodland, Australia, dry hot air, distant environmental bed,
no birds, no foreground events, no music, no machinery, no strong wind
```

Expected scene characteristics:

- hot summer afternoon ambience
- stationary or near-stationary insect/cicada texture
- distant environmental bed rather than foreground event playback
- no deliberate birds, voices, vehicles, music, machinery, rain, or strong wind
- low output level for use as a Layer A bed

Out-of-distribution requests include other seasons, other times of day, cold or
wet scenes, non-Bowra sites, bird-focused prompts, explicit species calls,
vehicles, human activity, music, machinery, strong wind, rain, and foreground
sound effects.

## Training Procedure

Reproducible training command:

```bash
cd acoustic_ai
./.venv/bin/accelerate launch modules/ambient/diffusion/train_audioldm2.py \
  --manifest_path ../resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/manifest.csv \
  --output_dir checkpoints/audioldm2-lora-insects-smoke \
  --batch_size 1 \
  --num_epochs 5 \
  --learning_rate 1e-5
```

Training notes:

- The dataset was manually audited after filtering.
- The dataset is intentionally small and insect/cicada-focused.
- Keep generated samples separate from the raw spring-night smoke checkpoint.
- Raw field-recording levels should be preserved unless a deliberate experiment
  requires mild normalization.

LoRA configuration:

- **Rank (`r`):** 8
- **Alpha:** 32
- **Dropout:** 0.1
- **Target modules:** `to_q`, `to_k`, `to_v`, `to_out.0`
- **PEFT version:** 0.19.1

## Recommended Inference

Use low classifier-free guidance and keep generated audio quiet:

```bash
cd acoustic_ai
./.venv/bin/python modules/ambient/diffusion/sample_audioldm2.py \
  --prompt "summer afternoon insect-rich ambient soundscape, cicada and insect texture, Bowra dry woodland, Australia, dry hot air, distant environmental bed, no birds, no foreground events, no music, no machinery, no strong wind" \
  --lora_dir checkpoints/audioldm2-lora-insects-smoke \
  --run_name insects_smoke_seed42 \
  --seed 42 \
  --num_inference_steps 100 \
  --guidance_scale 2.0 \
  --output_target_rms 0.0015 \
  --highpass_hz 80
```

Expected sample bundles should remain under the checkpoint-named directory:

```text
debug/layer_a/audioldm2/samples/audioldm2-lora-insects-smoke/insects_smoke_seed42/
```

The development endpoint for this checkpoint should stay locked to the
insect/cicada smoke-test prompt while the model is trained on this tiny dataset.

## Uses

### Direct Use

Use this adapter for Layer A smoke-test generation of summer afternoon
insect/cicada ambient beds from the Bowra dry woodland distribution.

### Downstream Use

This checkpoint can be used to validate the project path for a separate
insect/cicada ambient layer, including seed behavior, metadata reporting,
sample-folder separation, and spectrogram rendering.

### Out-of-Scope Use

Do not use this checkpoint as evidence that the system can generate all seasons,
all diel bins, all weather conditions, all taxa, or arbitrary soundscape prompts.
Those capabilities require broader datasets and separate model or layer support.

## Limitations

- Extremely small dataset: 35 clips and about 6.93 minutes of audio.
- Scene coverage is limited to hot summer afternoons at Bowra dry woodland.
- The dataset is deliberately biased toward insect/cicada texture.
- The model is not a robust controllable temperature, wind, season, or time model
  despite the caption descriptors.
- The model is intended to generate ambient beds only; foreground acoustic events
  belong in separate event/weather layers.
- Outputs should be audited by listening and spectrogram inspection before use in
  demos or reports.

## Framework Versions

- PEFT 0.19.1
