---
base_model: /Users/lucas/.cache/huggingface/hub/models--cvssp--audioldm2/snapshots/c8e7e189d324425c05c4c2f81214041ef4107983/unet
library_name: peft
tags:
- lora
- audioldm2
- ecoacoustics
- soundscape-generation
---

# AudioLDM2 LoRA Raw Smoke Checkpoint

This checkpoint is the Layer A AudioLDM2 LoRA smoke-test adapter for generating
quiet ambient soundscape beds from Bowra dry woodland field recordings. It is a
narrow validation checkpoint, not a general seasonal or weather-conditioned
soundscape model.

## Model Details

- **Base model:** `cvssp/audioldm2`
- **Adapter type:** PEFT LoRA on the AudioLDM2 UNet attention projections
- **Checkpoint path:** `acoustic_ai/checkpoints/audioldm2-lora-raw-smoke`
- **Project role:** Layer A ambient site-bed generation smoke test
- **Validated use:** quiet, low-volume environmental ambience with no foreground
  events, music, or machinery
- **Status:** user-validated on 2026-05-06 for the spring-night smoke-test path

## Training Dataset

The adapter was trained from:

```text
resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv
```

Dataset coverage:

- **Site / scene:** Bowra dry woodland, Australia
- **Source type:** raw ecoacoustic field-recording clips
- **Clip count:** 50 clips
- **Total duration:** 621.758 seconds, about 10.36 minutes
- **Clip duration range:** approximately 10-18 seconds
- **Season:** spring only
- **Time of day / diel bin:** night only
- **Recording dates represented:** 2019-09-01, 2019-09-23, 2023-09-12,
  2023-09-28, and 2024-09-09
- **Caption pattern:** `spring night, ambient soundscape, Bowra dry woodland,
  Australia`, with light environmental descriptors such as cool/mild
  temperature, dry air, light breeze or moderate wind, and extended dry spell

This checkpoint should therefore be described as a **spring-night Bowra ambient
smoke model**. It has no training examples for summer, autumn, winter, dawn,
morning, afternoon, evening, wet-season/rain scenes, strong event scenes, or
other sites.

## Scene Coverage And Prompt Scope

Use prompts that stay inside the observed scene distribution:

```text
quiet spring night ambient soundscape, Bowra dry woodland, Australia,
distant environmental bed, no foreground events, no music, no machinery
```

Expected scene characteristics:

- quiet environmental bed
- mostly stationary background texture
- low output level close to the original field-recording loudness
- no deliberate birdsong, insects, voices, vehicles, music, machinery, or rain
  foreground events

Out-of-distribution requests include other seasons, other times of day, other
locations, explicit species or event prompts, heavy rain, strong wind, human
activity, and foreground sound effects.

## Training Procedure

Reproducible training command:

```bash
cd acoustic_ai
./.venv/bin/accelerate launch modules/ambient/diffusion/train_audioldm2.py \
  --manifest_path ../resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv \
  --output_dir checkpoints/audioldm2-lora-raw-smoke \
  --batch_size 1 \
  --num_epochs 5 \
  --learning_rate 1e-5
```

Training notes:

- Raw field-recording levels were preserved by default.
- Do not normalize this smoke dataset to `0.05` RMS; that over-amplifies quiet
  background recorder noise and can produce pulsing or machine-like artifacts.
- If normalization is needed for another experiment, use only mild normalization
  such as `--normalize_audio --target_rms 0.005`.

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
  --prompt "quiet spring night ambient soundscape, Bowra dry woodland, Australia, distant environmental bed, no foreground events, no music, no machinery" \
  --lora_dir checkpoints/audioldm2-lora-raw-smoke \
  --run_name spring_night_raw_smoke_seed42 \
  --seed 42 \
  --num_inference_steps 100 \
  --guidance_scale 2.0 \
  --output_target_rms 0.0015 \
  --highpass_hz 80
```

The development UI/backend path should expose only the seed while this checkpoint
remains trained on the tiny smoke dataset. The AI server owns the fixed prompt,
checkpoint, inference steps, guidance scale, output RMS, and high-pass settings.

## Uses

### Direct Use

Use this adapter for Layer A smoke-test generation of quiet spring-night ambient
beds from the Bowra dry woodland soundscape distribution.

### Downstream Use

This checkpoint can be used in the project development path for validating
AudioLDM2 LoRA integration, seed behavior, metadata reporting, and spectrogram
rendering.

### Out-of-Scope Use

Do not use this checkpoint as evidence that the system can generate all seasons,
all times of day, weather layers, event layers, or arbitrary soundscape prompts.
Those capabilities require broader datasets and separate model or layer support.

## Limitations

- Extremely small dataset: 50 clips and about 10 minutes of audio.
- Scene coverage is limited to spring nights at Bowra dry woodland.
- Environmental descriptors are present in captions, but this checkpoint is not a
  robust controllable weather, time, or season model.
- The model is intended to generate ambient beds only; foreground acoustic events
  belong in separate event/weather layers.
- Outputs should be audited by listening and spectrogram inspection before use in
  demos or reports.

## Framework Versions

- PEFT 0.19.1
