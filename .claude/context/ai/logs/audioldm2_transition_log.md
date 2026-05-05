# Switching to AudioLDM2

## Decision
We are switching the foundational approach for generating the "ambient site bed" (Layer A) from a custom VAE + Latent Diffusion setup to **AudioLDM2** (via Hugging Face `diffusers`).

## Rationale
- AudioLDM2 provides a robust, pre-trained text-to-audio foundation that significantly accelerates the capability to generate high-quality soundscapes.
- Training our own VAE + diffusion models from scratch requires massive data and compute. Fine-tuning a pre-trained foundation model is more practical.

## Technical Approach
1. **Foundation Model**: `cvssp/audioldm2`
2. **Fine-tuning**: We will use **Low-Rank Adaptation (LoRA)** on the UNet attention layers. This allows us to train on standard hardware (MPS/GPU) without modifying the billions of parameters in the base model.
3. **Data**: The model uses the existing `smoking_test_dataset` (`resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv`).

## Current Working Checkpoint
- Current checkpoint: `acoustic_ai/checkpoints/audioldm2-lora-raw-smoke`
- Base model: `cvssp/audioldm2`
- Sample output bundle: `debug/layer_a/audioldm2/samples/spring_night_raw_smoke_seed42/`
- Status: user validation says this model works well for Layer A ambient beds, with only minor issues.
- Branch status: this branch is one attempted Layer A implementation, and it has succeeded for the smoke test. If this branch is merged into `main`, update the broader architecture, pipeline, and handoff docs so they consistently describe AudioLDM2 LoRA as the main Layer A path rather than a branch-local validation attempt.
- Expected sound: quiet environmental ambience similar to the sample data, low volume, mostly stationary, without foreground events or obvious generated-machine texture.
- Deprecated checkpoint: `acoustic_ai/checkpoints/audioldm2-lora-rms005-smoke`. Do not use it for quality testing; it was trained with `--target_rms 0.05`, which over-amplified quiet recordings and produced pulsing/machine-like artifacts.

## Fixed Dev Generation Contract
- The smoke model was trained on a very small dataset, so Layer A dev generation is intentionally narrow.
- Fixed prompt:
  `quiet spring night ambient soundscape, Bowra dry woodland, Australia, distant environmental bed, no foreground events, no music, no machinery`
- Standard generation settings:
  - `num_inference_steps=100`
  - `guidance_scale=2.0`
  - `audio_length_in_s=10`
  - `output_target_rms=0.0015`
  - `highpass_hz=80`
- Frontend contract: expose only seed selection for the dev UI. Do not allow user-specified prompts yet.
- Express backend contract: forward only `{ seed }` to the AI server.
- FastAPI contract: own the fixed prompt and model settings, and return prompt/checkpoint/seed/steps/guidance/RMS/high-pass metadata.

Seed behavior:
- The seed initializes the random noise used by the diffusion process.
- `spring_night_raw_smoke_seed42`, `seed43`, and `seed44` are generated from the same model, prompt, and parameters; they differ because the seed changes the starting noise.
- Reusing the same seed should reproduce effectively the same audio when model, prompt, settings, code path, library versions, and device are unchanged.
- Use non-negative integer seeds. The practical portable range is `0` to `2147483647`.
- Seed is not temperature. This AudioLDM2 Layer A path does not expose temperature; the available controls are prompt, checkpoint, guidance scale, inference steps, audio length, and seed.

## Integration Details
- Dependencies added: `diffusers`, `accelerate`, `peft`
- Code Location: `acoustic_ai/modules/ambient/diffusion/`
  - `audioldm2_dataset.py`: PyTorch `Dataset` that reads the manifest, loads audio waveforms, preserves quiet field-recording levels by default, and returns raw caption strings for `AudioLDM2Pipeline.encode_prompt`.
  - `train_audioldm2.py`: The `accelerate`-based LoRA fine-tuning script.
  - `sample_audioldm2.py`: The inference script that merges the base model with the LoRA adapter for generating soundscapes.

## Smoke Model Training Test
Use this workflow to reproduce the successful Layer A smoke-test model from
`resources/site_257_bowra-dry-a/smoking_test_dataset/`.

1. Train the smoke LoRA:
   ```bash
   cd acoustic_ai
   ./.venv/bin/accelerate launch modules/ambient/diffusion/train_audioldm2.py \
     --manifest_path ../resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv \
     --output_dir checkpoints/audioldm2-lora-raw-smoke \
     --batch_size 1 \
     --num_epochs 5 \
     --learning_rate 1e-5
   ```
   Keep smoke tests at raw field-recording level by default. The Bowra smoke clips
   are intentionally quiet (roughly 0.001-0.003 RMS); normalizing them to 0.05 RMS
   over-amplifies recorder noise and can make the LoRA collapse into a pulsing,
   machine-like texture. If normalization is needed, use `--normalize_audio` with a
   mild `--target_rms 0.005`.

2. Generate three validation audios and mel-spectrograms:
   ```bash
   for seed in 42 43 44; do
     ./.venv/bin/python modules/ambient/diffusion/sample_audioldm2.py \
       --prompt "quiet spring night ambient soundscape, Bowra dry woodland, Australia, distant environmental bed, no foreground events, no music, no machinery" \
       --lora_dir checkpoints/audioldm2-lora-raw-smoke \
       --run_name spring_night_raw_smoke_seed${seed} \
       --seed ${seed} \
       --num_inference_steps 100 \
       --guidance_scale 2.0 \
       --output_target_rms 0.0015 \
       --highpass_hz 80
   done
   ```
   The sample script writes its WAV, spectrogram PNG, and metadata JSON under
   project-root `debug/layer_a/audioldm2/samples/` unless `--output_path` is set.
   It defaults to a quiet environmental output level (`--output_target_rms 0.0015`)
   and high-passes sub-bass rumble at 80 Hz.

Expected outputs:
```text
debug/layer_a/audioldm2/samples/spring_night_raw_smoke_seed42/
  generated_ambient.wav
  generated_ambient_spectrogram.png
  generated_ambient_metadata.json
debug/layer_a/audioldm2/samples/spring_night_raw_smoke_seed43/
  generated_ambient.wav
  generated_ambient_spectrogram.png
  generated_ambient_metadata.json
debug/layer_a/audioldm2/samples/spring_night_raw_smoke_seed44/
  generated_ambient.wav
  generated_ambient_spectrogram.png
  generated_ambient_metadata.json
```

Acceptance target: all three WAVs should sound like quiet, environmental ambient
beds at sample-data volume, without foreground events, music, machinery, pulsing,
or obvious generated texture. Inspect the spectrogram PNGs for mostly stationary
energy and no strong periodic bands.

Spectrogram comparison note: `sample_audioldm2.py` and the dev
frontend/backend response both render Layer A spectrogram PNGs through
`modules.ambient.diffusion.layer_a_visualization`, using the same log-mel
parameters and image settings. If the CLI and frontend images do not visually
align, first verify both services are restarted and the compared WAVs were
generated with the same seed/model/prompt/settings.

3. Backend/frontend integration status:
   - `acoustic_ai/server/inference.py` exposes fixed-prompt Layer A AudioLDM2 generation with `audioldm2-lora-raw-smoke`.
   - `acoustic_ai/server/server.py` serves `/layer_a/generate` from that fixed AI path, not retrieval.
   - `backend/src/index.js` proxies only `{ seed }` to the AI server so prompt/env fields cannot reach Layer A.
   - `frontend/src/pages/LayerATestPage.jsx` exposes only seed selection for the dev test UI; user-specified prompts are disabled at this stage.
