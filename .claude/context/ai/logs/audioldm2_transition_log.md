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

## Integration Details
- Dependencies added: `diffusers`, `accelerate`, `peft`
- Code Location: `acoustic_ai/modules/ambient/diffusion/`
  - `audioldm2_dataset.py`: PyTorch `Dataset` that reads the manifest, loads audio waveforms, and tokenizes captions using both CLAP and T5 tokenizers.
  - `train_audioldm2.py`: The `accelerate`-based LoRA fine-tuning script.
  - `sample_audioldm2.py`: The inference script that merges the base model with the LoRA adapter for generating soundscapes.

## Next Steps for Execution
1. Run the training script:
   ```bash
   cd acoustic_ai
   ./.venv/bin/accelerate launch modules/ambient/diffusion/train_audioldm2.py --manifest_path ../resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv --output_dir checkpoints/audioldm2-lora
   ```
   Keep smoke tests at raw field-recording level by default. The Bowra smoke clips
   are intentionally quiet (roughly 0.001-0.003 RMS); normalizing them to 0.05 RMS
   over-amplifies recorder noise and can make the LoRA collapse into a pulsing,
   machine-like texture. If normalization is needed, use `--normalize_audio` with a
   mild `--target_rms 0.005`.
2. Generate a test sample:
   ```bash
   ./.venv/bin/python modules/ambient/diffusion/sample_audioldm2.py --prompt "spring night, ambient soundscape, Bowra dry woodland, Australia, cool (15°C), dry air, light breeze, extended dry spell" --lora_dir checkpoints/audioldm2-lora
   ```
   The sample script writes its WAV, spectrogram PNG, and metadata JSON under
   project-root `debug/layer_a/audioldm2/samples/` unless `--output_path` is set.
   It defaults to a quiet environmental output level (`--output_target_rms 0.0015`)
   and high-passes sub-bass rumble at 80 Hz.
3. Once validated, update `acoustic_ai/server/inference.py` to use this new pipeline instead of the old latent retrieval method.
