# layer-a-audioldm2-rms005-smoke

## Summary

- Owner: Lucas
- Layer / role: Module A — deprecated high-RMS variant of smoke 1
- Status: deprecated (documented negative result)
- Base model: `cvssp/audioldm2` (frozen UNet; PEFT LoRA on attention projections)
- Trained at: 2026-05 (pre-smoke-1)

## Purpose / hypothesis

Early attempt at Layer A smoke training that normalized the Bowra spring-night
dataset to `target_rms = 0.05` before training, on the assumption that
louder/normalized inputs would help the LoRA latch onto a useful ambient
distribution. **The hypothesis failed.** Retained as a documented negative
result so future training runs don't repeat the mistake.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv` (same as smoke 1)
- Filtering or preprocessing: **normalized to `target_rms = 0.05`** — this is the differentiating setting and the cause of the failure
- Known data caveats: Bowra recordings are quiet by nature; aggressive RMS normalization over-amplifies recorder background noise

## Training or promotion context

- Training command: same as smoke 1 (`train_audioldm2.py`) but with `--normalize_audio --target_rms 0.05`
- Code branch / commit: pre-smoke-1 attempt
- Hardware: single Apple Silicon machine, MPS
- Important settings: see `params.yaml` — 5 epochs, batch_size 1, lr 1e-5; LoRA `r=8, alpha=32, dropout=0.1, target_modules=[to_q, to_k, to_v, to_out.0]`; PEFT 0.19.1

## Artifacts

- Checkpoint binaries: `adapter_model.safetensors` (LoRA adapter only)
- DVC pointer files: `adapter_model.safetensors.dvc`
- Config: `adapter_config.json`
- Params: `params.yaml` (records `target_rms: 0.05` as the failure-cause setting)
- Metrics: n/a
- Sample outputs: not retained
- Related runbook or log: see [.claude/context/ai/runbooks/layer_a_smoke_1_spring_night.md](../../../../.claude/context/ai/runbooks/layer_a_smoke_1_spring_night.md) which explicitly references this as the deprecated companion

## Results / metrics

Failed qualitatively. Outputs were pulsing / machine-like, with the background
recorder noise amplified into a foreground texture. No objective metrics
recorded.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

- **Do not use for any quality testing or comparison.** The output distribution
  is not representative of the data the system is supposed to produce.
- The `audioldm2-lora-rms005` short name is explicitly called out in
  `inference.py` and the smoke 1 runbook as a deprecated checkpoint.

## Follow-up actions

- None. Kept for archival reference and as a documented negative result.
- If a future audit wants to publish the failure mode (over-amplification →
  pulsing artifacts), reuse this candidate's params + sample regeneration.
