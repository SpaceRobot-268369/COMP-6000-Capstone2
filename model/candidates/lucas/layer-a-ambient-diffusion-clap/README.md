# layer-a-ambient-diffusion-clap

## Summary

- Owner: Lucas
- Layer / role: Module A — CLAP-conditioned latent diffusion (Method X smoke, superseded)
- Status: deprecated
- Base model: Custom latent diffusion conditioned on 512-dim CLAP text embedding (`laion/clap-htsat-unfused`)
- Trained at: 2026-05-05

## Purpose / hypothesis

CLAP-conditioned latent diffusion smoke test. Intentional overfit on 50
spring/night/September clips to validate that CLAP-text guidance over a
custom latent diffusion stack could produce coherent ambient beds.
Superseded by AudioLDM2 LoRA, which gave higher-quality output with a
frozen pretrained base.

## Dataset / inputs

- Dataset: 50 spring/night/September clips from `resources/site_257_bowra-dry-a/smoking_test_dataset/`
- Filtering or preprocessing: intentional overfit (500 epochs on 50 clips); CLAP text caption per clip
- Known data caveats: tiny dataset by design; do not generalize beyond the spring-night-Bowra prompt scope

## Training or promotion context

- Training command: `python3 acoustic_ai/modules/ambient/diffusion/train_clap.py` (DEFAULT_OUT now points at this folder)
- Code branch / commit: pre-AudioLDM2 transition
- Hardware: single Apple Silicon machine, MPS
- Important settings: see `params.yaml` — 500 epochs, batch_size 16, lr 2e-4, cond_dim 512, hidden_dim 512, 6 blocks, v-prediction with cosine schedule

## Artifacts

- Checkpoint binaries: `best.pt`, `last.pt`
- DVC pointer files: `best.pt.dvc`, `last.pt.dvc`
- Params: `params.yaml`
- Metrics: n/a
- Sample outputs: n/a
- Related runbook or log: [.claude/context/ai/logs/audioldm2_transition_log.md](../../../../.claude/context/ai/logs/audioldm2_transition_log.md)

## Results / metrics

Qualitative output was acceptable but inferior to AudioLDM2 LoRA at similar
prompts. The decision to drop this path is documented in the AudioLDM2
transition log.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

Deprecated. Tiny training set, narrow scope, custom architecture without the
ecosystem benefits of a frozen pretrained base. Do not use for current Layer A.

## Follow-up actions

None. Kept for archival reference.
