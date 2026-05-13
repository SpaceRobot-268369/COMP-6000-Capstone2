# layer-a-audioldm2-raw-smoke

## Summary

- Owner: Lucas
- Layer / role: Module A — spring-night ambient bed (smoke 1)
- Status: candidate (user-validated 2026-05-06)
- Base model: `cvssp/audioldm2` (frozen UNet; PEFT LoRA on attention projections)
- Trained at: 2026-05-06

## Purpose / hypothesis

Layer A smoke test 1. Validate that a PEFT LoRA over the frozen AudioLDM2
UNet can produce quiet, low-volume, environmentally-plausible ambient beds
from a tiny Bowra spring-night dataset. Success criterion: outputs are
stationary, low-level, and free of foreground events when prompted within
distribution.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv`
- Source clips / manifests: 50 raw ecoacoustic field-recording clips, Bowra dry woodland
- Total duration: ~621.8 s (~10.4 min); clip durations 10–18 s
- Season / diel: spring + night only
- Recording dates: 2019-09-01, 2019-09-23, 2023-09-12, 2023-09-28, 2024-09-09
- Caption pattern: `spring night, ambient soundscape, Bowra dry woodland, Australia` + light environmental descriptors (cool/mild temperature, dry air, light breeze or moderate wind, extended dry spell)
- Filtering or preprocessing: raw field-recording levels preserved (no normalization)
- Known data caveats: no examples for other seasons, other diel bins, wet-season/rain, strong events, or other sites — model is **narrow** to spring-night Bowra ambient

## Training or promotion context

- Training command: see [.claude/context/ai/runbooks/layer_a_smoke_1_spring_night.md](../../../../.claude/context/ai/runbooks/layer_a_smoke_1_spring_night.md)
- Code branch / commit: `model/lucas/layer-c-event-attemp-1` family
- Hardware: single Apple Silicon machine, MPS
- Important settings: see `params.yaml` — 5 epochs, batch_size 1, lr 1e-5; LoRA `r=8, alpha=32, dropout=0.1, target_modules=[to_q, to_k, to_v, to_out.0]`; PEFT 0.19.1

## Artifacts

- Checkpoint binaries: `adapter_model.safetensors` (LoRA adapter only)
- DVC pointer files: `adapter_model.safetensors.dvc`
- Config: `adapter_config.json`
- Params: `params.yaml`
- Metrics: not yet recorded (qualitative listening + spectrogram inspection only)
- Sample outputs: under `debug/layer_a/audioldm2/samples/audioldm2-lora-raw-smoke/spring_night_raw_smoke_seed{42,43,44}/`
- Related runbook or log: [.claude/context/ai/runbooks/layer_a_smoke_1_spring_night.md](../../../../.claude/context/ai/runbooks/layer_a_smoke_1_spring_night.md), [.claude/context/ai/logs/audioldm2_transition_log.md](../../../../.claude/context/ai/logs/audioldm2_transition_log.md)

## Results / metrics

User-validated on 2026-05-06: outputs are quiet, environmental-like ambient
beds with only minor issues. Used by `acoustic_ai/server/inference.py` as the
"Dev > Smoke 1" generation path. No objective metrics recorded.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

- Extremely small dataset (50 clips, ~10 min). Mode coverage is narrow by design.
- Spring-night-Bowra scope only. Out-of-distribution prompts (other seasons,
  diel bins, sites, weather, events, music, machinery, strong wind) will not
  generalize and should not be attempted from this candidate.
- The dev endpoint is locked to the fixed smoke prompt; do not expose
  arbitrary user prompts on this checkpoint.
- Not a controllable weather/time/season model despite caption descriptors.

## Follow-up actions

- Add objective metrics (e.g. spectral statistics vs. dataset, listening-test
  scores) so a future audit isn't pure listening.
- Compare against the insects-smoke candidate to characterize what each LoRA
  has actually specialized in.
