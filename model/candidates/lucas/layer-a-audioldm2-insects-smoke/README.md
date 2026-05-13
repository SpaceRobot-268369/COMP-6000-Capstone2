# layer-a-audioldm2-insects-smoke

## Summary

- Owner: Lucas
- Layer / role: Module A — summer-afternoon insect/cicada ambient bed (smoke 2)
- Status: candidate
- Base model: `cvssp/audioldm2` (frozen UNet; PEFT LoRA on attention projections)
- Trained at: 2026-05-06

## Purpose / hypothesis

Layer A smoke test 2. Validate that a separate per-context LoRA can
specialize the same AudioLDM2 base for a markedly different scene
distribution (hot summer afternoons with cicada/insect texture) without
contaminating the smoke 1 spring-night model. Tests the per-LoRA branching
pattern that Module A will use long-term.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/manifest.csv`
- Source clips / manifests: 35 manually-audited ecoacoustic field-recording clips
- Total duration: ~415.9 s (~6.93 min); clip durations 10.0–18.6 s
- Season / diel: summer + afternoon only
- Recording dates: 2019-12-11, 2020-01-28, 2020-02-01, 2020-02-15, 2024-02-16, 2024-12-12, 2024-12-17, 2025-01-01, 2025-02-13, 2025-02-19
- Caption temperatures present: 36°C, 37°C, 40°C, 41°C, 42°C, 44°C
- Scene focus: insect-rich ambient, cicada and insect texture; hot dry air; light breeze or moderate wind
- Filtering or preprocessing: manually audited after filtering; excludes segments overlapping annotated events and strong-wind rows; raw field-recording levels preserved
- Selection metadata in manifest rows: `insect_score`, `high_ratio`, `low_ratio`, `high_cv`, `rms`
- Known data caveats: deliberately biased toward insect/cicada texture; no examples for spring, autumn, winter, other diel bins, wet-season/rain, or other sites

## Training or promotion context

- Training command: see [.claude/context/ai/runbooks/layer_a_smoke_2_insects.md](../../../../.claude/context/ai/runbooks/layer_a_smoke_2_insects.md)
- Code branch / commit: `model/lucas/layer-c-event-attemp-1` family
- Hardware: single Apple Silicon machine, MPS
- Important settings: see `params.yaml` — 5 epochs, batch_size 1, lr 1e-5; LoRA `r=8, alpha=32, dropout=0.1, target_modules=[to_q, to_k, to_v, to_out.0]`; PEFT 0.19.1

## Artifacts

- Checkpoint binaries: `adapter_model.safetensors` (LoRA adapter only)
- DVC pointer files: `adapter_model.safetensors.dvc`
- Config: `adapter_config.json`
- Params: `params.yaml`
- Metrics: not yet recorded (qualitative inspection only)
- Sample outputs: under `debug/layer_a/audioldm2/samples/audioldm2-lora-insects-smoke/insects_smoke_seed{42,43,44}/`
- Related runbook or log: [.claude/context/ai/runbooks/layer_a_smoke_2_insects.md](../../../../.claude/context/ai/runbooks/layer_a_smoke_2_insects.md)

## Results / metrics

Used by `acoustic_ai/server/inference.py` as the "Dev > Smoke 2" generation
path. Qualitative inspection only.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

- Extremely small dataset (35 clips, ~7 min). Narrower than smoke 1.
- Hot-summer-afternoon-Bowra scope only. Out-of-distribution prompts (other
  seasons, diel bins, cold/wet scenes, non-Bowra sites, bird-focused prompts,
  explicit species, vehicles, music, machinery, strong wind, rain) will not
  generalize.
- Despite temperature/wind descriptors in captions, this is not a controllable
  weather model.
- Layer A bed only; foreground events belong in Layer C.

## Follow-up actions

- Add objective metrics that distinguish insect-texture quality from baseline
  AudioLDM2 outputs.
- Decide whether per-context LoRA branching (one per scene type) is the long-term
  pattern, or whether a single multi-conditioning LoRA could subsume both smoke 1
  and smoke 2.
