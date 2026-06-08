# layer-c-sa3-spotted-nightjar-core30-human-bandpass-step1200

## Summary

- Owner: burger
- Layer / role: Layer C event generation, Spotted Nightjar
- Status: candidate
- Base model: Stable Audio 3 small-sfx-base
- Trained at: not recorded in this artifact

## Purpose / hypothesis

Stable Audio 3 LoRA candidate for live Layer C generation of isolated Spotted
Nightjar calls. The registered runtime uses this checkpoint for request-time
generation before species-specific post-processing.

## Dataset / inputs

- Dataset: Spotted Nightjar SA3 LoRA training set
- Source clips / manifests: not recorded in this artifact
- Filtering or preprocessing: core-30 human bandpass training setup
- Known data caveats: detailed training data provenance is not recorded here yet

## Training or promotion context

- Training command: not recorded in this artifact
- Code branch / commit: not recorded in this artifact
- Hardware: not recorded in this artifact
- Runtime: Stable Audio 3 LoRA training/inference stack
- Important settings: checkpoint step 1200; see `params.yaml`

## Artifacts

- Checkpoint binaries:
  - `lora_checkpoints/epoch=39-step=1200.ckpt`
- DVC pointer files:
  - `lora_checkpoints/epoch=39-step=1200.ckpt.dvc`
- Params: `params.yaml`
- Metrics: not provided
- Sample outputs:
  - Expected: `acoustic_ai/layers/layer_c/attempts/burger__mvp_3__sa3_generative_live/expected/spotted_nightjar/`
  - Showcase: not provided
- Related runbook or log: not provided

## Results / metrics

Not evaluated yet.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

- Requires Stable Audio 3 dependencies, model access, and a GPU worker.
- Training provenance and objective metrics are not recorded in this artifact yet.

## Follow-up actions

- Add training command, source manifest, hardware, runtime, and evaluation notes.
- Add showcase samples when the attempt artifact tier is completed.
