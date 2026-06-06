# murphy__smoke_1__audioldm2_rain

Layer B rain-only generation smoke attempt.

This attempt validates whether a true generator can learn rain texture from the
first small, real-site rain pool:

`debug/murphy_layer_b_rain_smoke_training_pool_v0_20260606`

## Scope

- Single weather element only: `rain`
- Pure generate route (no retrieval at runtime)
- Smoke objective: verify learnable rain texture, not final controllability

## Data

- Source pool: 72 clips (5 s, 22.05 kHz mono), site_257 real recordings only
- Labels: rain=68, rain+wind=4
- Split: recording-group train/val manifests (no random split)
- Caption policy:
  - rain -> pure-rain caption
  - rain+wind -> rain-with-light-wind caption

## Runtime contract

- Runtime input: `seed` only
- Server-locked: prompt, negative prompt, guidance, steps, duration, postprocess
- Duration is aligned to training clip length (5 s)

## Expected outputs

- Candidate checkpoint under `model/candidates/murphy/smoke_1__audioldm2_rain/`
- Smoke showcase seeds and listening notes
- Manifest and run metadata for reproducibility
