# murphy__smoke_1__audioldm2_wind

Layer B wind-only generation smoke attempt.

This attempt explores a pure generate path for Layer B using AudioLDM2 + LoRA,
reusing the validated Layer A training and inference pipeline.

## Scope

- Single weather element only: `wind`.
- No rain generation in this attempt.
- No mixed weather (`rain+wind`) conditioning in this attempt.
- No Layer D timeline/mixing logic in this attempt.

## Status

Scaffolded. The attempt includes copied training/inference code from
`layer_a/lucas__smoke_1__audioldm2_spring_night` as a starting point.

Next steps:

1. Build a wind-only training manifest from Layer B weather pool.
2. Lock prompt and tune params for wind texture generation.
3. Train smoke LoRA checkpoint.
4. Wire registry + handler contract for Layer B.

## Notes

- Runtime contract target: seed-driven generation with server-side locked prompt.
- Checkpoints should be stored under:
  `model/candidates/murphy/smoke_1__audioldm2_wind/`.
