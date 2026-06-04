# murphy__smoke_2__audioldm2_wind

Layer B wind-only generation smoke attempt (round 2 fix).

This attempt addresses smoke_1 artifacts (electronic timbre) by aligning with
Layer A validated training recipe and tightening data filtering.

## Scope

- Single weather element only: `wind`
- Pure generate route (no retrieval at runtime)
- Focus: reduce synthetic/electronic artifacts

## Status

Scaffolded from `murphy__smoke_1__audioldm2_wind` with smoke_2 parameter and
manifest-filter strategy updates.

## Expected outputs

- Candidate checkpoint under `model/candidates/murphy/smoke_2__audioldm2_wind/`
- 20-sample showcase batch for manual audit (seed 42-61)
