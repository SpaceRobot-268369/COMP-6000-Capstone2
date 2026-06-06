# murphy__smoke_2__audioldm2_wind

Layer B wind-only generation smoke attempt (round 2 fix).

This attempt addresses smoke_1 artifacts (electronic timbre) by aligning with
Layer A validated training recipe and tightening data filtering.

## Scope

- Single weather element only: `wind`
- Pure generate route (no retrieval at runtime)
- Focus: reduce synthetic/electronic artifacts

## Status

Closed for smoke-stage MVP showcase selection.

- Final locked inference profile: Variant A (`denoise_strength=0.15`, `denoise_floor_ratio=0.40`)
- User-approved good seeds: `48, 50, 52, 55, 59, 72`
- Final listening entry: `showcase_s3a4_final/listen_generated.html`
- Full 40-seed scan archive: `showcase_s3a4_final/`

## Expected outputs

- Candidate checkpoint under `model/candidates/murphy/smoke_2__audioldm2_wind/`
- Final curated MVP showcase set in the retained S3a.4 final showcase
- Archived 40-seed audit batch for traceability (seed 42-81)
