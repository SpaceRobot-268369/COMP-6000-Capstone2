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

- Runtime input: optional `seed` only
- Server-locked: prompt, negative prompt, guidance, steps, duration, postprocess
- Duration is aligned to training clip length (5 s)
- Seed mode is curated. The accepted seed whitelist is split into two
  user-facing intensity bins: `light` and `heavy`.
- The AI service is the sole seed arbiter. If the request seed is missing,
  invalid, or outside `good_seeds`, the server picks one reviewed seed from the
  whitelist and writes the resolved value back to response metadata.
- Seed robustness is intentionally limited to audited seeds. This attempt does
  not claim arbitrary seed stability; only the reviewed whitelist is exposed for
  smoke-stage use.
- This intensity split is curated, not model-conditioned. Because runtime output
  is RMS-normalised, intensity means rain texture, density, and spectral balance,
  not volume.
- The current split is `light=13` and `heavy=15`; the evidence is recorded in
  `good_seeds_audit.csv` with feature metrics plus human listening labels.
- Raw AudioLDM2 output is 16 kHz; this attempt applies the validated rain BWE
  postprocess and delivers showcase/runtime WAVs at 24 kHz.
- Postprocess order: BWE mix -> 80 Hz high-pass -> RMS match -> 20 ms fade ->
  peak limit if needed. Spectral denoise is kept available but disabled.

The whitelist and intensity labels are bound to the current checkpoint and BWE
parameters. Changing model weights, prompt/postprocess settings, or BWE
parameters requires re-auditing the seed whitelist and bins.

## Sample-rate strategy

The rain showcase/attempt output is intentionally 24 kHz so the BWE stage can
restore perceptually useful rain brightness above the original 8 kHz Nyquist
ceiling. Layer A still outputs 16 kHz, but Layer D mixing is currently a
placeholder and does not block this smoke attempt.

Future Layer D work must choose a common mix sample rate and explicitly
resample layer stems before summing/exporting the final soundscape.

## Expected outputs

- Candidate checkpoint under `model/candidates/murphy/smoke_1__audioldm2_rain/`
- Good-seed showcase bundle at `showcase/` using 10 curated representatives:
  `42`, `43`, `44`, `51`, `999983`, `46`, `48`, `2000000001`, `123456789`,
  and `69317`. WAVs are DVC-tracked; spectrogram PNGs and metadata JSON are
  git-tracked.
- Manifest and run metadata for reproducibility
- Curated seed audit table at `good_seeds_audit.csv`

## A6 validation

The refreshed good-seed showcase was retested in
`spectrum_diagnostic_20260606/bwe_prototype/a6_good_seed_showcase_retest_20260607/`.
Objective gates passed against the 72-clip real-rain reference:

- 8-11 kHz gap vs real: -2.48 dB
- 8-11 minus 2-8 kHz drop: -13.70 dB
- 2-8 kHz gap vs real: -0.75 dB

This closes the showcase loop for the current checkpoint, BWE parameters, and
curated seed whitelist.
