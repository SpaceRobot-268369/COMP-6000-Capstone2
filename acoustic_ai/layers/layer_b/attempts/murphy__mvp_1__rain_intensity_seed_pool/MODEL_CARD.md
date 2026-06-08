# Model Card: murphy__mvp_1__rain_intensity_seed_pool

## Purpose

Layer B rain-only smoke generator for short site-derived rain stems. This is a
research smoke attempt, not a general weather generator.

## Model

- Base model: `cvssp/audioldm2`
- Adapter: LoRA under `model/candidates/murphy/mvp_1__rain_intensity_seed_pool/`
- Runtime prompt and generation settings are locked in `params.yaml`.
- Raw model output is 16 kHz; validated BWE postprocess exports 24 kHz WAVs.

## Data

- Source: site_257 real rain/rain+wind recordings
- Training pool: 72 curated 5 s clips
- Labels: `rain=68`, `rain+wind=4`
- Split: recording-group train/validation manifests

## Runtime Contract

- `seed_mode=curated`
- `uses_intensity=true`
- `intensities=[light, heavy]`
- `default_intensity=light`
- `good_seeds_by_intensity` and flat `good_seeds` are stored in `params.yaml`
  and mirrored in `acoustic_ai/registry.yaml`.

The AI service is the final seed arbiter:

- A request seed in `good_seeds` is used directly for reproducibility.
- Missing, invalid, or non-whitelisted seeds are replaced by a random reviewed
  seed from `good_seeds`.
- The resolved seed is returned in response metadata.

Seed robustness is intentionally limited to the reviewed whitelist. Arbitrary
seed quality is not guaranteed for this smoke attempt.

## Intensity Labels

`light` and `heavy` are curated bins, not model-conditioned controls. Because
outputs are RMS-normalised, the labels describe perceived texture, density, and
spectral balance rather than loudness. Binning evidence is recorded in
`good_seeds_audit.csv` with features and human listening acceptance.

## Showcase

The formal showcase uses 10 representative reviewed seeds:

- light: `42`, `43`, `44`, `51`, `999983`
- heavy: `46`, `48`, `2000000001`, `123456789`, `69317`

WAVs are DVC-tracked; spectrogram PNGs and metadata JSON are git-tracked.

## Validation

A6 retest output:

`spectrum_diagnostic_20260606/bwe_prototype/a6_good_seed_showcase_retest_20260607/`

Objective gates passed against the 72-clip real-rain reference:

- 8-11 kHz gap vs real: -2.48 dB
- 8-11 minus 2-8 kHz drop: -13.70 dB
- 2-8 kHz gap vs real: -0.75 dB

## Limitations

- Smoke-stage model with a small, clustered rain pool.
- Quality depends on current checkpoint, prompt, postprocess, and BWE settings.
- Changing model weights, prompt/postprocess behavior, BWE parameters, or seed
  policy requires re-auditing `good_seeds_audit.csv` and refreshing showcase.
