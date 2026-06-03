# murphy__mvp_1__weather_direct_detection

Layer E-B weather analysis MVP. This attempt analyzes an uploaded audio mixture
and reports which weather elements are present, how strong they are, and how
confident the detector is.

This is an analysis feature only. It does not generate audio, select Layer B
assets, or decompose the mixture into separated stems.

## Goal

Given one uploaded audio clip, return a structured weather-layer analysis:

- overall weather label: `none`, `rain`, `wind`, `thunder`, `rain+wind`,
  `rain+thunder`, `wind+thunder`, or `rain+thunder+wind`
- per-element presence for `rain`, `wind`, and `thunder`
- per-element intensity: `none`, `light`, `medium`, or `heavy`
- per-element confidence in `[0, 1]`
- warnings for likely ambiguity, contamination, overload, or low confidence

If no clear weather layer is detected, the result must be `overall_label: none`
with every element marked absent.

## MVP Method

Direct detection on the mixture:

1. Preprocess uploaded audio to mono and a fixed analysis sample rate.
2. Split long audio into short overlapping windows.
3. Score each window with pre-trained analysis models:
   - CLAP audio-text similarity for open-vocabulary weather prompts.
   - PANNs or YAMNet AudioSet tags for weather cross-checks.
4. Compute lightweight acoustic features as sanity checks:
   RMS, peak, clipping, spectral centroid, spectral flatness, spectral entropy,
   low-frequency energy, and high-frequency energy.
5. Fuse model scores and feature checks into per-window decisions.
6. Aggregate windows into a clip-level result.

CLAP and AudioSet models are the primary detectors. Acoustic features explain
and calibrate the result; they are not standalone classifiers.

## Non-Goals

- No source separation.
- No Layer B retrieval or pool selection.
- No generated weather audio.
- No Layer D mixing.
- No species or event analysis.
- No training new weather model for MVP.

## Files

- `schema.md` — output contract.
- `weather_analysis_policy.md` — scoring, fusion, warnings, and MVP thresholds.
- `params.yaml` — initial analysis parameters.
- `code/` — implementation will be added after the schema/policy step.
- `dev-artifacts-self-testing/` — local/serverB smoke outputs; gitignored except
  `.gitkeep`.

## Smoke Test

Run from the repo root on serverB or an environment with the AI dependencies:

```bash
./acoustic_ai/.venv/bin/python \
  acoustic_ai/layers/layer_e/attempts/murphy__mvp_1__weather_direct_detection/code/run_weather_analysis.py \
  /path/to/input.wav \
  --out acoustic_ai/layers/layer_e/attempts/murphy__mvp_1__weather_direct_detection/dev-artifacts-self-testing/weather_smoke.json
```

The smoke passes when:

- the command exits successfully
- output JSON follows `schema.md`
- `debug.model_scores_available` is `true` in the real CLAP environment

If model dependencies are unavailable, the command should still complete with
`model_scores_unavailable`; that only validates the fallback path, not E-B
accuracy.
