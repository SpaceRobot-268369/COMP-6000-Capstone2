# murphy__mvp_1__weather_direct_detection

Layer E-B weather analysis MVP. This attempt analyzes an uploaded audio mixture
and reports which weather elements are present, how strong they are, and how
confident the detector is.

This is an analysis feature only. It does not generate audio, select Layer B
assets, or decompose the mixture into separated stems.

## Goal

Given one uploaded audio clip, return a structured weather-layer analysis:

- aggregator-facing continuous summaries for `rain`, `wind`, and `thunder`
- overall weather label: `none`, `rain`, `wind`, `thunder`, `rain+wind`,
  `rain+thunder`, `wind+thunder`, or `rain+thunder+wind`
- per-element presence for `rain`, `wind`, and `thunder`
- per-element intensity: `none`, `light`, `medium`, or `heavy`
- per-element confidence in `[0, 1]`
- warnings for likely ambiguity, contamination, overload, or low confidence

If no clear weather layer is detected, the result must be `overall_label: none`
with every element marked absent.

For the Layer E aggregator, the authoritative weather output is
`observations.weather`: continuous 0-1 `summary` fields for wind/rain/thunder,
plus optional timeline fields. The older top-level `weather` block remains for
MVP gate calibration and debug compatibility.

## MVP Method

Direct detection on the mixture:

1. Preprocess uploaded audio to mono and a fixed analysis sample rate.
2. Split long audio into short overlapping windows.
3. Score each window with pre-trained analysis models:
   - CLAP audio-text similarity for open-vocabulary weather prompts.
   - PANNs CNN14 AudioSet tags for broad weather cross-checks.
   - AST AudioSet tags as a conservative guard for ambiguous cases.
4. Compute lightweight acoustic features as sanity checks:
   RMS, peak, clipping, spectral centroid, spectral flatness, spectral entropy,
   low-frequency energy, and high-frequency energy.
5. Fuse model scores and feature checks into per-window decisions.
6. Aggregate windows into a clip-level result.

CLAP is the sensitive weather detector. PANNs and AST are independent evidence
channels used to confirm, guard, or lower confidence. Acoustic features explain
and calibrate the result; they are not standalone classifiers.

## Current Gate v1.1

Gate v1.1 is the frozen MVP fusion rule after Server B calibration and a small
holdout listen check:

- `CLAP + PANNs + AST` is the MVP model stack.
- BEATs was tested as a research direction but is not part of the MVP main path.
- Pure `rain`, `wind`, `thunder`, and `none` cases are handled conservatively.
- `rain+wind` is allowed only when rain evidence is close under wind and PANNs
  or AST provides some rain support.
- Weak mixed-weather rain promotions keep low rain confidence and add
  `possible_rain_under_wind`.
- Thunder remains conservative because wind overload can sound thunder-like.

Calibration summary:

- 12-sample Server B calibration: `10/12` exact after gate v1.1.
- 8-sample holdout: `6/8` exact; the two misses were user-reviewed as wind with
  extremely subtle rain, so the conservative `wind` output is acceptable.

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
  --model-backend clap \
  --audioset-backend panns \
  --guard-backend ast \
  --out acoustic_ai/layers/layer_e/attempts/murphy__mvp_1__weather_direct_detection/dev-artifacts-self-testing/weather_smoke.json
```

The smoke passes when:

- the command exits successfully
- output JSON follows `schema.md`
- `debug.model_scores_available` is `true` in the real CLAP environment

If model dependencies are unavailable, the command should still complete with
`model_scores_unavailable`; that only validates the fallback path, not E-B
accuracy.

## Calibration Report

Use `code/evaluate_weather_outputs.py` after a calibration or holdout run to
compare analysis JSON files against expected labels:

```bash
./acoustic_ai/.venv/bin/python \
  acoustic_ai/layers/layer_e/attempts/murphy__mvp_1__weather_direct_detection/code/evaluate_weather_outputs.py \
  /path/to/manifest.csv \
  --results-dir /path/to/result-jsons \
  --out /path/to/summary.json
```

The manifest must contain one id column (`audio_id`, `id`, `clip_id`,
`sample_id`, or `stem`) and one expected label column (`expected_label`,
`expected`, `label`, or `human_label`). If the manifest includes `result_json`,
`json_path`, or `output_json`, that path is used directly; otherwise the script
looks for matching JSON files in `--results-dir`.

This report is for gate calibration only. It does not run models and does not
change any Layer B pool or runtime assets.
