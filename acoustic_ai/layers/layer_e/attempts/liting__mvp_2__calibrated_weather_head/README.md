# Layer E MVP 2 — Calibrated Weather Head

Owner: `liting`

## Purpose

Train a small E-B weather calibration head over frozen PANNs CNN14 evidence and
explainable DSP features. This is the follow-up to
`liting__mvp_1__panns_weather_baseline`.

This attempt does **not** fine-tune PANNs. PANNs remains a frozen AudioSet
weather feature extractor. The trainable part is a tiny linear head for:

- rain intensity: `none`, `light`, `moderate`, `heavy`
- wind intensity: `none`, `light`, `moderate`, `strong`

Thunder remains suppressed unless Site257 thunder evidence becomes available.

## Why This Attempt Exists

The MVP-1 Site257-only bounded validation passed the policy-aligned gate, but
many mixed or intensity-adjacent cases landed in `partial` or `boundary`.
MVP-2 tests whether a small calibrated head can improve confidence and bucket
selection without training a heavy model.

## Server B Runtime Target

Expected split:

- feature extraction: depends on number of clips and PANNs device
- actual head training: less than 5 minutes
- evaluation/report writing: a few minutes

The Server B job should report these timings separately so reviewers can see
that the trainable stage is small.

## Outputs

```text
model/candidates/liting/mvp_2__calibrated_weather_head/weather_head.pt
model/candidates/liting/mvp_2__calibrated_weather_head/metrics.json
debug/e_b_weather_mvp2/
```

Binary checkpoints are DVC-tracked after the run. Metrics and README metadata
are git-tracked.

## Analysis Policy Output

The registry handler returns both the older frontend-friendly `summary` block
and the current analysis-policy `observations.weather` block:

- `observations.weather.wind.summary`
- `observations.weather.rain.summary`
- `observations.weather.thunder`

Wind and rain include a label, numeric intensity, confidence, coverage, and a
placeholder variability value. Thunder is explicitly returned as suppressed
until Site257 thunder evidence is validated.
