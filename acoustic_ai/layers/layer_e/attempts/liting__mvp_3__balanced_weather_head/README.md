# Layer E MVP 3 — Balanced Weather Head

Owner: `liting`

## Purpose

Train a stronger E-B weather analysis head over frozen PANNs CNN14 evidence and
explainable DSP features. This is the follow-up to
`liting__mvp_2__calibrated_weather_head`.

This attempt does **not** fine-tune PANNs. PANNs remains a frozen AudioSet
weather feature extractor. The trainable part is two class-balanced MLP heads
for:

- rain intensity: `none`, `light`, `moderate`, `heavy`
- wind intensity: `none`, `light`, `moderate`, `strong`

Thunder remains suppressed unless Site257 thunder evidence becomes available.

## Why This Attempt Exists

MVP-2 passed the E-B gate with rain accuracy 0.769, wind accuracy 0.731, and
joint accuracy 0.615. MVP-3 tests whether a slightly stronger but still cheap
head can improve joint rain+wind performance before spending time on full
backbone fine-tuning.

## Server B Runtime Target

Expected split:

- feature extraction: depends on number of clips and PANNs device
- actual head training: less than 5 minutes
- evaluation/report writing: a few minutes

The Server B job should report these timings separately so reviewers can see
that the trainable stage is small.

## Outputs

```text
model/candidates/liting/mvp_3__balanced_weather_head/weather_head.pt
model/candidates/liting/mvp_3__balanced_weather_head/metrics.json
debug/e_b_weather_mvp3/
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
