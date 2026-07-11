# Layer E MVP 3 — Balanced Weather Head

Owner: `liting`

## Purpose

Train a stronger E-B weather analysis head over frozen PANNs CNN14 evidence and
explainable DSP features. This is the follow-up to
`liting__mvp_2__calibrated_weather_head`.

This attempt does **not** fine-tune PANNs. PANNs remains a frozen AudioSet
weather feature extractor. The trainable part is a hybrid pair of small heads:

- rain intensity: weighted MLP, `none`, `light`, `moderate`, `heavy`
- wind intensity: weighted linear head, `none`, `light`, `moderate`, `strong`

Thunder remains suppressed unless Site257 thunder evidence becomes available.

## Why This Attempt Exists

MVP-2 passed the E-B gate with rain accuracy 0.769, wind accuracy 0.731, and
joint accuracy 0.615. MVP-3 tests a hybrid head after a quick Server B sweep:
rain benefited from a small MLP, while wind remained more stable with the
linear head. The goal is to improve joint rain+wind performance before spending
time on full backbone fine-tuning.

## Server B Runtime Target

Expected split:

- feature extraction: depends on number of clips and PANNs device
- actual head training: less than 5 minutes
- evaluation/report writing: a few minutes

The Server B job should report these timings separately so reviewers can see
that the trainable stage is small.

## Server B Result

Latest run on the same 101 Site257 materialised clips:

- Rain validation accuracy: 0.885
- Wind validation accuracy: 0.692
- Joint validation accuracy: 0.654
- Actual head training time: 2.30 seconds
- Total runtime including feature extraction: 18.18 seconds
- Gate status: `needs_iteration`

Interpretation: MVP3 improves rain accuracy and joint accuracy over MVP2, but
wind accuracy falls slightly below the 0.70 gate. MVP3 is therefore a useful
iteration checkpoint, while MVP2 remains the more stable default candidate for
integration unless the team prefers the higher rain/joint trade-off.

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
