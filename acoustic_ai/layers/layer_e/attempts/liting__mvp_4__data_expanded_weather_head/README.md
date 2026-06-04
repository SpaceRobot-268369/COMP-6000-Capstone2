# Layer E MVP 4 — Data-Expanded Weather Head

Owner: `liting`

## Purpose

Train the next E-B weather analysis attempt with a larger Site257 training pool
while keeping evaluation anchored to the audited weather assets. This follows
`liting__mvp_2__calibrated_weather_head` and
`liting__mvp_3__balanced_weather_head`.

This attempt does **not** fine-tune PANNs. PANNs remains a frozen AudioSet
weather feature extractor. The trainable part is still small:

- rain intensity: weighted MLP, `none`, `light`, `moderate`, `heavy`
- wind intensity: weighted linear head, `none`, `light`, `moderate`, `strong`

Thunder remains suppressed unless Site257 thunder evidence becomes available.

## Why This Attempt Exists

MVP-2 passed the weather gate on 101 audited Site257 weather clips. MVP-3
improved rain and joint accuracy but dropped wind accuracy below the gate. The
main issue is now data coverage, not Server B runtime or backbone capacity.

MVP-4 therefore expands training with pseudo-labelled Site257 audio discovered
from materialised site resources on Server B. These extra clips are used for
training only. The validation split remains drawn from the audited 101 weather
assets, so the result can be compared directly against MVP-2 and MVP-3.

This matches the Layer E-B goal: uploaded Site257 audio should be analysed for
the audible weather layer, not only recognised inside Murphy's curated weather
asset pool.

## Server B Runtime Target

Expected runtime:

- extra Site257 scan + PANNs/DSP feature extraction: target under 5 minutes
- actual head fitting: target under 5 minutes
- evaluation/report writing: a few minutes

The Server B job should report these timings separately so reviewers can see
where time is spent.

## Dataset Plan

- Audited validation anchor: 101 Site257 weather clips from the Layer B weather
  asset index.
- Expanded training candidates: extra Site257 audio files found under Server B
  resource roots.
- Pseudo-label source: frozen PANNs weather scores plus DSP weather features.
- Pseudo-label split role: training only.
- Sound-library assets: excluded.

## Pass Bar

- Training completes on Server B with actual head fitting under 5 minutes.
- Validation remains audited-only.
- Wind validation accuracy should recover to at least 0.70.
- Joint validation accuracy should beat MVP3 if possible.
- Report includes pseudo-label counts and selected extra roots.

## Outputs

```text
model/candidates/liting/mvp_4__data_expanded_weather_head/weather_head.pt
model/candidates/liting/mvp_4__data_expanded_weather_head/metrics.json
debug/e_b_weather_mvp4/
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
