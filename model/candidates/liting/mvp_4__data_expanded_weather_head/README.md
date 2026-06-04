# E-B MVP 4 — Data-Expanded Weather Head Candidate

Owner: `liting`

## Role

Layer E-B weather analysis checkpoint candidate. Given an uploaded Site257
audio clip, the handler returns:

```json
{
  "wind_intensity": "none | light | moderate | strong",
  "rain_intensity": "none | light | moderate | heavy",
  "thunder_intensity": "none",
  "confidence": 0.0
}
```

Thunder is explicitly suppressed until validated Site257 thunder examples are
available.

## Model

- Frozen base: PANNs CNN14 AudioSet tagger.
- Supporting features: transparent DSP/weather features.
- Trainable part: small rain and wind heads over frozen features.
- No PANNs backbone fine-tuning.

## Dataset Plan

Training uses audited Site257 Layer B weather assets plus pseudo-labelled
Site257 audio discovered from Server B resource roots. Validation remains
audited-only so MVP4 can be compared against MVP2 and MVP3.

## Expected Artifacts

```text
weather_head.pt
metrics.json
```

Metrics are git-tracked. Because this run did not pass the E-B gate, the
checkpoint is not promoted as the integration candidate.

## Server B Result

Latest run used 101 audited Site257 rows plus 30 pseudo-labelled train-only
rows:

- Rain validation accuracy: 0.846
- Wind validation accuracy: 0.615
- Joint validation accuracy: 0.615
- Gate: `needs_iteration`

This candidate is kept as an iteration checkpoint. It shows that naive
pseudo-labelled data expansion can hurt the wind boundary, so the next attempt
should use stricter pseudo-label selection or a manually audited expansion set.
