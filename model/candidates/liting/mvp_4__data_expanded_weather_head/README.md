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

The checkpoint is DVC-tracked after Server B training. Metrics are git-tracked.
