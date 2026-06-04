# E-B MVP4 Demo Notes

Owner: `liting`

## What This Demonstrates

This attempt demonstrates the E-B weather analysis head for Analysis Mode. Given
an uploaded Site257 audio clip, the head estimates the audible weather layer:

```json
{
  "wind_intensity": "none | light | moderate | strong",
  "rain_intensity": "none | light | moderate | heavy",
  "thunder_intensity": "none",
  "confidence": 0.0
}
```

Thunder is intentionally suppressed because current Site257 evidence is not
strong enough to claim thunder detection.

## Model Used

- Frozen base model: PANNs CNN14 AudioSet tagger.
- Extra evidence: DSP weather features such as RMS, band energy ratios,
  spectral flatness, onset strength, and modulation.
- Trainable part: two small calibrated heads:
  - rain-intensity head: weighted MLP
  - wind-intensity head: weighted linear classifier

The PANNs backbone is not fine-tuned. The Server B training step only fits the
small weather heads.

## Current Server B Result

MVP4 should be compared directly against the audited validation anchor used by
MVP2/MVP3:

- MVP2 baseline: rain 0.769, wind 0.731, joint 0.615.
- MVP3 result: rain 0.885, wind 0.692, joint 0.654.
- MVP4 target: use more Site257 train-only examples to recover wind accuracy
  while keeping the higher joint score.
- Server B result: see `metrics.json` after the run.

Report paths:

```text
model/candidates/liting/mvp_4__data_expanded_weather_head/metrics.json
debug/e_b_weather_mvp4/report.json
debug/e_b_weather_mvp4/validation_predictions.csv
```

## How To Explain It

The system is not generating weather audio here. It is analysing an uploaded
clip and estimating the weather layer that is audible in the recording.

The detector first uses PANNs CNN14 as a frozen pretrained audio model to obtain
weather-related evidence such as rain, wind, and thunder scores. It also
computes transparent DSP features from the waveform. A small weather head is
then trained on audited Site257 weather clips plus pseudo-labelled Site257
train-only clips so the output matches our project labels: rain intensity and
wind intensity. The extra wind candidates are score-stratified so the model sees
both moderate and strong pseudo-wind examples during training.

## Current Limitations

- Pseudo-labelled training rows can be noisy.
- Mixed rain+wind cases remain harder than single-component cases.
- Thunder is not included until validated Site257 thunder examples exist.
- Confidence is model probability, not a guarantee of ecological truth.

## Next Iteration

If MVP4 improves the audited validation gate, the next step is to promote it as
the E-B integration candidate. If not, the next attempt should improve the
pseudo-label selection thresholds or move to a small manually audited expansion
set.
