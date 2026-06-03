# E-B MVP3 Demo Notes

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
small MLP heads.

## Current Server B Result

MVP3 is compared directly against the MVP2 deterministic split:

- MVP2 baseline: rain 0.769, wind 0.731, joint 0.615.
- MVP3 target: improve joint rain+wind accuracy without fine-tuning PANNs.
- Quick sweep result: rain improved with MLP, wind stayed better with linear,
  so the final MVP3 uses a hybrid head.
- Server B result: see `metrics.json` after the run.

Report paths:

```text
model/candidates/liting/mvp_3__balanced_weather_head/metrics.json
debug/e_b_weather_mvp3/report.json
debug/e_b_weather_mvp3/validation_predictions.csv
```

## How To Explain It

The system is not generating weather audio here. It is analysing an uploaded
clip and estimating the weather layer that is audible in the recording.

The detector first uses PANNs CNN14 as a frozen pretrained audio model to obtain
weather-related evidence such as rain, wind, and thunder scores. It also
computes transparent DSP features from the waveform. A balanced MLP head is
then trained on Site257-labelled weather clips so the output matches our
project labels: rain intensity and wind intensity.

## Current Limitations

- The current dataset is still small for a robust weather detector.
- Mixed rain+wind cases are harder than single-component cases.
- Thunder is not included until validated Site257 thunder examples exist.
- Confidence is model probability, not a guarantee of ecological truth.

## Next Iteration

The next useful iteration is not a larger neural model yet. The priority is to
expand the Site257 labelled weather set, especially balanced buckets for:

- no weather
- light/moderate/heavy rain
- light/moderate/strong wind
- mixed rain+wind boundary cases

After that, rerun the same Server B training script and compare the confusion
matrix and per-policy breakdown.
