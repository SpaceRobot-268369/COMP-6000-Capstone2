# E-B MVP2 Demo Notes

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
- Trainable part: two small linear calibration heads:
  - one rain-intensity head
  - one wind-intensity head

The PANNs backbone is not fine-tuned. The Server B training step only fits the
small calibration heads.

## Current Server B Result

Latest MVP2 run:

- Site257 materialised clips: 101
- Train split: 75
- Validation split: 26
- Rain validation accuracy: 0.769
- Wind validation accuracy: 0.731
- Joint validation accuracy: 0.615
- Actual calibration-head training time: 1.36 seconds
- Total run time including feature extraction: 16.50 seconds

Report paths:

```text
model/candidates/liting/mvp_2__calibrated_weather_head/metrics.json
debug/e_b_weather_mvp2/report.json
debug/e_b_weather_mvp2/validation_predictions.csv
```

## How To Explain It

The system is not generating weather audio here. It is analysing an uploaded
clip and estimating the weather layer that is audible in the recording.

The detector first uses PANNs CNN14 as a frozen pretrained audio model to obtain
weather-related evidence such as rain, wind, and thunder scores. It also
computes transparent DSP features from the waveform. A small calibrated head is
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
