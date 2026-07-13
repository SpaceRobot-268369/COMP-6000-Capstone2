# Plan — E-B MVP 2 Calibrated Weather Head

## Goal

Train a small calibration head for E-B weather analysis so uploaded Site257
audio can return:

```json
{
  "wind_intensity": "none | light | moderate | strong",
  "rain_intensity": "none | light | moderate | heavy",
  "thunder_intensity": "none",
  "confidence": 0.0
}
```

## Model

- Frozen feature extractor: PANNs CNN14 AudioSet tagger.
- Supporting features: spectral/DSP features from the E-B smoke detector.
- Trainable model: two small linear classifiers.
- No full PANNs fine-tuning.

## Dataset

Use Site257 weather assets only. Sound-library assets are excluded from
client-facing calibration/training. Mixed rain+wind assets can be used as
boundary/evaluation cases, but the report must separate them from primary
single-component cases.

## Training Run

1. Materialise Site257 weather WAV assets on Server B.
2. Build feature table:
   - PANNs rain/wind/thunder component scores.
   - matched PANNs label scores.
   - DSP weather features.
3. Make deterministic train/validation split.
4. Train rain and wind heads.
5. Write checkpoint and metrics.
6. Evaluate validation cases and write report.

## Pass Bar

- Training completes with actual head fitting under 5 minutes.
- Report includes split counts, training time, evaluation metrics, confusion
  matrix, and limitations.
- PANNs backbone remains frozen.
- Thunder is not claimed from Site257 unless separately validated.

