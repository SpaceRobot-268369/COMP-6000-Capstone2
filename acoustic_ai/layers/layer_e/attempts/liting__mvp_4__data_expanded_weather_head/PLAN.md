# Plan — E-B MVP 4 Data-Expanded Weather Head

## Goal

Expand E-B weather training beyond the original audited weather pool while
keeping evaluation fair and audited-only. Uploaded Site257 audio should return:

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
- Trainable model: rain weighted MLP + wind weighted linear classifier.
- No full PANNs fine-tuning.

## Dataset

Use Site257 audio only.

- Audited anchor: 101 manually/audit-labelled Site257 weather assets from the
  Layer B weather asset index.
- Expanded training data: extra Site257 files found from Server B materialised
  resource roots.
- Pseudo-labels: assigned from frozen PANNs weather evidence and DSP weather
  features.
- Pseudo wind candidates are score-stratified into moderate/strong so data
  expansion targets the wind boundary that MVP3 missed.
- Validation: audited rows only, never pseudo-labelled rows.
- Sound-library assets: excluded.

## Training Run

1. Materialise Site257 weather WAV assets on Server B.
2. Build feature table:
   - PANNs rain/wind/thunder component scores.
   - matched PANNs label scores.
   - DSP weather features.
3. Scan extra Site257 audio roots and select pseudo-labelled train-only cases.
4. Make deterministic train/validation split over audited rows.
5. Add pseudo-labelled rows only to train.
6. Train rain and wind heads with component-specific model settings.
7. Write checkpoint and metrics.
8. Evaluate audited validation cases and write report.

## Pass Bar

- Training completes with actual head fitting under 5 minutes.
- Report includes split counts, training time, evaluation metrics, confusion
  matrix, pseudo-label counts, extra roots, and limitations.
- PANNs backbone remains frozen.
- Thunder is not claimed from Site257 unless separately validated.
- Compare directly against MVP-2 and MVP-3 on the same audited validation
  anchor.
