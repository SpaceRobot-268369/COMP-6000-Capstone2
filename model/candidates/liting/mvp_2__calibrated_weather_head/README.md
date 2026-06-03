# E-B MVP 2 — Calibrated Weather Head

Owner: `liting`

## What This Checkpoint Is

Small calibrated weather head for Layer E-B analysis. It predicts audible rain
and wind intensity from frozen PANNs CNN14 evidence plus DSP features.

The PANNs backbone is **not** fine-tuned. Only two small linear heads are
trained:

- rain: `none`, `light`, `moderate`, `heavy`
- wind: `none`, `light`, `moderate`, `strong`

Thunder remains suppressed because Site257 thunder evidence is not validated.

## Training Data

- Source: Site257 weather assets from the Layer B weather asset index.
- Sound-library assets: excluded from training.
- Cases: 101 materialised Site257 clips.
- Split: 75 train / 26 validation.

## Server B Result

```text
rain_val_accuracy: 0.769
wind_val_accuracy: 0.731
joint_val_accuracy: 0.615
feature_seconds: 15.132
training_seconds: 1.359
total_seconds: 16.501
```

This confirms the trainable step is well under the 5-minute target. Current
quality is an MVP-2 first pass, not a production detector.

## Files

```text
weather_head.pt      # DVC-tracked checkpoint
metrics.json         # git-tracked validation metrics
params.yaml          # git-tracked training parameters
```

