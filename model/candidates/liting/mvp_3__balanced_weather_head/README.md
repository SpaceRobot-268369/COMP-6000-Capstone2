# E-B MVP3 Balanced Weather Head

Owner: `liting`

## Purpose

Candidate checkpoint for the Layer E-B weather analysis head. This attempt uses
frozen PANNs CNN14 and DSP features with a hybrid calibrated head:

- Rain: weighted MLP.
- Wind: weighted linear classifier.

## Inputs

- Site257 weather-labelled clips from the Layer B curated asset index.
- Frozen PANNs CNN14 weather evidence.
- DSP weather features from the waveform.

## Outputs

- `weather_head.pt` — DVC-tracked checkpoint.
- `metrics.json` — git-tracked Server B training/evaluation report.
- `params.yaml` — run configuration.

## Baseline

MVP2 baseline on the same split:

- Rain validation accuracy: 0.769
- Wind validation accuracy: 0.731
- Joint validation accuracy: 0.615

MVP3 is considered useful if it improves joint accuracy without making wind
accuracy worse in a meaningful way. The selected hybrid setting came from a
short Server B sweep: MLP improved rain, while linear remained more stable for
wind.
