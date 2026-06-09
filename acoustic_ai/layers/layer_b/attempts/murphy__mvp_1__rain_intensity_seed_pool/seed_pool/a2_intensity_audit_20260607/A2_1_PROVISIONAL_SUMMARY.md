# A2.1 Provisional Intensity Split

Status: provisional feature-based split only. Human listening is still the final decision.

Input accepted seeds: 28
Light provisional count: 14
Heavy provisional count: 14

Scoring rule:

`z(low_mid_ratio_db) - 0.35*z(spectral_centroid_hz) - 0.25*z(envelope_variance) - 0.20*z(crest_factor)`

Higher score biases heavy; lower score biases light.

Generated files:
- `a2_intensity_features_provisional.csv`
- `listening_provisional/light`
- `listening_provisional/heavy`
