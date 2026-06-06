# BWE phase 6 decision

Decision: **do not integrate yet**.

Reason: phase 5 improved the 8-11 kHz deficit substantially, but the current prototype overshoots the real-rain reference. Manual A/B gates have not been evaluated.

Measured result:

- Real 8-11 kHz: -81.22 dB
- Before 8-11 kHz: -105.53 dB
- After BWE 8-11 kHz: -74.96 dB
- Before gap vs real: -24.31 dB
- After gap vs real: 6.25 dB
- Absolute gap improvement: 18.06 dB
- Passes +/-5 dB band gate: False

Current exposed parameters are in `bwe_parameters_current.json`.
Recommended next trial parameters are in `bwe_parameters_next_trial.json`.

Recommended next highband trim: `-2.75` dB.

Next action: rerun phase 4 and phase 5 with the trimmed highband, then manually A/B seed_48 and seed_51.
