# Layer B rain smoke spectrum diagnostic

Goal: compare 10 generated showcase samples against the 72-file human-audited rain training pool, focusing on whether generated samples are lower than real rain in the 2-8 kHz band.

- Showcase generated samples: 10
- Real rain training samples: 72
- Generated mean 2-8 kHz: -64.52 dB
- Real rain mean 2-8 kHz: -69.62 dB
- Generated minus real in 2-8 kHz: 5.10 dB
- Generated 2-8k minus 0-2k: -7.67 dB
- Real 2-8k minus 0-2k: -8.19 dB
- Relative ratio difference: 0.52 dB

Judgement: **NO CLEAR GAP: generated samples are not obviously lower than real rain in 2-8 kHz by mean PSD.**

Figures:
- `debug/murphy_layer_b_rain_spectrum_diagnostic_20260606/figures/mean_spectrum_generated_vs_real.png`
- `debug/murphy_layer_b_rain_spectrum_diagnostic_20260606/figures/band_energy_comparison.png`
- `debug/murphy_layer_b_rain_spectrum_diagnostic_20260606/figures/mel_examples`

CSV outputs:
- `debug/murphy_layer_b_rain_spectrum_diagnostic_20260606/band_energy_metrics.csv`
- `debug/murphy_layer_b_rain_spectrum_diagnostic_20260606/per_file_2_8khz_metrics.csv`
