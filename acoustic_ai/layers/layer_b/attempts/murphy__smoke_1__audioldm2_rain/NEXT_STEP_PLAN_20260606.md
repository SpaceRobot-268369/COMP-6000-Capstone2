# Layer B Rain Smoke-1 — Next Step Plan (2026-06-06)

## Background

Based on the local diagnostic folder:

- `spectrum_diagnostic_20260606/REPORT.md`
- `spectrum_diagnostic_20260606/band_energy_metrics.csv`
- `spectrum_diagnostic_20260606/per_file_2_8khz_metrics.csv`

the main observations are:

- No clear deficit in 2-8 kHz for generated samples vs real rain (mean PSD).
- Main gap is in 8-11 kHz, where generated audio is far lower than real rain.
- This aligns with the 16 kHz generation ceiling (Nyquist 8 kHz) of current AudioLDM2 path.

## Key Judgement

The current "muffled" perception is mainly a missing >8 kHz brightness issue,
not a 2-8 kHz weakness. Therefore:

- Extra denoise is not the main fix path.
- Post-generation bandwidth extension is the most direct low-cost intervention.
- Overfitting concern should be handled in a separate retraining track (early stop + augmentation),
  not mixed into the immediate timbre-brightness fix.

## Recommended Next Steps (No execution in this file)

### Step 1 (Priority): Offline BWE prototype on existing showcase outputs

Goal:

- Synthesize plausible >8 kHz rain texture and export to 22.05/24 kHz outputs.

Method:

- Use a noise-shaped high-band reconstruction strategy (SBR-like / exciter-style).
- Match high-band spectral slope to real-rain reference statistics from diagnostic CSV.
- Run only on existing showcase seeds (42-51) for A/B listening and spectrum checks.

Deliverables:

- New diagnostic comparison plots (before vs after BWE vs real reference).
- Per-file and aggregated band metrics update.
- Listening notes for subjective pass/fail.

### Step 2: Reduce or disable current spectral denoise in the test branch

Reason:

- Rain itself is noise-like; aggressive spectral subtraction can remove natural rain texture.

Action direction:

- Start from denoise disabled (or much weaker), then re-check hiss vs openness trade-off.

### Step 3: Retraining track for overfitting risk (separate track)

Goal:

- Improve generalization under the 8-recording pool constraint.

Direction:

- Select checkpoint by validation (not simply last epoch).
- Add small-data augmentation (gain/time shift/light stretch/mix-style variants).
- Increase regularization conservatively (e.g., LoRA dropout up one notch).

## Decision Rule

Proceed in order:

1. Validate whether BWE solves perceived muffled quality.
2. Only if needed, tune denoise and prompt/inference details.
3. Keep retraining changes isolated from postprocess experiments for clean attribution.
