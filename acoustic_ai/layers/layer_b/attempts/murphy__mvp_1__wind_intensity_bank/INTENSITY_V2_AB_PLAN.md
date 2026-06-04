# Layer B Wind Intensity v2 A/B Plan

Date: 2026-06-04
Attempt: `murphy__mvp_1__wind_intensity_bank`

## Goal

Address the latest listening feedback:

- `light` is often too weak/noisy or dull.
- `medium` vs `heavy` separation is present but not always obvious.
- Keep known-good seeds (especially around medium/heavy seed 48) from regressing.

## Strategy

Run a controlled A/B where only `light` spectral shaping differs:

- `light_a`: keep low-pass but relax it.
- `light_b`: remove low-pass and instead increase high-pass.

Use one shared v2 profile for `medium`/`heavy` to strengthen intensity spacing.

## v2 Profiles

### Shared changes (all runs)

- seed set for evaluation: `42..51` (10 seeds)
- fixed base/model/post chain (except profile-specific fields below)
- denoise pipeline retained

### `medium` (v2 baseline)

- adapter: `medium`
- prompt: unchanged
- guidance: `3.0`
- output_target_rms: `0.05` (from 0.06)
- highpass_hz: `80`
- denoise: `0.15 / 0.40`

### `heavy` (v2 stronger separation)

- adapter: `heavy`
- prompt: strengthened wording (louder/stronger gust descriptors)
- guidance: `3.4` (from 3.2)
- output_target_rms: `0.09` (from 0.075)
- highpass_hz: `80`
- denoise: `0.15 / 0.40`

### `light_a` (derived from medium)

- derived_from: `medium`
- prompt: light breeze prompt
- guidance: `2.5`
- output_target_rms: `0.045` (from 0.03)
- highpass_hz: `80`
- lowpass_hz: `6000`
- denoise: `0.22 / 0.30`

### `light_b` (derived from medium)

- derived_from: `medium`
- prompt: light breeze prompt
- guidance: `2.5`
- output_target_rms: `0.045`
- highpass_hz: `120`
- lowpass_hz: disabled
- denoise: `0.22 / 0.30`

## Generation Matrix

For each seed `42..51`, generate:

- `light_a`
- `light_b`
- `medium`
- `heavy`

Total: `10 x 4 = 40` samples.

## Output Layout

`showcase_intensity_eval_v2/`

- `light_a/seed_<N>_generated/`
- `light_b/seed_<N>_generated/`
- `medium/seed_<N>_generated/`
- `heavy/seed_<N>_generated/`
- `listen_intensity_compare_v2.html` (4-column same-seed comparison)

## Decision Rule (keep A or B)

Pick one `light` profile globally using:

1. Problem-seed priority: `42/43/44`
2. No regression at known-good references (especially medium/heavy around seed 48)
3. Better worst-case behavior:
   - audible light wind
   - less intrusive noise
   - no obvious dull/muffled artifact

## Rollout after selection

1. Keep winner (`light_a` or `light_b`) in registry + attempt params.
2. Remove losing profile from active runtime config.
3. Keep both sets in audit folder for traceability.
