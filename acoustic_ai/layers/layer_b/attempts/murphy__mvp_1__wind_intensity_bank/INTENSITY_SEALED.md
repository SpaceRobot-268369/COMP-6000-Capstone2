# Layer B Wind Intensity — Sealed Profiles

Date: 2026-06-04  
Attempt: `murphy__mvp_1__wind_intensity_bank`  
Status: **sealed** (human sign-off after v3 eval audition)

## Decision

| Intensity | Sealed source | Notes |
|-----------|---------------|-------|
| **light** | **v2 light_a** | User preferred light_a over light_c / light_b |
| **medium** | **v3** | Stronger denoise + rms 0.048 |
| **heavy** | **v2** | Unchanged from v2 eval (already frozen) |

## Runtime parameters (authoritative)

### `heavy` (v2 frozen)

- adapter: `heavy`
- prompt: loud roaring blustery wind … intense strong gusts, Bowra, Australia
- guidance_scale: `3.4`
- output_target_rms: `0.09`
- highpass_hz: `80`
- denoise: `0.15 / 0.40`

### `medium` (v3 sealed)

- adapter: `medium`
- prompt: steady wind … gentle natural breeze, Bowra, Australia
- guidance_scale: `3.0`
- output_target_rms: `0.048`
- highpass_hz: `80`
- denoise: `0.19 / 0.38`

### `light` (v2 light_a)

- derived_from: `medium`
- prompt: gentle light breeze … soft faint airflow, Bowra, Australia
- guidance_scale: `2.5`
- output_target_rms: `0.045`
- highpass_hz: `80`
- lowpass_hz: `6000`
- denoise: `0.22 / 0.30`
- negative_prompt: shared base (no anti-strong-wind extension)

## Audit folders (listen-only, not runtime)

- v2 A/B: `showcase_intensity_eval_v2/` (light_a, light_b, medium, heavy)
- v3 eval: `showcase_intensity_eval_v3/` (light_c trial — superseded for light)
- Sealed eval reference: use **v2 `light_a`** + **v3 `medium`/`heavy`** columns from respective folders, or regenerate with current `params.yaml` / registry.

## Wired in

- `params.yaml` → `intensity_profile_version: sealed`
- `acoustic_ai/registry.yaml` → same profiles for FastAPI / dev UI
- **完整留存**：`GENERATE_WIND_FINAL.md`（generate-wind 最终封板单页文档）
