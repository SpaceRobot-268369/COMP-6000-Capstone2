# Layer B AudioLDM2 Wind Smoke 2 Implementation Plan

Date: 2026-06-04
Attempt: `murphy__smoke_2__audioldm2_wind`
Status: closed smoke-stage MVP showcase selection

## Scope

This attempt is the second wind-only AudioLDM2 LoRA smoke test for Layer B. It
addresses smoke_1 electronic artifacts by using the Layer A validated training
recipe and stricter wind data filtering.

Runtime scope:

- `weather_type=wind`
- single medium-wind profile
- generated wind stem, no retrieval at runtime
- seed-deterministic output

## Training Setup

Manifest filtering:

- `contamination <= 0.28`
- `wind_intensity=medium`
- exclude `nov2019_storm_scout001`
- maximum 3 clips per `source_recording_id`

The resulting manifest contains 35 medium-wind samples from 24 recordings.
Server B training commands are kept in `TRAINING_COMMAND_SERVER_B.md`.

## Iteration Summary

The postprocess tuning was run as a sequence of S3a rounds:

1. **S3a baseline**
   - Increased target RMS from `0.003` to `0.06`.
   - Raised high-pass from `60` to `80`.
   - Added negative prompt support for hiss/static/noise rejection.
2. **S3a.2**
   - Added spectral denoise after high-pass and before RMS matching.
   - Initial denoise was too aggressive for some seeds.
3. **S3a.3**
   - Reworked denoise reconstruction to avoid boundary clicks.
   - Scanned lighter denoise strengths while protecting wind body.
4. **S3a.4 final**
   - Compared Variant A and Variant B.
   - User selected Variant A.
   - Ran a 40-seed scan from seed `42` through `81`.

## Final Locked Profile

The final profile is S3a.4 Variant A:

| Parameter | Value |
|---|---|
| `guidance_scale` | `3.0` |
| `num_inference_steps` | `200` |
| `audio_length_in_s` | `8.0` |
| `output_target_rms` | `0.06` |
| `highpass_hz` | `80` |
| `fade_ms` | `80` |
| `denoise_enabled` | `true` |
| `denoise_strength` | `0.15` |
| `denoise_floor_ratio` | `0.40` |
| `denoise_noise_quantile` | `0.2` |
| `denoise_hop_length` | `512` |

Prompt:

```text
steady wind through dry eucalyptus woodland, gentle natural breeze, Bowra, Australia
```

Negative prompt:

```text
hiss, static noise, background hum, tape noise, insects, low quality, distortion
```

## Final Showcase

The retained final showcase is:

`showcase_s3a4_final/listen_generated.html`

It contains the final 40-seed scan used for review and audit. User-approved good
seeds are:

- `48`
- `50`
- `52`
- `55`
- `59`
- `72`

The previous `showcase/` directory was removed during PR cleanup because it was
a duplicate final listening entry pointing into `showcase_s3a4_final`.

## Relationship To Wind Intensity Bank

This attempt provides the medium-wind adapter source for:

`../murphy__mvp_1__wind_intensity_bank/`

The follow-up bank attempt adds explicit `light`, `medium`, and `heavy`
intensity routing. `light` is parameter-derived from this medium adapter, while
`heavy` uses a separately trained adapter.

## Known Limits

- This attempt covers medium wind only.
- Some seeds remain noisy or weak; the selected good seeds are used for MVP review.
- Further light/heavy behavior belongs to the MVP intensity bank attempt.
