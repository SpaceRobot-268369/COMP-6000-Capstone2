# Layer B Wind Intensity v3 Closure Plan

Date: 2026-06-04
Attempt: `murphy__mvp_1__wind_intensity_bank`
Status: **approved** — execute eval batch, then human audition before production lock.

## v2 listening summary (input to v3)

| Profile | Outcome |
|---------|---------|
| **light_a** | Wind discernible but often muffled / dull mic quality |
| **light_b** | Brighter, sometimes near medium; more noise; light wind can disappear |
| **medium v2** | Layering OK; some seeds still noisy |
| **heavy v2** | High good-seed rate — **freeze, no further tuning** |

Decision: do **not** pick light_a or light_b wholesale. Ship **light_c** (hybrid) + **medium v3** (denoise bump) + **heavy v2 locked**.

## v3 runtime profiles (authoritative)

Shared inference: `num_inference_steps=200`, `audio_length_in_s=8.0`, `highpass` where noted, `fade_ms=80`.

Base negative (medium/heavy):

```text
hiss, static noise, background hum, tape noise, insects, low quality, distortion
```

### `heavy` — frozen from v2 eval

| Field | Value |
|-------|-------|
| adapter | `heavy` |
| prompt | loud roaring blustery wind through dry eucalyptus woodland, intense strong gusts, Bowra, Australia |
| guidance_scale | `3.4` |
| output_target_rms | `0.09` |
| highpass_hz | `80` |
| denoise | `0.15 / 0.40` |

### `medium` — v2 + noise cleanup

| Field | v2 | v3 |
|-------|----|----|
| adapter | medium | medium |
| prompt | unchanged steady-wind prompt | unchanged |
| guidance_scale | 3.0 | **3.0** |
| output_target_rms | 0.05 | **0.048** |
| highpass_hz | 80 | 80 |
| denoise strength / floor | 0.15 / 0.40 | **0.19 / 0.38** |

### `light` — light_c hybrid (derived from medium)

| Field | light_a | light_b | **light_c (v3)** |
|-------|---------|---------|------------------|
| derived_from | medium | medium | medium |
| guidance_scale | 2.5 | 2.5 | **2.3** |
| output_target_rms | 0.045 | 0.045 | **0.041** |
| highpass_hz | 80 | 120 | **100** |
| lowpass_hz | 6000 | off | **off** |
| denoise | 0.22 / 0.30 | 0.22 / 0.30 | **0.20 / 0.34** |
| negative_prompt | base | base | base **+** strong wind / gust rejection |

Rationale: B-like brightness without 120 Hz highpass; A-like denoise without 6 kHz lowpass; lower guidance/RMS to separate from medium.

## Eval batch

- Seeds: `42..51` (10)
- Intensities: `light`, `medium`, `heavy` (3 columns)
- Output: `showcase_intensity_eval_v3/{light,medium,heavy}/seed_<N>_generated/`
- Listen page: `showcase_intensity_eval_v3/listen_intensity_compare_v3.html`

## Pass criteria (human audition)

1. **light**: seeds 42–44 — wind audible, less muffled than light_a, less noise / less “medium-like” than light_b.
2. **medium**: previously noisy seeds improved; **seed 48** not worse than v2 medium.
3. **heavy**: **seed 48** and spot checks match v2 heavy (regression guard).
4. **Tiering**: light < medium < heavy on loudness and wind energy.

## Rollout after v3 pass

1. `params.yaml` + `acoustic_ai/registry.yaml` already carry v3 (this commit).
2. Optional: copy golden seeds into formal `showcase/` once signed off.
3. DVC-track adapter weights separately (not in git).

## Audit trail

- v1: `showcase_intensity_eval/`
- v2 A/B: `showcase_intensity_eval_v2/` (light_a, light_b retained)
- v3: `showcase_intensity_eval_v3/`
