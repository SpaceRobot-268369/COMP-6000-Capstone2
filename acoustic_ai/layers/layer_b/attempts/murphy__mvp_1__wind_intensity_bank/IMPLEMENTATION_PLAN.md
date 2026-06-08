# Layer B Wind Intensity Bank Implementation Plan

Date: 2026-06-04
Attempt: `murphy__mvp_1__wind_intensity_bank`
Status: sealed production-candidate runtime profile

## Scope

This attempt exposes Layer B wind generation with explicit intensity control:

- `light`: derived from the medium adapter with the sealed `light_a` postprocess profile.
- `medium`: learned adapter copied from `murphy__smoke_2__audioldm2_wind`.
- `heavy`: learned adapter trained for heavy wind.

The generation contract is server-locked. Runtime callers may pass only `seed`,
`wind_intensity`/`intensity`, and optional `weather_type=wind`; prompts,
guidance, diffusion steps, adapter routing, and postprocess settings come from
`params.yaml` and `acoustic_ai/registry.yaml`.

## Implementation Summary

Completed work:

1. Created the `murphy__mvp_1__wind_intensity_bank` attempt scaffold.
2. Added an intensity-bank handler that routes `light`, `medium`, and `heavy`.
3. Added a per-intensity manifest builder in `code/build_wind_manifest.py`.
4. Added attempt parameters, README, and Server B training commands.
5. Registered the attempt in `acoustic_ai/registry.yaml`.
6. Added API pass-through for `wind_intensity`.
7. Created the checkpoint-bank layout under
   `model/candidates/murphy/mvp_1__wind_intensity_bank/adapters/`.

## Final Runtime Profiles

Shared inference defaults:

- `num_inference_steps`: `200`
- `audio_length_in_s`: `8.0`
- `highpass_hz`: profile-specific where listed below
- `fade_ms`: `80`
- Base negative prompt:
  `hiss, static noise, background hum, tape noise, insects, low quality, distortion`

| Intensity | Source | Adapter | guidance | target RMS | highpass | lowpass | denoise |
|---|---|---|---:|---:|---:|---:|---|
| `light` | v2 `light_a` | derived from `medium` | `2.5` | `0.045` | `80` | `6000` | `0.22 / 0.30` |
| `medium` | v3 sealed | `medium` | `3.0` | `0.048` | `80` | none | `0.19 / 0.38` |
| `heavy` | v2 frozen | `heavy` | `3.4` | `0.09` | `80` | none | `0.15 / 0.40` |

Locked prompts:

- `light`: gentle light breeze through dry eucalyptus woodland, soft faint airflow, Bowra, Australia
- `medium`: steady wind through dry eucalyptus woodland, gentle natural breeze, Bowra, Australia
- `heavy`: loud roaring blustery wind through dry eucalyptus woodland, intense strong gusts, Bowra, Australia

## Decision History

The intensity bank went through three evaluation rounds:

- v1: initial `light`/`medium`/`heavy` separation check.
- v2: A/B comparison for `light_a` and `light_b`, with stronger medium/heavy spacing.
- v3: closure round for medium/heavy cleanup plus a `light_c` trial.

Final decision:

- `light`: use v2 `light_a`, selected by user over `light_b` and `light_c`.
- `medium`: use v3 denoise/RMS tuning.
- `heavy`: keep the v2 profile frozen because it had the best seed hit rate.

Historical v1/v2 artifacts were removed from the PR to keep the review surface
small. The retained showcase artifact is:

`showcase_intensity_eval_v3/listen_intensity_compare_v3.html`

## Training And Checkpoints

Checkpoint bank:

```text
model/candidates/murphy/mvp_1__wind_intensity_bank/
├── adapters/
│   ├── medium/
│   └── heavy/
├── README.md
└── params.yaml
```

`medium` is copied from the smoke_2 adapter. `heavy` is trained from the heavy
wind subset; commands are kept in `TRAINING_COMMAND_SERVER_B.md`.

Adapter weights remain DVC-managed and should not be committed directly to git.

## Final Showcase Policy

For PR review, only the final intensity evaluation showcase is retained:

- keep: `showcase_intensity_eval_v3/`
- removed: `showcase_intensity_eval/`, `showcase_intensity_eval_v2/`

`expected/` is unchanged.

## Known Limits

- `light` is parameter-derived, not an independently trained LoRA.
- AudioLDM2 output is 16 kHz, so high-frequency wind texture is limited by the base model.
- Seed quality still varies; the runtime remains deterministic for a fixed seed and profile.

## Follow-up

- Collect enough clean light-wind data to train a true `light` adapter.
- DVC-track medium/heavy adapter weights in the bank layout.
- Run Layer D integration checks using the sealed intensity profiles.
