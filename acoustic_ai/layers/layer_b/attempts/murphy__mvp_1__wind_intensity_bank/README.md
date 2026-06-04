# murphy__mvp_1__wind_intensity_bank

Layer B wind generate-route MVP attempt with explicit intensity control:

- `medium`: learned adapter (reuses smoke_2 checkpoint)
- `heavy`: learned adapter (newly trained on heavy wind subset)
- `light`: derived **light_a** profile (v2 eval winner; lowpass 6 kHz)

## Goal

Expose stable `light / medium / heavy` wind generation while keeping the
generation contract server-locked and seed-deterministic.

## Runtime contract

- Inputs: `seed`, `intensity` (`light|medium|heavy`), optional `weather_type=wind`
- Prompt/steps/guidance/postprocess are owned by registry params.
- Invalid or missing intensity falls back to `default_intensity=medium`.

## Artifacts

- Attempt code: this folder
- Checkpoint bank: `model/candidates/murphy/mvp_1__wind_intensity_bank/adapters/`
  - `medium/` (copied from smoke_2)
  - `heavy/` (trained in this attempt)

## Status

**Sealed (2026-06-04):** light = v2 light_a · medium = v3 · heavy = v2 frozen.  
See `INTENSITY_SEALED.md` and **`GENERATE_WIND_FINAL.md`**（generate-wind 完整封板留存）。

## Notes

- `light` is parametric-derived (medium adapter + light_a postprocess).
- Upgrade path: collect enough clean light clips and replace with a true
  `light` adapter.
