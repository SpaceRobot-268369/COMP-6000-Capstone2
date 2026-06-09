# mvp_1__wind_intensity_bank

Layer B wind intensity bank checkpoint layout.

## Adapter layout

- `adapters/medium/` — copied from `smoke_2__audioldm2_wind`
- `adapters/heavy/` — trained by `murphy__mvp_1__wind_intensity_bank`

## Notes

- `light` currently has no standalone adapter due to data scarcity and is
  derived from `medium` at runtime (see attempt README / plan docs).
- Weight binaries (`adapter_model.safetensors`) must be DVC-tracked.
