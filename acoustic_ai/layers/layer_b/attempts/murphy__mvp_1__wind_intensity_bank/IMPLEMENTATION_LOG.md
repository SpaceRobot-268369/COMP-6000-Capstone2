# Implementation Log — wind_intensity_bank

Date: 2026-06-04

## Completed in this step

1. Created new attempt scaffold:
   - `acoustic_ai/layers/layer_b/attempts/murphy__mvp_1__wind_intensity_bank/`
2. Added new intensity-bank handler:
   - routes by `wind_intensity`/`intensity`
   - `medium` + `heavy` use named adapters
   - `light` derives from `medium` with reduced guidance/RMS + low-pass
3. Added per-intensity manifest builder:
   - `code/build_wind_manifest.py`
4. Added attempt params and runbook docs:
   - `params.yaml`
   - `TRAINING_COMMAND_SERVER_B.md`
   - `README.md`
5. Registered new attempt in `acoustic_ai/registry.yaml`.
6. Extended API pass-through for `wind_intensity`:
   - `acoustic_ai/server/server.py`
   - `backend/src/index.js`
   - `frontend/src/lib/api.js`
   - `frontend/src/pages/LayerATestPage.jsx` (dev test control)
7. Created bank checkpoint scaffold:
   - `model/candidates/murphy/mvp_1__wind_intensity_bank/`
   - copied `adapter_config.json` into `adapters/medium/`

## Sealed profiles (2026-06-04)

- **light**: v2 **light_a** (user sign-off; reverted from light_c)
- **medium**: v3 (denoise 0.19/0.38, rms 0.048)
- **heavy**: v2 frozen (unchanged)
- Docs: `INTENSITY_SEALED.md`; runtime: `params.yaml` + `registry.yaml` (`intensity_profile_version: sealed`)
- Eval archives: `showcase_intensity_eval_v2/` (light_a ref), `showcase_intensity_eval_v3/` (medium/heavy ref)

## Pending

- Optional: formal `showcase/` golden seeds with sealed params.
- DVC-track medium/heavy `adapter_model.safetensors` in bank layout.
