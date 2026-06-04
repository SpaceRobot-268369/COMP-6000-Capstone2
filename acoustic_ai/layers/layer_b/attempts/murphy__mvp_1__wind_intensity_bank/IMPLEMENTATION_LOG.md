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

## v3 closure (2026-06-04)

- Human v2 A/B: heavy frozen; medium denoise bump; light_c hybrid (not light_a/b).
- Docs: `INTENSITY_V3_CLOSURE_PLAN.md`
- Runtime: `params.yaml` + `acoustic_ai/registry.yaml` updated to v3 profiles.
- Eval: `dev-artifacts-self-testing/run_intensity_v3_eval.py` → `showcase_intensity_eval_v3/`

## Pending

- Server B v3 batch + local scp + listen page.
- Human sign-off on v3 before formal `showcase/` promotion.
- DVC-track medium/heavy `adapter_model.safetensors` in bank layout.
