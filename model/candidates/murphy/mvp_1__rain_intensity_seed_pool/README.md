# mvp_1__rain_intensity_seed_pool

Layer B rain-only AudioLDM2 LoRA smoke checkpoint (murphy).

## Layout

- `adapter_config.json` — PEFT LoRA config
- `adapter_model.safetensors` — UNet LoRA weights (DVC-tracked)
- `params.yaml` — frozen training snapshot at seal time

## Training

- Base model: `cvssp/audioldm2`
- Attempt: `murphy__mvp_1__rain_intensity_seed_pool`
- Data: 72 site_257 rain clips (recording-group split)
- Epochs: 4, batch 4, lr 1e-5, LoRA r/alpha/dropout 8/32/0.1
- Trained on serverB (2026-06-06)

## Runtime

Served via attempt `murphy__mvp_1__rain_intensity_seed_pool` with curated good-seed
policy and BWE postprocess (24 kHz export). See attempt `MODEL_CARD.md` for
limitations and audit evidence.

## Audit

<!-- Evaluation notes and review findings go here. -->
