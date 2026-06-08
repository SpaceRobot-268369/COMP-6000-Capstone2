# mvp_1__weather_stem_selector

## Summary

- Layer: B
- Kind: retrieval asset bank
- Owner: murphy
- Status: MVP candidate
- Model training: none

## Bank

This artifact is a site-only weather stem retrieval bank for Bowra dry woodland
site 257. The committed `index.json` describes 113 assets and
`media_asset_bank.dvc` is a folder-level DVC pointer for the WAV bank.

## Runtime Contract

The registry exposes this artifact as `asset_bank` for
`murphy__mvp_1__weather_stem_selector`. The handler selects a deterministic
asset and start offset from `retrieval_seed`, `weather_type`, `intensity`, and
`duration_s`.

Frontend/API weather types are limited to `rain`, `wind`, and `rain+wind`.
Thunder assets remain backup-only and are not exposed as controls.

## Validation

Required checks before review:

- `dvc pull model/candidates/murphy/mvp_1__weather_stem_selector/media_asset_bank.dvc`
- confirm all 113 `index.json` `audio_path` entries resolve on disk
- smoke `registry.generate(...)` for `rain`, `wind`, and `rain+wind`
