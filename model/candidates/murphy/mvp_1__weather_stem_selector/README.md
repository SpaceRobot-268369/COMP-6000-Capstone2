# mvp_1__weather_stem_selector

## Summary

- Owner: murphy
- Layer / role: Layer B weather stem selector retrieval bank
- Status: candidate
- Base model: none (site-derived retrieval bank)

## Purpose / hypothesis

Provide a DVC-tracked site-only weather stem bank for the Layer B MVP retrieval handler. The handler selects deterministic short rain, wind, rain+wind, and backup thunder stems by weather type, intensity, duration, and retrieval seed.

## Dataset / inputs

- Source site: site_257 Bowra dry woodland
- Assets: 113 site-derived WAV stems
- Index: index.json
- Audio bank: media_asset_bank/

## Results analysis / audit

## Materialization

The audio bank is tracked by a folder-level DVC pointer:

```bash
dvc pull model/candidates/murphy/mvp_1__weather_stem_selector/media_asset_bank.dvc
```

After materialization, `index.json` should contain 113 assets and all
`audio_path` entries should resolve relative to this folder.

## Runtime Defaults

- exposed weather types: `rain`, `wind`, `rain+wind`
- backup-only weather assets: `thunder`, `storm`
- intensities: `light`, `medium`, `heavy`
- default retrieval seed: `42`
- default duration: `10s`

## Results Analysis / Audit

Smoke validation should run the real registry path for `rain`, `wind`, and
`rain+wind`, then record the selected `asset_id` and retrieval params in the
attempt showcase metadata.
