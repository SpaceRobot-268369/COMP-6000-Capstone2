# Weather Stem Selector MVP

Layer B MVP handler for short weather-only stems.

Frontend controls:

- weather type: `rain`, `wind`, `rain+wind`
- intensity: `light`, `medium`, `heavy`
- duration in seconds
- retrieval seed

The retrieval seed controls both asset selection and start offset inside longer
assets. Layer B returns a short WAV stem plus metadata; Layer D owns final
timeline placement and full soundscape mixing.

The registry wires this attempt to the first-class retrieval artifact:

```text
model/candidates/murphy/mvp_1__weather_stem_selector
```

That artifact contains `index.json` plus a folder-level
`media_asset_bank.dvc` pointer. Thunder assets are retained as backup-only
bank members, but thunder/storm controls are not exposed in the frontend or
API for this MVP.

Validation:

- `dvc pull model/candidates/murphy/mvp_1__weather_stem_selector/media_asset_bank.dvc`
- verify all 113 `index.json` `audio_path` entries resolve on disk
- smoke `registry.generate(...)` for `rain`, `wind`, and `rain+wind`
