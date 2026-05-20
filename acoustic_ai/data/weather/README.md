# Layer B Weather Assets

Layer B is retrieval-based. Curated wind and rain clips should be stored under
`weather_assets/` and tracked with DVC; `asset_index.csv` stays in git as the
small metadata and license index.

Recommended layout:

```text
weather_assets/
├── wind/
│   ├── light/
│   ├── moderate/
│   └── strong/
└── rain/
    ├── light/
    ├── moderate/
    └── heavy/
```

## Source Recommendation

Preferred source for the MVP: **FSD50K / Freesound clips with per-clip license
metadata**. FSD50K includes human-labeled environmental audio, ships metadata
with uploader/license fields, and makes it possible to filter to CC0 or CC-BY
assets before adding clips to DVC.

References:

- FSD50K dataset page: https://zenodo.org/records/4060432
- FSD50K browser: https://annotator.freesound.org/fsd/release/FSD50K/
- Freesound datasets: https://labs.freesound.org/datasets/

Use only assets with a recorded source URL, license, and attribution. Avoid
assets whose license is unclear or incompatible with the intended project use.

BBC Sound Effects can be useful for research-only experiments, but its standard
license is limited to personal, educational, or research use, so it should not
be the default asset source for a reusable project library.

## FSD50K Candidate Workflow

Download the FSD50K metadata and audio outside git, then build a candidate list:

```bash
python3 script/dataset/build_layer_b_fsd50k_candidates.py \
  --fsd50k-root /path/to/FSD50K \
  --output acoustic_ai/data/weather/fsd50k_weather_candidates.csv
```

The candidate CSV is not the final asset index. Listen to candidates, reject
clips with foreground speech/music/animals, assign the final intensity bucket,
copy approved WAV files into `weather_assets/`, then add rows to
`asset_index.csv`.

## Smoke Asset Set

Current smoke set: 12 WAV files, two clips per bucket:

- `wind/light`
- `wind/moderate`
- `wind/strong`
- `rain/light`
- `rain/moderate`
- `rain/heavy`

The smoke set uses Freesound public preview clips where direct access was
available, with Wikimedia Commons clips used to complete the wind and heavy-rain
buckets. Each file was converted to 22,050 Hz mono WAV and trimmed to at most
30 seconds.

Readable S3 mirror:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/training_dataset/layer-b/weather-assets-smoke/
```

This readable mirror is not a substitute for DVC. The local `weather_assets/`
directory still needs to be DVC-tracked before merge.

## Index Rules

Each indexed asset must have:

- `asset_id`: stable local identifier.
- `clip_path`: path relative to `weather_assets/`.
- `layer`: `wind` or `rain`.
- `intensity`: `light`, `moderate`, `strong` for wind; `light`, `moderate`,
  `heavy` for rain.
- `source_url`, `license`, `attribution`: enough information for audit.

After adding real clips:

```bash
dvc add acoustic_ai/data/weather/weather_assets
git add acoustic_ai/data/weather/asset_index.csv acoustic_ai/data/weather/weather_assets.dvc
dvc push
```
