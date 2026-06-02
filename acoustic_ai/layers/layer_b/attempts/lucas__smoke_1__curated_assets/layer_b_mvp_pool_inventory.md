# Layer B MVP Pool Inventory

Last updated: 2026-06-02

## Runtime Pool

The frontend-facing Layer B MVP reads this index:

`acoustic_ai/layers/layer_b/attempts/lucas__smoke_1__curated_assets/data/weather/asset_index.csv`

Current runtime index contents:

| Source | Rows | Runtime status |
| --- | ---: | --- |
| sound_library | 40 | Active index |
| site | 0 | Not merged into runtime index yet |

Runtime-usable and locally materialized rows after path cleanup:

| Primary weather | Count |
| --- | ---: |
| wind | 5 |
| rain | 10 |
| rain+wind | 1 |
| storm | 10 |
| thunder | 3 |

The old seed rows previously pointed at `acoustic_ai/data/weather/...`. They
now point at the current attempt-local data root:

`acoustic_ai/layers/layer_b/attempts/lucas__smoke_1__curated_assets/data/weather/...`

## DVC / S3 Backing

The curated sound-library folders are tracked by DVC:

| DVC file | Output path | Files |
| --- | --- | ---: |
| `rain.dvc` | `data/weather/rain` | 6 |
| `wind.dvc` | `data/weather/wind` | 5 |
| `thunder.dvc` | `data/weather/thunder` | 5 |
| `metadata.dvc` | `data/weather/metadata` | 3 |

DVC remote:

`s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache`

Local note: this machine does not currently have the `dvc` command available.
For local verification, the already-pulled seed assets from the Desktop checkout
were copied into the ignored attempt-local weather folders. The source of truth
remains the DVC/S3 cache above.

## Site-Derived Pool

Site-derived candidates are not yet part of the runtime index.

Important manifests:

| Manifest | Meaning |
| --- | --- |
| `debug/site_weather_layer_d_assets_mvp_002_rain_expansion/layer_d_ready_manifest.csv` | 101 ready site assets; WAV files have been copied from Server A into local staging |
| `debug/layer_b_mvp_integrated_001/layer_b_mvp_manifest.csv` | 77-row integration draft: 64 site + 13 library |
| `debug/site_weather_candidate_pool_mvp_002_rain_expansion/candidate_pool_manifest.csv` | Larger candidate pool: accept / backup / reject |

Best observed site-ready counts:

| Pool label | Count |
| --- | ---: |
| rain | 6 |
| rain+wind | 7 |
| wind | 88 |

The Server A ready WAVs were found at:

`/home/ubuntu/layer_b_site_weather_job/runs/mvp_pool_20260531_002_rain_expansion_layer_d_assets_accept_only/assets_wav_22050_mono/`

They are now staged locally under:

`acoustic_ai/layers/layer_b/attempts/lucas__smoke_1__curated_assets/data/weather/site_mvp_002/`

Local staged counts: `rain_primary=6`, `rain_wind_mixed=7`, `wind_primary=88`.
They should be merged into runtime only after the index rows are converted from
the server `runs/...` paths to these local staged paths, or after the same assets
are stored in a durable DVC/S3-backed location.

## MVP Runtime Policy

For the current MVP:

- `rain`, `wind`, `thunder`, and `storm` resolve to materialized library assets.
- `wind` now resolves to pure wind assets instead of falling back to `rain+wind`.
- Site assets are locally staged but not yet active in the runtime index.
- Handler fallback is acceptable for intensity mismatch, but not for silently
  replacing a requested weather type with a mixed asset when pure assets exist.
