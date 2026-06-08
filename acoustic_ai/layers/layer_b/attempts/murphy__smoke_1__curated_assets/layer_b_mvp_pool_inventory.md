# Layer B MVP Pool Inventory

Last updated: 2026-06-02

## Runtime Pool

The frontend-facing Layer B MVP reads this index:

`acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/asset_index.csv`

Current site-only runtime index contents:

| Source | Rows | Runtime status |
| --- | ---: | --- |
| site | 113 | Active runtime pool |
| sound_library | 40 | Marked unavailable; excluded from runtime |

Runtime-usable and locally materialized rows after path cleanup:

| Primary weather | Count |
| --- | ---: |
| wind | 88 |
| rain | 6 |
| rain+wind | 12 |
| thunder | 2 |

The old seed rows previously pointed at `acoustic_ai/data/weather/...`. They
now point at the current attempt-local data root:

`acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/...`

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

Site-derived candidates are now the only runtime pool for the site-only branch.

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
| rain+wind | 12 |
| wind | 93 |
| thunder | 2 |

The Server A ready WAVs were found at:

`/home/ubuntu/layer_b_site_weather_job/runs/mvp_pool_20260531_002_rain_expansion_layer_d_assets_accept_only/assets_wav_22050_mono/`

They are now staged locally under:

`acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/`

Local staged counts: `rain_primary=6`, `rain_wind_mixed=7`, `wind_primary=88`.
The runtime index now points to these local staged paths. The WAV folders remain
ignored local materialized assets; the CSV index records the active site-only
runtime pool.

Additional Nov 2019 storm-period scout assets were promoted from
`debug/site_weather_nov2019_storm_scout_001/listen.html` after human review:
`rain+wind=5`, `wind=5`, `thunder=2`. These WAVs are staged under
`data/weather/site_nov2019_storm_scout_001/`. The two thunder rows are marked
`backup` because the human notes still describe them as uncertain thunder-like
events.

## MVP Runtime Policy

For the current site-only MVP branch:

- `rain` resolves to 6 materialized site assets.
- `wind` resolves to 93 materialized site assets.
- `rain+wind` resolves to 12 materialized site mixed-weather assets.
- `thunder` has 2 site-derived backup rows only; use cautiously because both
  were marked as uncertain by human review.
- Sound-library assets are retained in the index for provenance but marked
  unavailable and excluded from runtime.
- `storm` / `rain+thunder` / `rain+thunder+wind` are not yet active runtime
  categories. The Nov 2019 scout improved mixed-weather coverage, but did not
  produce clean enough site-derived storm clips to promote as primary storm.
- Handler fallback is acceptable for intensity mismatch, but not for silently
  replacing a requested weather type with a mixed asset when pure assets exist.

Current frontend-facing weather controls:

| Weather type | Active? | Source | Notes |
| --- | --- | --- | --- |
| `rain` | yes | site only | Pure rain rows where available; intensity fallback may select nearby rain intensity. |
| `wind` | yes | site only | Strongest current pool; includes light/medium/heavy wind rows. |
| `rain+wind` | yes | site only | Mixed weather stem, not pure rain and not pure wind. |
| `thunder` | backup only | site only | Two uncertain thunder-like rows from Nov 2019 scout; not enough for a reliable primary pool. |
| `storm` / `rain+thunder` / `rain+thunder+wind` | no | site only target, no active rows | Future category if reliable site storm clips are promoted. |
