# E-B MVP1-MVP5 Fixed Review Evidence

Owner: `liting`

This review set is built from Murphy's audited Site257 Layer B weather asset index.
It is intended to answer PR #34 reviewer questions about exact sample clips and sample results.

## Source

- Asset index: `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/asset_index.csv`
- Source restriction: `source_type=site` only.
- Eligible pools: `site_ready_pool` and `site_backup_pool`.

## Coverage

| Review scene | Cases in matrix | Locally runnable now | Note |
|---|---:|---:|---|
| `light_rain` | 1 | 1 | Only one audited Site257 light-rain row exists in the current index. |
| `heavy_rain` | 1 | 1 | Only one audited Site257 heavy-rain row exists in the current index. |
| `moderate_rain` | 2 | 2 | Two selected as rain-positive review cases. |
| `mixed_rain_wind` | 2 | 2 | Known hard mixed-weather cases. |
| `breezing_light_wind` | 2 | 2 | Two selected from three available light-wind rows. |
| `strong_wind` | 2 | 2 | Two selected from strong Site257 wind rows. |
| `thunder_backup` | 2 | 2 | Backup-only; E-B candidates suppress thunder until more site evidence exists. |

## Attempt Checkpoint State

| Attempt | State |
|---|---|
| `liting__mvp_1__panns_weather_baseline` | `no_checkpoint_required` |
| `liting__mvp_2__calibrated_weather_head` | `checkpoint_materialized` |
| `liting__mvp_3__balanced_weather_head` | `checkpoint_materialized` |
| `liting__mvp_4__data_expanded_weather_head` | `checkpoint_materialized` |
| `liting__mvp_5__clap_weather_probe` | `checkpoint_materialized` |

## Attempt Result Summary

| Attempt | Pass | Partial | Fail | Not run | Error |
|---|---:|---:|---:|---:|---:|
| `liting__mvp_1__panns_weather_baseline` | 0 | 12 | 0 | 0 | 0 |
| `liting__mvp_2__calibrated_weather_head` | 3 | 7 | 2 | 0 | 0 |
| `liting__mvp_3__balanced_weather_head` | 4 | 6 | 2 | 0 | 0 |
| `liting__mvp_4__data_expanded_weather_head` | 5 | 5 | 2 | 0 | 0 |
| `liting__mvp_5__clap_weather_probe` | 10 | 0 | 2 | 0 | 0 |

## Sample Results

| Scene | Asset ID | Expected rain | Expected wind | Expected thunder | Local WAV | MVP1 | MVP2 | MVP3 | MVP4 | MVP5 |
|---|---|---|---|---|---|---|---|---|---|---|
| `light_rain` | `site_mvp002_site257_214665_001923_001938` | light | none | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/rain_primary__rain__site257_214665_001923_001938.wav` | partial (rain=heavy 0.816, wind=none 0.561, thunder=none 0.679) | pass (rain=light 1.0, wind=none 0.863, thunder=none 0.9) | pass (rain=light 0.998, wind=none 0.997, thunder=none 0.9) | pass (rain=light 0.353, wind=none 1.0, thunder=none 0.9) | pass (rain=light 0.994, wind=none 1.0, thunder=none 0.9) |
| `heavy_rain` | `site_mvp002_site257_1401640_003387_003402` | heavy | none | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/rain_primary__rain__site257_1401640_003387_003402.wav` | partial (rain=light 0.632, wind=strong 0.9, thunder=heavy 0.9) | partial (rain=light 0.641, wind=strong 0.804, thunder=none 0.9) | partial (rain=light 0.811, wind=strong 0.828, thunder=none 0.9) | partial (rain=moderate 0.328, wind=none 0.633, thunder=none 0.9) | pass (rain=heavy 0.972, wind=none 1.0, thunder=none 0.9) |
| `moderate_rain` | `site_mvp002_site257_1313196_000202_000217` | moderate | none | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/rain_primary__rain__site257_1313196_000202_000217.wav` | partial (rain=heavy 0.9, wind=none 0.565, thunder=none 0.656) | partial (rain=light 1.0, wind=none 0.999, thunder=none 0.9) | partial (rain=light 1.0, wind=none 1.0, thunder=none 0.9) | pass (rain=moderate 0.412, wind=none 1.0, thunder=none 0.9) | pass (rain=moderate 0.978, wind=none 1.0, thunder=none 0.9) |
| `moderate_rain` | `site_mvp002_site257_1313196_000778_000793` | moderate | none | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/rain_primary__rain__site257_1313196_000778_000793.wav` | partial (rain=heavy 0.9, wind=none 0.553, thunder=none 0.631) | partial (rain=light 1.0, wind=none 0.999, thunder=none 0.9) | partial (rain=light 1.0, wind=none 1.0, thunder=none 0.9) | pass (rain=moderate 0.407, wind=none 1.0, thunder=none 0.9) | pass (rain=moderate 0.858, wind=none 1.0, thunder=none 0.9) |
| `mixed_rain_wind` | `site_mvp002_site257_1313184_006689_006704` | moderate | moderate | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_wind_mixed/rain_wind_mixed__rain_wind__site257_1313184_006689_006704.wav` | partial (rain=none 0.727, wind=strong 0.9, thunder=heavy 0.864) | partial (rain=moderate 0.932, wind=strong 0.449, thunder=none 0.9) | partial (rain=moderate 0.963, wind=light 0.665, thunder=none 0.9) | partial (rain=none 0.31, wind=light 0.895, thunder=none 0.9) | pass (rain=moderate 0.793, wind=moderate 0.959, thunder=none 0.9) |
| `mixed_rain_wind` | `site_mvp002_site257_1539525_006050_006065` | moderate | moderate | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_wind_mixed/rain_wind_mixed__rain_wind__site257_1539525_006050_006065.wav` | partial (rain=none 0.729, wind=strong 0.85, thunder=heavy 0.821) | partial (rain=none 0.587, wind=moderate 0.521, thunder=none 0.9) | pass (rain=moderate 0.924, wind=moderate 0.643, thunder=none 0.9) | partial (rain=none 0.284, wind=light 0.596, thunder=none 0.9) | pass (rain=moderate 0.858, wind=moderate 1.0, thunder=none 0.9) |
| `breezing_light_wind` | `site_mvp002_site257_214837_001286_001301` | none | light | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_214837_001286_001301.wav` | partial (rain=none 0.719, wind=strong 0.892, thunder=heavy 0.735) | pass (rain=none 0.849, wind=light 0.578, thunder=none 0.9) | pass (rain=none 0.805, wind=light 0.724, thunder=none 0.9) | pass (rain=none 0.27, wind=light 0.892, thunder=none 0.9) | pass (rain=none 0.992, wind=light 0.994, thunder=none 0.9) |
| `breezing_light_wind` | `site_mvp002_site257_5299_002749_002764` | none | light | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_5299_002749_002764.wav` | partial (rain=none 0.753, wind=strong 0.9, thunder=heavy 0.9) | pass (rain=none 0.989, wind=light 0.727, thunder=none 0.9) | pass (rain=none 0.999, wind=light 0.839, thunder=none 0.9) | pass (rain=none 0.346, wind=light 0.924, thunder=none 0.9) | pass (rain=none 0.999, wind=light 1.0, thunder=none 0.9) |
| `strong_wind` | `site_mvp002_site257_1313184_000390_000405` | none | strong | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_1313184_000390_000405.wav` | partial (rain=none 0.764, wind=strong 0.9, thunder=heavy 0.9) | partial (rain=none 0.966, wind=moderate 0.603, thunder=none 0.9) | partial (rain=none 0.968, wind=moderate 0.497, thunder=none 0.9) | partial (rain=none 0.327, wind=light 0.621, thunder=none 0.9) | pass (rain=none 0.661, wind=strong 1.0, thunder=none 0.9) |
| `strong_wind` | `site_mvp002_site257_1313184_001227_001242` | none | strong | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_1313184_001227_001242.wav` | partial (rain=none 0.756, wind=strong 0.9, thunder=heavy 0.9) | partial (rain=none 0.954, wind=moderate 0.497, thunder=none 0.9) | partial (rain=none 0.933, wind=light 0.482, thunder=none 0.9) | partial (rain=none 0.362, wind=light 0.705, thunder=none 0.9) | pass (rain=none 0.355, wind=strong 1.0, thunder=none 0.9) |
| `thunder_backup` | `site_nov2019_storm001_site257_214707_000781_000811` | none | none | moderate | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_nov2019_storm_scout_001/thunder_backup/thunder_backup__thunder__site257_214707_000781_000811.wav` | partial (rain=none 0.761, wind=strong 0.9, thunder=heavy 0.9) | fail (rain=heavy 1.0, wind=light 1.0, thunder=none 0.9) | fail (rain=none 1.0, wind=light 0.757, thunder=none 0.9) | fail (rain=none 0.392, wind=light 0.997, thunder=none 0.9) | fail (rain=light 1.0, wind=none 1.0, thunder=none 0.9) |
| `thunder_backup` | `site_nov2019_storm001_site257_214872_001700_001730` | none | none | strong | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_nov2019_storm_scout_001/thunder_backup/thunder_backup__thunder__site257_214872_001700_001730.wav` | partial (rain=none 0.772, wind=strong 0.9, thunder=heavy 0.9) | fail (rain=heavy 1.0, wind=light 1.0, thunder=none 0.9) | fail (rain=none 0.999, wind=light 0.67, thunder=none 0.9) | fail (rain=none 0.401, wind=light 0.996, thunder=none 0.9) | fail (rain=light 1.0, wind=none 1.0, thunder=none 0.9) |

## Interpretation

- This matrix confirms that the current Murphy-audited Site257 pool does contain multiple weather combinations.
- The missing Site257 WAVs and all MVP2-MVP5 checkpoint artifacts were materialized on Server B, then the full matrix was rerun there.
- The matrix now has 12/12 locally runnable samples: all selected WAVs are materialized and all MVP1-MVP5 attempts have a resolved checkpoint state.
- MVP5 is the strongest result on this fixed matrix: 10 exact passes and 2 failures. The two failures are both thunder backup cases.
- MVP2 remains the safest current frontend/integration candidate because it is already wired for demo output, but MVP5 should be treated as the strongest candidate-model result from this review run.
- The reviewer bar is still not fully satisfied for every requested scene because the current audited Site257 index only contains one `light_rain` row and one `heavy_rain` row. Adding two local examples for those exact scenes requires either expanding the audited Site257 rain pool or relaxing the scene definition to rain-positive cases.
