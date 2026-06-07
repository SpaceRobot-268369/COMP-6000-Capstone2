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
| `light_rain` | 1 | 0 | Only one audited Site257 light-rain row exists in the current index. |
| `heavy_rain` | 1 | 1 | Only one audited Site257 heavy-rain row exists in the current index. |
| `moderate_rain` | 2 | 2 | Two selected as rain-positive review cases. |
| `mixed_rain_wind` | 2 | 2 | Known hard mixed-weather cases. |
| `breezing_light_wind` | 2 | 1 | Two selected from three available light-wind rows. |
| `strong_wind` | 2 | 1 | Two selected from strong Site257 wind rows. |
| `thunder_backup` | 2 | 0 | Backup-only; E-B candidates suppress thunder until more site evidence exists. |

## Attempt Checkpoint State

| Attempt | State |
|---|---|
| `liting__mvp_1__panns_weather_baseline` | `no_checkpoint_required` |
| `liting__mvp_2__calibrated_weather_head` | `checkpoint_materialized` |
| `liting__mvp_3__balanced_weather_head` | `checkpoint_pointer_only:weather_head.pt.dvc` |
| `liting__mvp_4__data_expanded_weather_head` | `checkpoint_materialized` |
| `liting__mvp_5__clap_weather_probe` | `checkpoint_materialized` |

## Sample Results

| Scene | Asset ID | Expected rain | Expected wind | Expected thunder | Local WAV | MVP1 | MVP2 | MVP3 | MVP4 | MVP5 |
|---|---|---|---|---|---|---|---|---|---|---|
| `light_rain` | `site_mvp002_site257_214665_001923_001938` | light | none | none | `not materialized locally` | not run: wav not local | not run: wav not local | not run: wav not local | not run: wav not local | not run: wav not local |
| `heavy_rain` | `site_mvp002_site257_1401640_003387_003402` | heavy | none | none | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/rain_primary/rain_primary__rain__site257_1401640_003387_003402.wav` | partial (rain=light 0.923, wind=strong 0.596, thunder=none 0.794) | partial (rain=light 0.641, wind=strong 0.804, thunder=none 0.9) | not_run: checkpoint_pointer_only:weather_head.pt.dvc | partial (rain=moderate 0.328, wind=none 0.633, thunder=none 0.9) | not_run: CLAP backbone is not cached locally in this environment; running it would attempt HuggingFace network access. Use the existing MVP5 metrics or rerun this script on Server B / a machine with the CLAP cache materialised. |
| `moderate_rain` | `site_mvp002_site257_1313196_000202_000217` | moderate | none | none | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/rain_primary/rain_primary__rain__site257_1313196_000202_000217.wav` | pass (rain=moderate 0.489, wind=none 0.95, thunder=none 0.556) | partial (rain=light 1.0, wind=none 0.999, thunder=none 0.9) | not_run: checkpoint_pointer_only:weather_head.pt.dvc | pass (rain=moderate 0.412, wind=none 1.0, thunder=none 0.9) | not_run: CLAP backbone is not cached locally in this environment; running it would attempt HuggingFace network access. Use the existing MVP5 metrics or rerun this script on Server B / a machine with the CLAP cache materialised. |
| `moderate_rain` | `site_mvp002_site257_1313196_000778_000793` | moderate | none | none | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/rain_primary/rain_primary__rain__site257_1313196_000778_000793.wav` | pass (rain=moderate 0.477, wind=none 0.95, thunder=none 0.556) | partial (rain=light 1.0, wind=none 0.999, thunder=none 0.9) | not_run: checkpoint_pointer_only:weather_head.pt.dvc | pass (rain=moderate 0.407, wind=none 1.0, thunder=none 0.9) | not_run: CLAP backbone is not cached locally in this environment; running it would attempt HuggingFace network access. Use the existing MVP5 metrics or rerun this script on Server B / a machine with the CLAP cache materialised. |
| `mixed_rain_wind` | `site_mvp002_site257_1313184_006689_006704` | moderate | moderate | none | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/rain_wind_mixed/rain_wind_mixed__rain_wind__site257_1313184_006689_006704.wav` | partial (rain=none 0.862, wind=moderate 0.691, thunder=none 0.794) | partial (rain=moderate 0.932, wind=strong 0.449, thunder=none 0.9) | not_run: checkpoint_pointer_only:weather_head.pt.dvc | partial (rain=none 0.31, wind=light 0.895, thunder=none 0.9) | not_run: CLAP backbone is not cached locally in this environment; running it would attempt HuggingFace network access. Use the existing MVP5 metrics or rerun this script on Server B / a machine with the CLAP cache materialised. |
| `mixed_rain_wind` | `site_mvp002_site257_1539525_006050_006065` | moderate | moderate | none | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/rain_wind_mixed/rain_wind_mixed__rain_wind__site257_1539525_006050_006065.wav` | partial (rain=none 0.95, wind=light 0.711, thunder=none 0.794) | partial (rain=none 0.587, wind=moderate 0.521, thunder=none 0.9) | not_run: checkpoint_pointer_only:weather_head.pt.dvc | partial (rain=none 0.284, wind=light 0.596, thunder=none 0.9) | not_run: CLAP backbone is not cached locally in this environment; running it would attempt HuggingFace network access. Use the existing MVP5 metrics or rerun this script on Server B / a machine with the CLAP cache materialised. |
| `breezing_light_wind` | `site_mvp002_site257_214837_001286_001301` | none | light | none | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/wind_primary/wind_primary__wind__site257_214837_001286_001301.wav` | pass (rain=none 0.941, wind=light 0.849, thunder=none 0.794) | pass (rain=none 0.849, wind=light 0.578, thunder=none 0.9) | not_run: checkpoint_pointer_only:weather_head.pt.dvc | pass (rain=none 0.27, wind=light 0.892, thunder=none 0.9) | not_run: CLAP backbone is not cached locally in this environment; running it would attempt HuggingFace network access. Use the existing MVP5 metrics or rerun this script on Server B / a machine with the CLAP cache materialised. |
| `breezing_light_wind` | `site_mvp002_site257_5299_002749_002764` | none | light | none | `not materialized locally` | not run: wav not local | not run: wav not local | not run: wav not local | not run: wav not local | not run: wav not local |
| `strong_wind` | `site_mvp002_site257_1313184_000390_000405` | none | strong | none | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/wind_primary/wind_primary__wind__site257_1313184_000390_000405.wav` | partial (rain=none 0.95, wind=moderate 0.593, thunder=none 0.794) | partial (rain=none 0.966, wind=moderate 0.603, thunder=none 0.9) | not_run: checkpoint_pointer_only:weather_head.pt.dvc | partial (rain=none 0.327, wind=light 0.621, thunder=none 0.9) | not_run: CLAP backbone is not cached locally in this environment; running it would attempt HuggingFace network access. Use the existing MVP5 metrics or rerun this script on Server B / a machine with the CLAP cache materialised. |
| `strong_wind` | `site_mvp002_site257_1313184_001227_001242` | none | strong | none | `not materialized locally` | not run: wav not local | not run: wav not local | not run: wav not local | not run: wav not local | not run: wav not local |
| `thunder_backup` | `site_nov2019_storm001_site257_214707_000781_000811` | none | none | moderate | `not materialized locally` | not run: wav not local | not run: wav not local | not run: wav not local | not run: wav not local | not run: wav not local |
| `thunder_backup` | `site_nov2019_storm001_site257_214872_001700_001730` | none | none | strong | `not materialized locally` | not run: wav not local | not run: wav not local | not run: wav not local | not run: wav not local | not run: wav not local |

## Interpretation

- This matrix confirms that the current Murphy-audited Site257 pool does contain multiple weather combinations.
- It does not yet satisfy the reviewer bar for two local/runnable samples in every requested scene: light rain, heavy rain, and thunder are data-limited or not materialized locally on this machine.
- MVP3 is not rerun locally unless its DVC checkpoint is materialized; this avoids silently falling back to another model.
- MVP2 remains the safest current integration candidate, but this evidence should be treated as a fixed review audit rather than a claim that all requested scenes are solved.
