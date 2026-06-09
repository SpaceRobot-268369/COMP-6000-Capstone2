# E-B MVP1-MVP5 Fixed Review Evidence

Owner: `liting`

This review set is built from Murphy's audited Site257 Layer B weather asset index.
It answers PR #34 reviewer questions about exact sample clips and sample results.

An additional 72-clip rain-only cross-check was run after the fixed matrix using
Murphy's Site257 human-audited rain generation pool. That pool is not claimed as
Liting training data; it is external validation evidence for E-B rain behaviour.

Thunder is intentionally excluded from the default E-B acceptance matrix because the current Site257 pool only has backup/uncertain thunder evidence. E-B now reports rain/wind weather layers; thunder can be reopened later if a validated Site257 thunder pool exists.

## Source

- Asset index: `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/asset_index.csv`
- Source restriction: `source_type=site` only.
- Eligible pools: `site_ready_pool` and `site_backup_pool`.
- Excluded scene: `thunder_backup`.

## Coverage

| Review scene | Cases in matrix | Locally runnable now | Note |
|---|---:|---:|---|
| `light_rain` | 1 | 1 | Only one Murphy-audited Site257 light-rain row exists in the current index. |
| `heavy_rain` | 1 | 1 | Only one Murphy-audited Site257 heavy-rain row exists in the current index. |
| `moderate_rain` | 2 | 2 | Two rain-positive Site257 review cases. |
| `mixed_rain_wind` | 2 | 2 | Two hard mixed-weather cases; MVP5 passes both. |
| `breezing_light_wind` | 2 | 2 | Two selected from three available light-wind rows. |
| `strong_wind` | 2 | 2 | Two selected from strong Site257 wind rows. |

## External Rain Cross-Check

This cross-check uses Murphy's Site257 rain generation training pool as
independent validation samples. It was run on Server B with Murphy E-B full
stack enabled:

```text
CLAP + PANNs + AST + gate fusion
```

Evidence file:

```text
acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/review/murphy_rain_pool_fullstack_cross_check.csv
```

Summary:

| Metric | Value |
|---|---:|
| Total Site257 WAVs | 72 |
| Murphy pool label: `rain` | 68 |
| Murphy pool label: `rain+wind` | 4 |
| E-B result: `rain` | 68 |
| E-B result: `rain+wind` | 4 |
| E-B rain present | 72 |
| E-B wind present | 4 |
| E-B thunder present | 0 |
| E-B light rain, wind none | 9 |
| E-B heavy rain, wind none | 57 |

Representative additional light-rain validation cases:

| Filename | E-B result | Rain confidence | Wind result | Source |
|---|---|---:|---|---|
| `002__rec_214659__clip_023__win_00__rain.wav` | rain=light | 0.428503 | wind=none | Site257 S3 URI in cross-check CSV |
| `003__rec_214659__clip_023__win_01__rain.wav` | rain=light | 0.423160 | wind=none | Site257 S3 URI in cross-check CSV |
| `007__rec_214665__clip_008__win_02__rain.wav` | rain=light | 0.582077 | wind=none | Site257 S3 URI in cross-check CSV |
| `009__rec_216095__clip_020__win_01__rain.wav` | rain=light | 0.587258 | wind=none | Site257 S3 URI in cross-check CSV |

Representative additional heavy-rain validation cases:

| Filename | E-B result | Rain confidence | Wind result | Source |
|---|---|---:|---|---|
| `005__rec_214665__clip_008__win_00__rain.wav` | rain=heavy | 0.591543 | wind=none | Site257 S3 URI in cross-check CSV |
| `006__rec_214665__clip_008__win_01__rain.wav` | rain=heavy | 0.587007 | wind=none | Site257 S3 URI in cross-check CSV |
| `010__rec_216095__clip_020__win_02__rain.wav` | rain=heavy | 0.597191 | wind=none | Site257 S3 URI in cross-check CSV |
| `011__rec_216095__clip_021__win_00__rain.wav` | rain=heavy | 0.597430 | wind=none | Site257 S3 URI in cross-check CSV |

The cross-check does not change the fixed 10-case MVP1-MVP5 matrix. It adds
reviewer-facing evidence that additional Site257 rain samples can be analysed
with the expected E-B schema and rain/wind separation.

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
| `liting__mvp_1__panns_weather_baseline` | 0 | 10 | 0 | 0 | 0 |
| `liting__mvp_2__calibrated_weather_head` | 3 | 7 | 0 | 0 | 0 |
| `liting__mvp_3__balanced_weather_head` | 4 | 6 | 0 | 0 | 0 |
| `liting__mvp_4__data_expanded_weather_head` | 5 | 5 | 0 | 0 | 0 |
| `liting__mvp_5__clap_weather_probe` | 10 | 0 | 0 | 0 | 0 |

## Sample Results

| Scene | Asset ID | Expected rain | Expected wind | Local WAV | MVP1 | MVP2 | MVP3 | MVP4 | MVP5 |
|---|---|---|---|---|---|---|---|---|---|
| `light_rain` | `site_mvp002_site257_214665_001923_001938` | light | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/rain_primary__rain__site257_214665_001923_001938.wav` | partial (rain=heavy 0.816, wind=none 0.561) | pass (rain=light 1.0, wind=none 0.863) | pass (rain=light 0.998, wind=none 0.997) | pass (rain=light 0.353, wind=none 1.0) | pass (rain=light 0.994, wind=none 1.0) |
| `heavy_rain` | `site_mvp002_site257_1401640_003387_003402` | heavy | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/rain_primary__rain__site257_1401640_003387_003402.wav` | partial (rain=light 0.632, wind=strong 0.9) | partial (rain=light 0.641, wind=strong 0.804) | partial (rain=light 0.811, wind=strong 0.828) | partial (rain=moderate 0.328, wind=none 0.633) | pass (rain=heavy 0.972, wind=none 1.0) |
| `moderate_rain` | `site_mvp002_site257_1313196_000202_000217` | moderate | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/rain_primary__rain__site257_1313196_000202_000217.wav` | partial (rain=heavy 0.9, wind=none 0.565) | partial (rain=light 1.0, wind=none 0.999) | partial (rain=light 1.0, wind=none 1.0) | pass (rain=moderate 0.412, wind=none 1.0) | pass (rain=moderate 0.978, wind=none 1.0) |
| `moderate_rain` | `site_mvp002_site257_1313196_000778_000793` | moderate | none | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/rain_primary__rain__site257_1313196_000778_000793.wav` | partial (rain=heavy 0.9, wind=none 0.553) | partial (rain=light 1.0, wind=none 0.999) | partial (rain=light 1.0, wind=none 1.0) | pass (rain=moderate 0.407, wind=none 1.0) | pass (rain=moderate 0.858, wind=none 1.0) |
| `mixed_rain_wind` | `site_mvp002_site257_1313184_006689_006704` | moderate | moderate | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_wind_mixed/rain_wind_mixed__rain_wind__site257_1313184_006689_006704.wav` | partial (rain=none 0.727, wind=strong 0.9) | partial (rain=moderate 0.932, wind=strong 0.449) | partial (rain=moderate 0.963, wind=light 0.665) | partial (rain=none 0.31, wind=light 0.895) | pass (rain=moderate 0.793, wind=moderate 0.959) |
| `mixed_rain_wind` | `site_mvp002_site257_1539525_006050_006065` | moderate | moderate | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_wind_mixed/rain_wind_mixed__rain_wind__site257_1539525_006050_006065.wav` | partial (rain=none 0.729, wind=strong 0.85) | partial (rain=none 0.587, wind=moderate 0.521) | pass (rain=moderate 0.924, wind=moderate 0.643) | partial (rain=none 0.284, wind=light 0.596) | pass (rain=moderate 0.858, wind=moderate 1.0) |
| `breezing_light_wind` | `site_mvp002_site257_214837_001286_001301` | none | light | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_214837_001286_001301.wav` | partial (rain=none 0.719, wind=strong 0.892) | pass (rain=none 0.849, wind=light 0.578) | pass (rain=none 0.805, wind=light 0.724) | pass (rain=none 0.27, wind=light 0.892) | pass (rain=none 0.992, wind=light 0.994) |
| `breezing_light_wind` | `site_mvp002_site257_5299_002749_002764` | none | light | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_5299_002749_002764.wav` | partial (rain=none 0.753, wind=strong 0.9) | pass (rain=none 0.989, wind=light 0.727) | pass (rain=none 0.999, wind=light 0.839) | pass (rain=none 0.346, wind=light 0.924) | pass (rain=none 0.999, wind=light 1.0) |
| `strong_wind` | `site_mvp002_site257_1313184_000390_000405` | none | strong | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_1313184_000390_000405.wav` | partial (rain=none 0.764, wind=strong 0.9) | partial (rain=none 0.966, wind=moderate 0.603) | partial (rain=none 0.968, wind=moderate 0.497) | partial (rain=none 0.327, wind=light 0.621) | pass (rain=none 0.661, wind=strong 1.0) |
| `strong_wind` | `site_mvp002_site257_1313184_001227_001242` | none | strong | `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_1313184_001227_001242.wav` | partial (rain=none 0.756, wind=strong 0.9) | partial (rain=none 0.954, wind=moderate 0.497) | partial (rain=none 0.933, wind=light 0.482) | partial (rain=none 0.362, wind=light 0.705) | pass (rain=none 0.355, wind=strong 1.0) |

## Interpretation

- This matrix confirms that the current Murphy-audited Site257 pool contains multiple rain/wind combinations.
- The matrix has 10/10 locally runnable samples after removing thunder from the default E-B scope.
- MVP5 is the strongest fixed-matrix result: 10 exact passes, 0 partials, and 0 failures.
- MVP5 improves mixed rain+wind classification: both selected mixed rain+wind cases pass as rain=moderate and wind=moderate.
- MVP2 remains the safest current frontend/integration candidate because it is already wired for demo output and has the stable checkpoint path.
- The fixed 10-case matrix still has one light-rain and one heavy-rain case because it is constrained to the current Layer B weather asset index. The separate Murphy rain-pool cross-check adds additional independent Site257 light/heavy rain validation samples, but those clips are documented as external validation samples rather than Liting training data.
