# E-B MVP2 Review Test Cases

Owner: `liting`

This file answers the PR review request for explicit sample clips and sample
results. The current integration candidate is:

```text
liting__mvp_2__calibrated_weather_head
```

For the cross-attempt reviewer matrix covering MVP1 through MVP5, see:

```text
acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/review/MVP1_TO_MVP5_REVIEW_EVIDENCE.md
```

Generated evidence files:

```text
acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/review/murphy_site257_fixed_review_matrix.csv
acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/review/mvp1_to_mvp5_cross_review_results.json
```

Latest cross-attempt reviewer evidence:

| Item | Status |
|---|---|
| Run environment | Server B (`shinypokemon`) isolated checkout |
| Site257 WAV materialization | 10/10 default review samples available |
| MVP2 checkpoint | materialized |
| MVP3 checkpoint | materialized |
| MVP4 checkpoint | materialized |
| MVP5 checkpoint | materialized |
| MVP5 CLAP run | enabled on Server B with `LITING_EB_RUN_CLAP=1` |

The cross-attempt matrix now runs MVP1-MVP5 against the same fixed sample list.
The current audited Site257 index still only contains one `light_rain` row and
one `heavy_rain` row, so the two-case reviewer bar is met for moderate rain,
mixed rain/wind, breezing wind, and strong wind, but not yet for light rain or
heavy rain without expanding the audited Site257 rain pool.

Thunder is no longer part of the default E-B acceptance scope. The current
Site257 pool has only backup/uncertain thunder candidates, so E-B should not
claim a thunder detector until a reliable Site257 thunder pool exists.

## Data Used

The MVP2 calibrated weather head was trained and validated on the Site257
weather policy snapshot:

```text
acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/data/site257_weather_policy_snapshot.csv
```

The policy snapshot was produced before the Layer B asset folder was renamed,
so some `clip_path` values inside that CSV still contain the historical
`lucas__smoke_1__curated_assets` folder name. In the current main-aligned tree,
the materialised / DVC-tracked weather assets live here:

```text
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/
```

To materialise the clips for listening review:

```bash
dvc pull acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002.dvc
dvc pull acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_nov2019_storm_scout_001.dvc
dvc pull model/candidates/liting/mvp_2__calibrated_weather_head/weather_head.pt.dvc
```

Snapshot coverage:

| Scene | Count in policy snapshot | Notes |
|---|---:|---|
| Rain primary | 6 | Site257-only rain clips. |
| Rain + wind mixed | 12 | Site257 mixed weather clips. |
| Wind primary | 93 | Main validated coverage for wind. |
| Thunder backup | 2 | Diagnostic only; excluded from the default E-B acceptance scope. |

Intensity coverage:

| Target | Available Site257 clips | Review note |
|---|---:|---|
| Light rain | 1 | Not enough for the requested two-case review bar yet. |
| Moderate rain | 16 | Includes rain-primary and rain+wind rows. |
| Heavy rain | 1 | Not enough for the requested two-case review bar yet. |
| Light/breezing wind | 3 | Enough for spot checks, but only one landed in the MVP2 held-out split. |
| Moderate wind | 71 | Strongest coverage. |
| Strong/heavy wind | 31 | Strong coverage. |
| Thunder | 2 | Backup/uncertain evidence only; not claimed as supported in E-B MVP. |

## Validation Split

Latest Server B MVP2 run:

| Metric | Value |
|---|---:|
| Total materialised Site257 clips | 101 |
| Train split | 75 |
| Validation split | 26 |
| Rain validation accuracy | 0.769 |
| Wind validation accuracy | 0.731 |
| Joint validation accuracy | 0.615 |
| Single-component joint accuracy | 0.818 |
| Calibration-head training time | 1.45 s |
| Total run time including feature extraction | 18.26 s |

Full validation rows are in:

```text
model/candidates/liting/mvp_2__calibrated_weather_head/metrics.json
```

## Sample Results

### Rain Primary

| Clip ID | Expected | Predicted | Confidence | Status |
|---|---|---|---:|---|
| `site_mvp002_site257_1313196_000778_000793` | rain=moderate, wind=none | rain=moderate, wind=none | rain 0.995, wind 1.000 | pass |
| `site_mvp002_site257_1313196_001630_001645` | rain=moderate, wind=none | rain=moderate, wind=none | rain 0.990, wind 1.000 | pass |

Source paths:

```text
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/rain_primary__rain__site257_1313196_000778_000793.wav
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/rain_primary__rain__site257_1313196_001630_001645.wav
```

Original Site257 S3 clips:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_1313196/site_257_item_1313196_clip_003.webm
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_1313196/site_257_item_1313196_clip_006.webm
```

### Wind: Breezing / Light

| Clip ID | Expected | Predicted | Confidence | Status |
|---|---|---|---:|---|
| `site_mvp002_site257_5495_003676_003691` | rain=none, wind=light | rain=none, wind=light | rain 0.911, wind 0.979 | pass |

The policy snapshot has three light-wind clips, but only one landed in the
held-out MVP2 validation split. Two additional light-wind policy rows are
available for future fixed-scene spot testing:

```text
site_mvp002_site257_5299_002749_002764
site_mvp002_site257_214837_001286_001301
```

Current main-aligned clip paths:

```text
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_5495_003676_003691.wav
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_5299_002749_002764.wav
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_214837_001286_001301.wav
```

Original Site257 S3 clips:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_5495/site_257_item_5495_clip_013.webm
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_5299/site_257_item_5299_clip_010.webm
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_214837/site_257_item_214837_clip_005.webm
```

### Wind: Strong / Heavy

| Clip ID | Expected | Predicted | Confidence | Status |
|---|---|---|---:|---|
| `site_mvp002_site257_5493_000879_000894` | rain=none, wind=strong | rain=none, wind=strong | rain 1.000, wind 0.526 | pass |
| `site_mvp002_site257_216466_002726_002741` | rain=none, wind=strong | rain=none, wind=strong | rain 0.956, wind 0.820 | pass |

Current main-aligned clip paths:

```text
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_5493_000879_000894.wav
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/wind_primary/wind_primary__wind__site257_216466_002726_002741.wav
```

Original Site257 S3 clips:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_5493/site_257_item_5493_clip_003.webm
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_216466/site_257_item_216466_clip_010.webm
```

### Mixed Rain + Wind

These are hard cases for MVP2, but the MVP5 CLAP weather probe improves them
on the fixed reviewer matrix:

| Clip ID | Expected | Predicted | Status |
|---|---|---|---|
| `site_mvp002_site257_1313184_006689_006704` | rain=moderate, wind=moderate | MVP2: rain=moderate, wind=strong; MVP5: rain=moderate, wind=moderate | MVP2 partial, MVP5 pass |
| `site_mvp002_site257_1539525_006050_006065` | rain=moderate, wind=moderate | MVP2: rain=none, wind=moderate; MVP5: rain=moderate, wind=moderate | MVP2 partial, MVP5 pass |

Current main-aligned clip paths:

```text
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_wind_mixed/rain_wind_mixed__rain_wind__site257_1313184_006689_006704.wav
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_wind_mixed/rain_wind_mixed__rain_wind__site257_1539525_006050_006065.wav
```

Original Site257 S3 clips:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_1313184/site_257_item_1313184_clip_024.webm
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_1539525/site_257_item_1539525_clip_021.webm
```

This is why the PR keeps MVP2 as the safe current frontend/integration
candidate, while documenting MVP5 as the stronger fixed-matrix candidate for
mixed rain+wind once its CLAP dependency is guaranteed in the deployment
environment.

### Thunder

Thunder is intentionally excluded from the default E-B acceptance scope. The
current Site257 pool has only two backup/uncertain thunder candidates:

```text
site_nov2019_storm001_site257_214872_001700_001730
site_nov2019_storm001_site257_214707_000781_000811
```

Current main-aligned clip paths:

```text
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_nov2019_storm_scout_001/thunder_backup/thunder_backup__thunder__site257_214872_001700_001730.wav
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_nov2019_storm_scout_001/thunder_backup/thunder_backup__thunder__site257_214707_000781_000811.wav
```

Original Site257 S3 clips:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_214872/site_257_item_214872_clip_006.webm
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_214707/site_257_item_214707_clip_003.webm
```

Because these are not enough to calibrate a robust thunder detector, E-B does
not claim thunder support in this PR. If the diagnostic path is explicitly
enabled later, the MVP2 output contract returns:

```json
{
  "thunder_intensity": "none",
  "thunder_status": "suppressed_until_site257_evidence_is_validated"
}
```

## Review Interpretation

The current E-B PR should be read as:

- MVP2 is the safest current integration candidate.
- MVP2 passes the rain/wind validation gate for single-component weather.
- MVP5 improves mixed rain+wind on the fixed reviewer matrix, passing both
  selected mixed cases.
- Thunder is not part of the current E-B acceptance scope because there is no
  reliable Site257 thunder pool.
- The remaining data blocker is light/heavy rain coverage: the Murphy-audited
  Site257 index has only one true light-rain row and one true heavy-rain row.
  The previously scouted local pure-rain folder is not promoted here because
  Murphy rejected that batch as not pure rain.
