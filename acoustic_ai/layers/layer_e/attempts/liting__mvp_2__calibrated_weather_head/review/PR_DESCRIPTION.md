# Layer E-B Weather Analysis MVP Attempts

## Summary

This PR adds Liting's Layer E-B weather analysis work from the smoke baseline through MVP5.

It includes:

- E-B weather analysis smoke baseline for Site257 wind/rain assets.
- MVP1 PANNs baseline.
- MVP2 calibrated Site257 weather head, currently the safest frontend/demo integration candidate.
- MVP3 balanced weather head.
- MVP4 data-expanded weather head.
- MVP5 CLAP weather probe.
- Registry entries for Liting E-B attempts in `acoustic_ai/registry.yaml`.
- Tests for E-B smoke/MVP report shape.
- Dev Analysis frontend wiring so the weather head prefers MVP2 for demo output.
- Reviewer evidence matrix listing exact Site257 sample clips and per-attempt outputs.

## Current Best Candidate

`liting__mvp_2__calibrated_weather_head` remains the safest integration candidate because it has a stable checkpoint path and is already wired for Dev Analysis output.

`liting__mvp_5__clap_weather_probe` is the strongest fixed-review-matrix model result. It passes all 10 default review cases after thunder is removed from E-B scope, including both mixed rain+wind cases.

## Reviewer Evidence

Main evidence files:

- `acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/REVIEW_TEST_CASES.md`
- `acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/review/MVP1_TO_MVP5_REVIEW_EVIDENCE.md`
- `acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/review/murphy_site257_fixed_review_matrix.csv`
- `acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/review/mvp1_to_mvp5_cross_review_results.json`
- `acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/review/murphy_rain_pool_fullstack_cross_check.csv`

The fixed reviewer matrix uses Murphy's audited Site257 Layer B weather asset index:

- `source_type=site` only.
- Eligible pools: `site_ready_pool` and `site_backup_pool`.
- Sound-library rows are not used for E-B reviewer evidence.
- Thunder is excluded from the default E-B acceptance scope because the current Site257 pool does not contain enough reliable thunder evidence.

## Fixed Review Matrix Coverage

| Scene | Cases | Note |
|---|---:|---|
| `light_rain` | 1 | Only one Murphy-audited Site257 light-rain row exists in the current index. |
| `heavy_rain` | 1 | Only one Murphy-audited Site257 heavy-rain row exists in the current index. |
| `moderate_rain` | 2 | Two rain-positive Site257 review cases. |
| `mixed_rain_wind` | 2 | Two hard mixed-weather cases. |
| `breezing_light_wind` | 2 | Two selected from Site257 light-wind rows. |
| `strong_wind` | 2 | Two selected from Site257 strong-wind rows. |

Total: 10/10 default review samples are locally runnable after DVC materialisation.

## Additional Rain Cross-Check

I also reran E-B on Murphy's Site257 human-audited rain generation pool as an
independent validation set. These clips are **not** claimed as Liting MVP1-MVP5
training data; they are included only as external cross-check evidence for E-B
rain output behaviour.

Run setup:

- Environment: Server B (`shinypokemon`)
- Backend: Murphy E-B full stack (`CLAP + PANNs + AST + gate fusion`)
- Cross-check samples: 72 Site257 WAV clips
- Result CSV:
  `acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/review/murphy_rain_pool_fullstack_cross_check.csv`

Cross-check summary:

| Result | Count |
|---|---:|
| E-B `rain` | 68 |
| E-B `rain+wind` | 4 |
| E-B rain present | 72 |
| E-B wind present | 4 |
| E-B thunder present | 0 |
| E-B light rain, wind none | 9 |
| E-B heavy rain, wind none | 59 |

Example added light-rain validation rows:

| Clip | E-B result |
|---|---|
| `002__rec_214659__clip_023__win_00__rain.wav` | rain=light, wind=none |
| `003__rec_214659__clip_023__win_01__rain.wav` | rain=light, wind=none |

Example added heavy-rain validation rows:

| Clip | E-B result |
|---|---|
| `005__rec_214665__clip_008__win_00__rain.wav` | rain=heavy, wind=none |
| `006__rec_214665__clip_008__win_01__rain.wav` | rain=heavy, wind=none |

## Sample Results

| Scene | Expected | MVP2 Result | MVP5 Result |
|---|---|---|---|
| `light_rain` | rain=light, wind=none | pass | pass |
| `heavy_rain` | rain=heavy, wind=none | partial | pass |
| `moderate_rain` case 1 | rain=moderate, wind=none | partial | pass |
| `moderate_rain` case 2 | rain=moderate, wind=none | partial | pass |
| `mixed_rain_wind` case 1 | rain=moderate, wind=moderate | partial | pass |
| `mixed_rain_wind` case 2 | rain=moderate, wind=moderate | partial | pass |
| `breezing_light_wind` case 1 | rain=none, wind=light | pass | pass |
| `breezing_light_wind` case 2 | rain=none, wind=light | pass | pass |
| `strong_wind` case 1 | rain=none, wind=strong | partial | pass |
| `strong_wind` case 2 | rain=none, wind=strong | partial | pass |

MVP5 summary on this fixed matrix:

- Pass: 10
- Partial: 0
- Fail: 0

MVP2 remains preferred for integration because it is deploy-stable. MVP5 is documented as the stronger model candidate once CLAP dependency/caching is guaranteed in the deployment environment.

## Exact Sample Clips

The exact WAV paths and original Site257 S3 clip URIs are documented in:

- `acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/REVIEW_TEST_CASES.md`
- `acoustic_ai/layers/layer_e/attempts/liting__mvp_2__calibrated_weather_head/review/MVP1_TO_MVP5_REVIEW_EVIDENCE.md`

Example local WAV path format:

```text
acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/data/weather/site_mvp_002/rain_primary/...
```

Example original source path format:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/...
```

## Known Data Limitation

The fixed 10-case MVP1-MVP5 matrix is still constrained by the current Layer B
weather asset index:

- Current Murphy-audited Site257 index has one true light-rain row.
- Current Murphy-audited Site257 index has one true heavy-rain row.
- Previously scouted local pure-rain folders are not promoted as true-rain evidence because Murphy rejected that batch as not pure rain.

This PR does not fake additional light/heavy rain rows inside the fixed matrix.
Instead, it documents the separate 72-clip Murphy rain-pool cross-check as
external validation evidence. Expanding production training coverage still
requires a larger audited Site257 rain pool.

## Thunder Scope

Thunder is removed from the E-B acceptance scope for this PR.

Reason:

- The current Site257 pool has only backup/uncertain thunder candidates.
- The client/team direction is that unsupported Site257 thunder should not be claimed.
- E-B currently reports rain and wind weather layers. Thunder can be reopened later if a reliable Site257 thunder pool is found.

## Validation

Validation already completed on this branch:

- Server B reviewer matrix run with MVP1-MVP5.
- MVP2-MVP5 checkpoint artifacts materialized.
- MVP5 CLAP run enabled on Server B with `LITING_EB_RUN_CLAP=1`.
- GitHub checks passed after latest conflict resolution.
