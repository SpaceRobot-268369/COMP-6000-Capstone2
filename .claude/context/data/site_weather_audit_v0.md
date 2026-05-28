# Site Weather Audit v0

Source: first MVP site-weather listening audit from the
`site_weather_audit_v0` batch.

Scope:

- Reviewed clips: 1-90
- Excluded from this summary: 91-100 `quiet_ambience_control`
- Reason for exclusion: quiet ambience belongs to Layer A, not this
  site-weather candidate pool.

## Overall Result

| Decision | Count |
|---|---:|
| `yes` | 22 |
| `maybe` | 20 |
| `no` | 48 |
| reviewed total | 90 |

If `maybe` is treated as usable-with-caution, the first pass yields 42/90
usable or semi-usable clips. If only `yes` is accepted, it yields 22/90.

## Result by Candidate Bucket

| Candidate bucket | Reviewed | Yes | Maybe | No | Main issue |
|---|---:|---:|---:|---:|---|
| `light_rain` | 20 | 3 | 6 | 11 | Low precipitation often does not produce audible rain; many clips are wind/insect dominated. |
| `medium_or_heavy_rain` | 20 | 8 | 6 | 6 | Best-performing bucket; intensity is often overestimated. |
| `light_wind` | 20 | 3 | 4 | 13 | Frequently sounds like rain or insect/bird ambience rather than wind. |
| `medium_wind` | 20 | 8 | 4 | 8 | Better than light wind, but still often contaminated by biological sound or rain. |
| `storm_or_thunder_prior` | 10 | 0 | 0 | 10 | No thunder/storm events found; env storm prior alone is not usable. |

## Key Findings

### Rain

`medium_or_heavy_rain` is the strongest current bucket.

Good examples:

- `021_medium_or_heavy_rain`: usable medium rain, slight knocking/noise.
- `022_medium_or_heavy_rain`: usable heavy rain.
- `023_medium_or_heavy_rain`: usable heavy rain with medium wind.
- `027_medium_or_heavy_rain`: usable medium rain.
- `028_medium_or_heavy_rain`: usable medium rain.
- `030_medium_or_heavy_rain`: clean heavy rain + wind, no insect noise.
- `037_medium_or_heavy_rain`: usable heavy rain.
- `038_medium_or_heavy_rain`: usable heavy rain with wind.

Good light-rain examples:

- `003_light_rain`: usable light rain after the first ~5 seconds; frog call at
  start.
- `007_light_rain`: usable rain, perceived closer to medium.
- `019_light_rain`: very clean light rain.

Rain issues:

- Very low `precipitation_mm` values such as `0.01`, `0.03`, `0.04` are weak
  priors.
- Many light-rain candidates were actually wind, insects, or ambiguous
  ambience.
- Env rain prior should not be enough by itself for light rain.

### Wind

`medium_wind` performs better than `light_wind`.

Good / usable wind examples:

- `053_light_wind`: wind, perceived medium, with some rain.
- `057_light_wind`: usable light wind.
- `058_light_wind`: light to medium wind, some insect sound.
- `061_medium_wind`: usable medium wind.
- `065_medium_wind`: medium wind with strong rain component.
- `066_medium_wind`: usable medium wind.
- `067_medium_wind`: usable medium wind.
- `068_medium_wind`: usable medium wind, slight biological sound.
- `073_medium_wind`: usable medium wind but biological sound is noticeable.
- `075_medium_wind`: wind + rain.
- `076_medium_wind`: light to medium wind.

Wind issues:

- Many `light_wind` candidates actually sounded like rain.
- `wind_max_ms` can create false medium-wind candidates when the local window
  does not contain audible wind.
- Biological sound contamination is common.
- Wind selection needs audio confirmation, not env prior alone.

### Storm / Thunder

The first storm/thunder prior failed completely.

Observed labels were rain, wind, ambience, or biological contamination. No clip
was accepted as storm/thunder.

Conclusion:

```text
Do not admit thunder/storm candidates from env metadata alone.
Require CLAP/event confirmation or a thunder-specific detector.
```

## Policy Implication

The next retrieval/indexing pass should treat CLAP/audio embeddings as the main
weather classifier and env metadata only as a prior, filter, or tie-breaker.

Recommended weighting direction:

```text
final_score =
  0.65 * CLAP_weather_similarity
+ 0.20 * audio_quality_or_contamination_score
+ 0.15 * env_prior_score
```

Env metadata can propose candidate time ranges, but audio embedding must confirm
weather type and intensity before a clip enters the Layer D site-weather pool.

