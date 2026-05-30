# Site Weather CLAP Retrieval Audit 002

Source: balanced target-first CLAP retrieval batch generated after manual review
of `audit_001`.

Policy version:

```text
site_weather_clap_retrieval_v0.2
```

Local review page:

```text
debug/site_weather_clap_retrieval_v0_audit_002/listen.html
```

## Retrieval Change

`audit_001` used global weather ranking and produced too many wind clips.
`audit_002` uses balanced target-first retrieval:

- target `rain`: score rain prompts against contamination and other weather;
- target `wind`: score wind prompts against contamination and other weather;
- target `thunder`: score thunder prompts only inside storm/rain-supported
  candidates and gate strictly.

Env metadata remains a prior and candidate selector. CLAP/audio similarity is
still the main retrieval signal.

## Machine Summary

| Metric | Count |
|---|---:|
| total windows | 65 |
| target `rain` | 30 |
| target `wind` | 25 |
| target `thunder` | 10 |

Gate counts:

| Gate | Count |
|---|---:|
| `candidate` | 28 |
| `maybe_target_confused_with_other_weather` | 12 |
| `reject_target_outcompeted_by_other_weather` | 16 |
| `reject_contamination_dominant` | 2 |
| `reject_thunder_without_clear_audio_confirmation` | 7 |

CLAP top-label counts:

| CLAP label | Count |
|---|---:|
| `wind` | 55 |
| `rain` | 7 |
| `thunder` | 3 |

## Early Interpretation

The balanced manifest now includes enough rain and thunder-targeted candidates
for manual review, but CLAP top labels are still heavily biased toward wind.

The important audit question is no longer "which global CLAP label wins?" but:

```text
For each retrieval_target, is the target sound usable enough for Layer D?
```

Fields to inspect during manual review:

- `retrieval_target`
- `target_clap_score`
- `target_vs_other_weather_margin`
- `weather_margin`
- `contamination_label`
- `contamination_score`
- `gate_status`

## Next Calibration Questions

- For target `rain`, do positive examples exist even when `clap_weather_label`
  is `wind`?
- Are `maybe_target_confused_with_other_weather` clips usable as mixed weather,
  or should they be rejected?
- Do any target `thunder` candidates contain true thunder, or are they still
  wind/transient artifacts?
- Should target `rain` require higher target-vs-wind margin, or is human review
  finding usable light rain beneath wind?

## Manual Review Summary

Manual review was pasted from the browser audit page because page export was not
working reliably in the in-app browser.

| Retrieval target | Yes | Maybe | No | Main result |
|---|---:|---:|---:|---|
| `rain` | 6 | 3 | 21 | Usable rain exists, mostly near the top of the target-rain ranking. Weak rain is often wind, ambience, or biological sound. |
| `wind` | 9 | 13 | 3 | Wind has the best yield, but many clips are wind with biological sound or wind+rain. |
| `thunder` | 0 | 1-2 | 8-9 | No convincing thunder. Most thunder-targeted clips are wind, rain+wind, or heavy rain. |

Notes:

- Wind and rain often co-occur. The pool should support `rain+wind` as a mixed
  candidate class instead of forcing every clip into pure rain or pure wind.
- Biological sound is the main quality risk, especially for wind candidates.
- Site thunder should not be admitted automatically for the MVP. Use library
  fallback for thunder unless a future detector confirms real thunder events.
- Manual review should now move from full-batch listening to stratified spot
  checks of automatically selected candidate pools.

## Candidate Pool Direction

Use `script/dataset/build_site_weather_candidate_pool.py` to convert scored
retrieval manifests into a candidate pool manifest.

Initial output classes:

- `rain_primary`
- `rain_wind_mixed`
- `rain_backup_maybe`
- `wind_primary`
- `wind_with_bio_backup`
- `wind_backup_maybe`
- `wind_weak_backup`
- `storm_rain_wind_backup`
- `reject`

Only `*_primary` and `rain_wind_mixed` should be used by default for Layer D
site-first retrieval. Backup classes can be enabled only when primary coverage
is insufficient or for manual spot checks.
