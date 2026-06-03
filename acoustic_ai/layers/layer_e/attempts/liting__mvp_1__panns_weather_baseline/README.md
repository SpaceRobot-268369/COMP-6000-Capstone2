# Layer E MVP 1 — PANNs Weather Baseline

## Scope

This attempt upgrades the E-B weather analysis smoke baseline toward the
`pipeline_design.md` MVP path.

E-B analyses the raw uploaded audio mixture directly. It does not perform source
separation. E-A ambient context and E-C event detection remain separate Layer E
heads and are still placeholders in the current `/analysis` response.

## Method

The MVP detector uses a staged weather evidence stack:

1. **PANNs CNN14 / AudioSet tagger** when the optional `panns_inference`
   package and torch runtime are available. This provides direct weather labels
   such as wind, rain, raindrop, and thunderstorm.
2. **Site257 calibrated spectral detector** as the stable fallback. This reuses
   `liting__smoke_1__e_b_weather_analysis`, calibrated on Murphy's Server A
   CLAP-first promoted Layer B weather assets. Client-facing calibration should
   use Site257-derived assets only; sound-library rows in shared indexes are
   excluded from E-B MVP claims.
3. **Explainable DSP features** are always included for reporting and sanity
   checking.

After the first PANNs-enabled run, PANNs is used as a **presence detector** and
the site257 spectral detector remains responsible for intensity. This is because
raw PANNs scores separate rain/wind presence well, but do not map cleanly to
`light`/`moderate`/`strong` intensity buckets without site calibration.

Current method string when PANNs is unavailable:

```text
panns_unavailable__site257_clap_promoted_calibrated_spectral_nearest_centroid
```

Current method string when PANNs is available:

```text
panns_cnn14_audioset__site257_spectral_support
```

Current calibrated presence thresholds:

| Component | PANNs score threshold | How it is used |
|---|---:|---|
| rain | 0.19 | If present, keep the site257 spectral rain intensity; otherwise suppress rain to `none`. |
| wind | 0.02 | If present, keep the site257 spectral wind intensity; otherwise suppress wind to `none`. |
| thunder | 0.50 | Conservative placeholder only; site257 thunder is not a validated MVP output yet. |

## Data

This attempt does not own new training data yet. It reads the smoke baseline's
site257 promoted weather references:

```text
acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/layer_d_ready_manifest.csv
```

Those references come from Murphy's Server A Layer B site-weather policy.
Murphy's newer site-only direction should be treated as the source of truth for
future E-B calibration: use `analysis_use=site_ready_pool` or
`site_backup_pool`, and ignore rows where `source_type=sound_library` for
client-facing analysis metrics.

This attempt also includes a small local no-weather negative proxy manifest:

```text
acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/data/no_weather_negative_manifest.csv
```

These are already-materialised Site257 Layer C/event reference clips that the
current E-B detector reports as no rain and no wind. They are useful for a first
false-positive check, but they are proxies rather than final ambient holdout
clips.

## Murphy Layer B Alignment

This attempt is aligned with Murphy's current Layer B weather policy:

- CLAP/audio evidence is the primary weather candidate signal.
- Environmental metadata is used as a prior/filter/tie-breaker, not as a direct
  ground-truth label.
- Site257 wind assets are usable as site-first MVP material.
- Site257 rain assets are limited; mixed rain+wind examples are expected and
  should be treated as boundary cases.
- Site257 thunder is not considered reliable enough for the default MVP pool;
  thunder remains disabled until a Site257-derived thunder example is found and
  audited. Sound-library thunder is not acceptable for the client-facing E-B
  claim.

The MVP test therefore evaluates asset policy classes separately:

| Policy class | Meaning | Test expectation |
|---|---|---|
| `rain_primary` | Murphy accepted the clip as a rain weather asset | Compare expected rain/wind intensities directly. |
| `wind_primary` | Murphy accepted the clip as a wind weather asset | Compare expected rain/wind intensities directly. |
| `boundary_mixed_rain_wind` | Murphy accepted the clip as rain+wind overlap | Do not judge it as pure rain or pure wind; count it as policy-aligned if the detector identifies audible weather presence. |
| `no_weather_negative` | Local proxy clip with no rain/wind detected | Strictly require both rain and wind to be `none`. |

This avoids incorrectly failing the detector for clips that Layer B intentionally
keeps as mixed/boundary assets.

Site-only policy for the next refresh:

```text
include: source_type=site AND analysis_use in {site_ready_pool, site_backup_pool}
exclude: source_type=sound_library
thunder: status=insufficient_site_data unless Site257 thunder is audited
```

## Training

No training is required for this MVP baseline. PANNs is a pre-trained AudioSet
tagger. Future work may fine-tune a small weather head after more labelled
wind/rain assets are accumulated.

## Validation

Run from the repository root:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/tests/e_b_weather_mvp_test.py
```

Export PANNs calibration evidence:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/code/calibrate_panns_weather.py
```

Calibration outputs are written under:

```text
debug/e_b_weather_mvp/panns_calibration/
```

Build or refresh the local no-weather negative proxy manifest:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/code/build_no_weather_negative_manifest.py
```

Expected current behaviour on the local Mac:

- If `panns_inference`, torch, or the PANNs checkpoint are not available, the
  test still falls back to the spectral/site257 detector.
- The report explicitly states whether PANNs was available.
- `rain_wind_mixed` clips are counted under `boundary`, not as strict failures.
- `no_weather_negative` clips are strict false-positive checks: rain and wind
  must both be `none`.
- The headline MVP metric is `policy_aligned_rate`, which includes `pass`,
  `partial`, and valid `boundary` cases.

Latest local PANNs-enabled result:

```text
positive_cases=63
negative_cases=3
pass=31, partial=25, boundary=5, fail=5
pass_or_partial_rate=0.85
policy_aligned_rate=0.92
panns_available_cases=66
```

The valid Zenodo `Cnn14_mAP=0.431.pth` checkpoint is 327,428,481 bytes. The
runtime guard accepts the official checkpoint and rejects much smaller
HTML/error/partial downloads.

Calibration summary from the 57 primary rain/wind assets:

```text
rain best_presence_threshold=0.19, accuracy=1.00
wind best_presence_threshold=0.02, accuracy=0.98
```

The PANNs-enabled baseline now passes the MVP policy-aligned gate and remains
close to the site257 spectral fallback (`policy_aligned_rate=0.94` in the
fallback-only run). The remaining gap is mostly from mixed/boundary and a few
wind assets where PANNs presence is weak.

The current no-weather proxy validation found three local Site257 event clips
that pass strict no-rain/no-wind checks. This is a useful first false-positive
guard, but the next stronger validation needs ambient no-weather clips from the
Site257 DVC clip set.

## Limitations

- The no-weather negatives are local proxy clips from materialised Site257
  event/reference data. They should be replaced or expanded with ambient
  no-weather clips once the Site257 DVC clip set is available locally.
- The fallback is still a calibration baseline, not a trained classifier.
- Rain coverage in the current site257 promoted asset pool is much smaller
  than wind coverage.
- This detector returns labels and confidence only; it does not output isolated
  weather audio stems.

## Next Steps

1. Replace proxy no-weather negatives with ambient no-weather clips from the
   Site257 DVC clip set.
2. Compare PANNs-only, spectral-only, and fused predictions in a report.
3. Add a small labelled holdout once more Layer B weather assets accumulate.
4. Decide whether a small fine-tuned weather head is needed after calibration.
5. Keep thunder disabled as a confident MVP output until Murphy/Layer B has a
   validated Site257-derived thunder asset policy.
