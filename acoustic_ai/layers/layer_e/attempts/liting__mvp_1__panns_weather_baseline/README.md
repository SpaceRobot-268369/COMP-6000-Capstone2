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
   CLAP-first promoted Layer B weather assets.
3. **Explainable DSP features** are always included for reporting and sanity
   checking.

Current method string when PANNs is unavailable:

```text
panns_unavailable__site257_clap_promoted_calibrated_spectral_nearest_centroid
```

Current method string when PANNs is available:

```text
panns_cnn14_audioset__site257_spectral_support
```

## Data

This attempt does not own new training data yet. It reads the smoke baseline's
site257 promoted weather references:

```text
acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/layer_d_ready_manifest.csv
```

Those references come from Murphy's Server A Layer B site-weather policy.

## Murphy Layer B Alignment

This attempt is aligned with Murphy's current Layer B weather policy:

- CLAP/audio evidence is the primary weather candidate signal.
- Environmental metadata is used as a prior/filter/tie-breaker, not as a direct
  ground-truth label.
- Site257 wind assets are usable as site-first MVP material.
- Site257 rain assets are limited; mixed rain+wind examples are expected and
  should be treated as boundary cases.
- Site257 thunder is not considered reliable enough for the default MVP pool;
  thunder remains a future/library-fallback path.

The MVP test therefore evaluates asset policy classes separately:

| Policy class | Meaning | Test expectation |
|---|---|---|
| `rain_primary` | Murphy accepted the clip as a rain weather asset | Compare expected rain/wind intensities directly. |
| `wind_primary` | Murphy accepted the clip as a wind weather asset | Compare expected rain/wind intensities directly. |
| `boundary_mixed_rain_wind` | Murphy accepted the clip as rain+wind overlap | Do not judge it as pure rain or pure wind; count it as policy-aligned if the detector identifies audible weather presence. |

This avoids incorrectly failing the detector for clips that Layer B intentionally
keeps as mixed/boundary assets.

## Training

No training is required for this MVP baseline. PANNs is a pre-trained AudioSet
tagger. Future work may fine-tune a small weather head after more labelled
wind/rain assets are accumulated.

## Validation

Run from the repository root:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/tests/e_b_weather_mvp_test.py
```

Expected current behaviour on the local Mac:

- If `panns_inference`, torch, or the PANNs checkpoint are not available, the
  test still falls back to the spectral/site257 detector.
- The report explicitly states whether PANNs was available.
- `rain_wind_mixed` clips are counted under `boundary`, not as strict failures.
- The headline MVP metric is `policy_aligned_rate`, which includes `pass`,
  `partial`, and valid `boundary` cases.

Latest local PANNs-enabled result:

```text
pass=15, partial=31, boundary=6, fail=11
pass_or_partial_rate=0.73
policy_aligned_rate=0.83
panns_available_cases=63
```

The valid Zenodo `Cnn14_mAP=0.431.pth` checkpoint is 327,428,481 bytes. The
runtime guard accepts the official checkpoint and rejects much smaller
HTML/error/partial downloads.

The PANNs-enabled baseline passes the MVP policy-aligned gate, but it performs
worse than the site257 spectral fallback (`policy_aligned_rate=0.94` in the
fallback-only run). This is expected for an uncalibrated AudioSet zero-shot
model and is the reason the next step is threshold/site calibration rather than
server training.

## Limitations

- PANNs is optional in the current local environment; the MVP branch is wired
  for it but may fall back until dependencies and weights are installed.
- The fallback is still a calibration baseline, not a trained classifier.
- Rain coverage in the current site257 promoted asset pool is much smaller
  than wind coverage.
- This detector returns labels and confidence only; it does not output isolated
  weather audio stems.

## Next Steps

1. Record and inspect PANNs logits for the 63 site257 promoted clips.
2. Calibrate intensity bucket thresholds against Murphy's accepted labels.
3. Add clean negative/no-weather site examples.
4. Compare PANNs-only, spectral-only, and fused predictions in a report.
5. Add a small labelled holdout once more Layer B weather assets accumulate.
6. Decide whether a small fine-tuned weather head is needed after calibration.
