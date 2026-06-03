# Implementation Plan - E-B Weather Analysis MVP

| | |
|---|---|
| **Attempt ID** | `liting__mvp_1__panns_weather_baseline` |
| **Layer / role** | `layer_e` - Analysis · E-B weather detector |
| **Stage** | `mvp_1` |
| **Primary model** | PANNs CNN14 AudioSet tagger |
| **Support signal** | Site257-only spectral calibration using Murphy Layer B weather assets |
| **Scope** | Upload-based weather analysis head + offline evaluation |
| **Author / date** | liting · 2026-06-03 |
| **Branch** | `feat/liting/e-b-weather-analysis-main-sync` |

> This is the E-B plan counterpart to Lucas's attempt planning style: define
> the attempt path, state the hypothesis, name sibling/fallback attempts, and
> set measurable pass bars before adding more implementation.

---

## 1. Attempt Roadmap / Bake-Off

E-B estimates audible weather directly from an uploaded raw mixture. The output
contract is:

```json
{
  "wind": {"intensity": "none | light | moderate | strong", "confidence": 0.0},
  "rain": {"intensity": "none | light | moderate | heavy", "confidence": 0.0},
  "thunder": {
    "intensity": "none",
    "confidence": 0.0,
    "status": "insufficient_site_data"
  }
}
```

This roadmap keeps each method as a separate attempt rather than rewriting one
folder repeatedly:

| Attempt | Stage | Method | Status / decision |
|---|---|---|---|
| `liting__smoke_1__e_b_weather_analysis` | `smoke_1` | Site257 spectral nearest-centroid calibrated on Murphy's CLAP-promoted weather assets | Done. Proves the pipeline and data handoff run end-to-end, but it is not a trained model. |
| `liting__mvp_1__panns_weather_baseline` | `mvp_1` | PANNs CNN14 AudioSet weather logits + Site257 spectral support | Current candidate. Aligns with `pipeline_design.md` MVP path. |
| `liting__mvp_2__calibrated_weather_head` | `mvp_2` | Frozen PANNs embedding/logits + small calibrated head trained on curated Site257 weather labels | Future only if mvp_1 is stable but accuracy/confidence is not enough. |
| `liting__mvp_3__site257_clap_weather_expansion` | `mvp_3` | Build an independent Site257 CLAP weather candidate database / validation holdout | Future data-expansion path; should not replace Murphy's pool until audited. |
| `liting__mvp_4__clap_weather_prompts` | `mvp_4` | LAION-CLAP zero-shot prompt scoring for wind/rain, with thunder disabled unless site evidence exists | Optional comparison if PANNs underperforms or CLAP is already loaded for E-A. |

Promotion rule:

- If `mvp_1` gives stable results and explainable evidence, keep it as the E-B
  MVP.
- If PANNs is available but confidence/intensity buckets are poorly calibrated,
  move to `mvp_2`.
- If Murphy's pool is too small or needs an independent validation set, run
  `mvp_3` as a Site257-only CLAP expansion attempt.
- If PANNs has a domain gap for ecoacoustic weather, run `mvp_4` as a bake-off
  rather than patching the same attempt endlessly.

---

## 2. Purpose / Hypothesis

**Purpose:** estimate audible wind and rain intensity from an uploaded
ecoacoustic audio clip, without separating stems. Thunder remains a reserved
field only; it is not a validated MVP class until site-derived thunder examples
exist.

**Hypothesis:** wind and rain have stable AudioSet-level acoustic signatures
that PANNs CNN14 can detect from a raw mixture:

- rain: dense broadband/high-frequency texture, raindrop impulses
- wind: broadband low-frequency energy, vegetation rustle, slow RMS modulation
- thunder: low-frequency burst energy and high onset strength, but this is
  disabled for MVP unless the evidence comes from Site257

Site257 spectral calibration is retained as an explainability and fallback
channel because the current demo data comes from Murphy's Layer B Site257
weather asset policy.

This means E-B is not trying to infer meteorological truth such as exact wind
speed or rainfall in mm. It estimates **audible weather contribution** in the
uploaded sound.

---

## 3. Data Inputs

| Use | Path | Notes |
|---|---|---|
| Positive weather assets | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/layer_d_ready_manifest.csv` | Murphy Layer B / Site257 CLAP-promoted weather candidates copied into this E-B smoke attempt for local calibration. |
| Materialised WAV assets | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/` | 63 local 15 s Site257 weather clips when materialised. |
| No-weather proxy negatives | `acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/data/no_weather_negative_manifest.csv` | Small proxy negative set from local event/reference clips; useful for sanity only, not final validation. |
| Latest Murphy site-only candidate source | `origin/feat/murphy/ayer-b-only-site` -> `acoustic_ai/layers/layer_b/attempts/lucas__smoke_1__curated_assets/data/weather/asset_index.csv` | Remote branch currently shows 153 rows: 113 site rows and 40 sound-library rows. Library rows are marked disabled/rejected for site-only use; E-B should consume only `analysis_use=site_ready_pool` or `site_backup_pool`. |
| Shared weather asset index | `acoustic_ai/layers/layer_b/attempts/lucas__smoke_1__curated_assets/data/weather/asset_index.csv` | Legacy/shared Layer B weather labels if available locally. Must filter out `source_type=sound_library` for E-B MVP claims. |
| PANNs checkpoint | `/private/tmp/panns_home/panns_data/Cnn14_mAP=0.431.pth` | Local machine cache; not committed to git. |

Known limitation:

- The 63 positive weather clips are not a large supervised dataset. They are
  enough for smoke/MVP calibration, but not enough to claim a robust trained
  classifier.
- Client-facing E-B validation should not use sound-library rain/wind/thunder.
  Library rows may remain in historical asset indexes, but they are excluded
  from E-B calibration/evaluation unless explicitly labelled as non-client demo
  fallback.
- Thunder is not validated in the current site-derived pool. The detector may
  keep a thunder field for schema stability, but it should report
  `status=insufficient_site_data` or conservative `none`.

---

## 4. Method

### 4.1 PANNs primary signal

Use PANNs CNN14 as a frozen pre-trained AudioSet tagger. Weather evidence is
read from AudioSet labels:

| E-B component | PANNs labels |
|---|---|
| rain | `Rain`, `Raindrop`, `Rain on surface` |
| wind | `Wind`, `Rustling leaves`, `Wind noise` |
| thunder | Reserved only. PANNs labels exist (`Thunderstorm`, `Thunder`), but MVP output is disabled until a Site257 thunder policy exists. |

PANNs output probabilities are converted into conservative intensity buckets.
This is zero-shot / pre-trained inference, not model training.

### 4.2 Site257 spectral support

The fallback/support detector extracts interpretable audio features:

- low-band energy ratio: wind / thunder evidence
- 2-8 kHz band ratio: rain texture evidence
- spectral flatness: noise-like rain/wind texture
- onset strength: raindrop/thunder impulse evidence
- RMS modulation: wind gust / weather movement evidence

Those features are calibrated against Site257-derived promoted weather assets
with a nearest-centroid classifier. Sound-library weather assets are excluded
from this calibration because the client requested site-derived audio rather
than external library material.

### 4.3 Fusion

Current `mvp_1` rule:

1. Run Site257 spectral detector for a stable baseline.
2. Run PANNs if dependency + checkpoint are available.
3. Use PANNs as the primary model signal.
4. Keep spectral output in `supporting_detector` for transparency and fallback.
5. Report per-component intensity and confidence for wind/rain.
6. Keep thunder as a reserved field with `insufficient_site_data` unless a
   site-derived thunder example is audited and accepted.

---

## 5. Evaluation Design

### 5.1 Smoke baseline

Command:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/tests/e_b_weather_smoke_test.py
```

Current result:

```text
case_count=63
pass=28
partial=25
fail=10
pass_or_partial_rate=0.84
```

Interpretation:

- This validates pipeline/data handoff.
- It does not validate a trained model.
- Failures are mostly mixed rain+wind or wind intensity boundary cases.

### 5.2 MVP baseline

Command:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/tests/e_b_weather_mvp_test.py
```

Current result:

```text
positive_cases=63
negative_cases=3
panns_available_cases=66
pass=31
partial=25
boundary=5
fail=5
policy_aligned_rate=0.92
```

Interpretation:

- PANNs is available and participates in the current run.
- The policy-aligned rate is high enough for an MVP candidate.
- More negative examples and better thunder validation are still required.
- Current positive set is still based on the older 63 local clips. It should be
  refreshed against Murphy's latest site-only pool once the updated data is
  merged or copied into the Liting attempt.

### 5.3 Success bar for `mvp_1`

`mvp_1` remains the preferred candidate if:

- PANNs is available in the demo/server environment.
- `policy_aligned_rate >= 0.90`.
- Rain primary examples are detected as non-none rain with high confidence.
- Wind primary examples are detected as non-none wind with acceptable boundary
  tolerance.
- No-weather examples mostly stay `rain=none`, `wind=none`.
- The output includes per-component confidence for wind and rain.
- Thunder is either omitted from the claim or marked as insufficient site data.

---

## 6. Phase 2 / Next Attempt Triggers

Move to `liting__mvp_2__calibrated_weather_head` only if one of these happens:

- PANNs logits are present but intensity buckets are unstable.
- False positives on no-weather clips are too high after adding a real ambient
  negative set.
- Confidence values do not correlate with correctness.
- Mixed rain+wind boundary clips keep failing in a way the demo cannot explain.
- Murphy's latest site-only pool changes the calibration labels enough that
  the 63-clip baseline is no longer representative.

`mvp_2` should not fine-tune the full PANNs model. It should train a small
calibration head or threshold model over frozen PANNs logits + DSP features.

Possible `mvp_2` inputs:

```text
[PANNs rain/wind/thunder logits,
 low_band_ratio,
 rain_band_ratio,
 spectral_flatness,
 onset_strength,
 rms_modulation]
```

Possible labels:

```text
rain_intensity, wind_intensity, thunder_intensity
```

---

## 7. Risks / Open Questions

- **Thunder data is weak.** Murphy's current Site257 promoted pool does not treat
  thunder as a validated default path. Thunder should remain conservative until
  real thunder examples exist.
- **Sound-library exclusion.** Client-facing E-B calibration and evaluation
  must use Site257-derived assets only. If a shared asset index includes
  `source_type=sound_library`, those rows are ignored for E-B MVP claims.
- **Murphy pool alignment.** Murphy owns the Layer B weather pool. E-B should
  align with his latest site-only policy before building a separate database.
- **No-weather negatives are proxy only.** The current negatives are not a final
  ambient/no-weather holdout. We need a cleaner ambient negative pool before
  claiming robustness.
- **PANNs domain gap.** PANNs is trained on general AudioSet, not specifically
  Bowra ecoacoustics. Site-specific calibration is required.
- **Long uploads.** Current detector analyzes the first 30 seconds by default.
  Demo uploads should be 10-30 s WAV files until long-window aggregation is
  implemented.
- **Confidence is not scientific certainty.** It is model/detector confidence,
  not ground-truth weather probability.

---

## 8. Definition of Done

This MVP attempt is done when:

- `liting__mvp_1__panns_weather_baseline` is registered as `head: weather` in
  `acoustic_ai/registry.yaml`.
- The attempt supports registry-based upload analysis:
  `POST /layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/analyze`.
- `e_b_weather_mvp_test.py` passes with PANNs available.
- Evaluation uses only site-derived weather assets for client-facing metrics.
- README documents:
  - model used
  - whether training happened
  - data source
  - upload duration assumptions
  - success/failure interpretation
- The report exposes per-component intensity + confidence for wind, rain, and
  a conservative thunder status.

---

## 9. Client-Facing Explanation

This MVP uses a pre-trained environmental audio tagger, PANNs CNN14, to detect
weather-like audio events in an uploaded soundscape. We combine the tagger's
AudioSet weather evidence with site-specific spectral calibration from Site257
weather assets prepared for Layer B. The result is an explainable E-B analysis
head that reports audible wind and rain intensity with confidence.

At this stage, the model is not trained from scratch. It is a pre-trained model
plus calibration baseline. The next step is to add a larger labelled validation
set from the same Site257 source and, if necessary, train a small calibration
head on top of frozen PANNs outputs. Thunder is intentionally not claimed as a
validated class until site-derived thunder audio is found and audited.
