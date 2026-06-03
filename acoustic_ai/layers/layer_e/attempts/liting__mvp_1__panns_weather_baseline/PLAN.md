# Implementation Plan - E-B Weather Analysis MVP

| | |
|---|---|
| **Attempt ID** | `liting__mvp_1__panns_weather_baseline` |
| **Layer / role** | `layer_e` - Analysis · E-B weather detector |
| **Stage** | `mvp_1` |
| **Primary model** | PANNs CNN14 AudioSet tagger |
| **Support signal** | E-B-owned Site257 spectral calibration and site-wide weather labels |
| **Scope** | Upload-based weather-layer analysis for arbitrary Site257 audio clips |
| **Author / date** | liting · 2026-06-03 |
| **Branch** | `feat/liting/e-b-weather-analysis-main-sync` |

> This plan is prepared for team review before expanding E-B beyond the current
> smoke baseline. It defines the attempt path, hypothesis, fallback attempts,
> data policy, server/runtime expectations, and measurable pass bars.

---

## 1. Target Scope and Output Contract

E-B estimates audible weather directly from an uploaded raw mixture. The target
feature is **not** a pool-only classifier for curated weather assets.
The target feature is:

```text
Given any Site257 audio clip uploaded by the user, estimate the audible weather
layer in that clip: wind intensity, rain intensity, and thunder status, with
confidence and limitations.
```

This means the model must handle three kinds of Site257 inputs:

| Input type | Expected behaviour |
|---|---|
| Weather-positive clips | Detect the audible weather layer and return a non-none wind/rain label when supported by audio evidence. |
| Mixed ecoacoustic clips | Report weather if present, while tolerating birds/insects/background ambience in the same mixture. |
| No-weather / ambient clips | Return `wind=none`, `rain=none`, and avoid false-positive weather claims. |

The output contract is:

```json
{
  "wind": {
    "summary": {
      "intensity": 0.0,
      "variability": 0.0,
      "coverage": 0.0,
      "label": "none | light | moderate | strong",
      "confidence": 0.0
    }
  },
  "rain": {
    "summary": {
      "intensity": 0.0,
      "variability": 0.0,
      "coverage": 0.0,
      "label": "none | light | moderate | heavy",
      "confidence": 0.0
    }
  },
  "thunder": {
    "intensity": 0.0,
    "event_count": 0,
    "events": [],
    "mean_interval_s": null,
    "label": "none",
    "confidence": 0.0,
    "status": "insufficient_site_data"
  }
}
```

This follows the current main-branch Layer E synthesis policy:

- E-B owns weather as an **authoritative observation**.
- E-B does not infer season or diel; the aggregator fuses latent context.
- `summary.intensity` is the primary continuous value.
- `summary.label` is the human-readable bucket derived from the continuous
  value.
- `segments` are optional and are not required for MVP-1.

Thunder remains in the schema for compatibility, but the MVP should not claim
thunder detection until Site257-derived thunder examples are found, audited,
and represented in the holdout set.

---

## 2. Attempt Roadmap / Bake-Off

This roadmap keeps each method as a separate attempt rather than rewriting one
folder repeatedly. The early weather pool is used to bootstrap validation, but
the promoted MVP must evaluate on a site-wide Site257 input set, not just the
weather-positive pool.

| Attempt | Stage | Method | Status / decision |
|---|---|---|---|
| `liting__smoke_1__e_b_weather_analysis` | `smoke_1` | Site257 spectral nearest-centroid calibrated on an initial CLAP-promoted weather pool | Done. Proves the pipeline and data handoff run end-to-end, but its scope is too narrow for the final feature. |
| `liting__mvp_1__panns_weather_baseline` | `mvp_1` | PANNs CNN14 AudioSet weather logits + Site257 spectral support | Current baseline. Must be upgraded from pool validation to site-wide Site257 validation on Server B. |
| `liting__mvp_2__calibrated_weather_head` | `mvp_2` | Frozen PANNs embedding/logits + DSP features + small calibrated head trained on E-B-owned Site257 weather/no-weather labels | Target MVP if mvp_1 confidence/intensity calibration is not enough. |
| `liting__mvp_3__site257_clap_weather_expansion` | `mvp_3` | Build an independent Site257 CLAP weather candidate database and site-wide validation holdout | Data-expansion path for building the E-B-owned training/evaluation set from Site257. |
| `liting__mvp_4__clap_weather_prompts` | `mvp_4` | LAION-CLAP zero-shot prompt scoring for wind/rain, with thunder disabled unless site evidence exists | Optional comparison if PANNs underperforms or CLAP is already loaded for E-A. |

Promotion rule:

- If `mvp_1` gives stable results on a site-wide Site257 validation split, keep
  it as the E-B MVP.
- If `mvp_1` only works on the curated weather pool, move to `mvp_2` or `mvp_3`;
  the feature is not complete.
- If PANNs is available but confidence/intensity buckets are poorly calibrated,
  move to `mvp_2`.
- If the current labelled Site257 set is too small or lacks no-weather coverage,
  run `mvp_3` as an E-B-owned Site257-only CLAP expansion attempt.
- If PANNs has a domain gap for ecoacoustic weather, run `mvp_4` as a bake-off
  rather than patching the same attempt endlessly.

---

## 3. Purpose / Hypothesis

**Purpose:** estimate audible wind and rain intensity from an uploaded
Site257 ecoacoustic audio clip, without separating stems. The model should
generalise across normal Site257 uploads, including clips with birds, insects,
quiet ambience, wind, rain, and mixed conditions. Thunder remains a reserved
field only; it is not a validated MVP class until site-derived thunder examples
exist.

**Hypothesis:** wind and rain have stable AudioSet-level acoustic signatures
that PANNs CNN14 can detect from a raw mixture:

- rain: dense broadband/high-frequency texture, raindrop impulses
- wind: broadband low-frequency energy, vegetation rustle, slow RMS modulation
- thunder: low-frequency burst energy and high onset strength, but this is
  disabled for MVP unless the evidence comes from Site257

Site257 spectral calibration is retained as an explainability and fallback
channel because the MVP must stay grounded in the client's requested site data
rather than external sound libraries.

This means E-B is not trying to infer meteorological truth such as exact wind
speed or rainfall in mm. It estimates **audible weather contribution** in the
uploaded sound. A no-weather result is a valid output and must be learned from
site-wide negative examples, not guessed from the absence of a curated weather
label.

---

## 4. Data Strategy

| Use | Path | Notes |
|---|---|---|
| Site-wide input universe | `resources/site_257_bowra-dry-a/` clips / manifests | Target feature scope. E-B should evaluate on ordinary Site257 clips, not only weather-positive clips. |
| Positive weather seed assets | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/layer_d_ready_manifest.csv` | Initial Site257 CLAP-promoted weather candidates used for the runnable smoke/MVP baseline and validation seeding. |
| Materialised WAV assets | `acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/` | 63 local 15 s Site257 weather clips when materialised. |
| No-weather proxy negatives | `acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/data/no_weather_negative_manifest.csv` | Small proxy negative set from local event/reference clips; useful for sanity only, not final validation. |
| Team Site257 weather policy reference | Team site-only weather index | Reference only. Current shared index shows 153 rows: 113 site rows and 40 sound-library rows. E-B consumes only `source_type=site` with `analysis_use=site_ready_pool` or `site_backup_pool`. |
| Liting E-B site-only policy snapshot | `acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/data/site257_weather_policy_snapshot.csv` | Local E-B planning snapshot of the 113 Site257 rows: 105 ready + 8 backup, excluding all sound-library rows. Not all referenced WAV files are materialised locally yet. |
| E-B-owned training manifest | `data/e_b_site257_weather_training_manifest.csv` | Planned next output. This should be built by materialising/auditing Site257 clips for E-B, with train/validation/holdout split and no sound-library rows. |
| E-B manifest template | `data/e_b_site257_weather_manifest_template.csv` | Header-only CSV template for the Server B manifest build. |
| PANNs checkpoint | `/private/tmp/panns_home/panns_data/Cnn14_mAP=0.431.pth` | Local machine cache; not committed to git. |

### 4.1 Role of the audited weather pool

The audited weather pool is useful, but it is not the feature scope.

| Pool role | Use it for? | Why |
|---|---|---|
| Calibration seed | Yes | Gives labelled positive wind/rain examples. |
| Validation seed | Yes | Lets us measure whether obvious weather cases are detected. |
| Final feature scope | No | The user can upload any Site257 clip, including no-weather and mixed clips. |
| Only training/evaluation data | No | This would overfit the detector to weather-positive examples and hide false positives. |

### 4.2 Required E-B-owned dataset

The next Server B run should build:

```text
data/e_b_site257_weather_training_manifest.csv
```

Minimum manifest columns:

```text
clip_id, audio_path, source_site_id, source_recording_id, start_s, end_s,
split, label_source, audit_status,
rain_intensity, wind_intensity, thunder_status,
mixed_weather, notes
```

Required split design:

| Split | Content | Purpose |
|---|---|---|
| train/calibration | Audited positives + audited no-weather negatives | Fit thresholds or small head. |
| validation | Held-out positives, no-weather clips, mixed clips | Tune threshold/confidence decisions. |
| holdout | Site-wide random clips not used in fitting | Estimate real upload behaviour and false positives. |

Known limitations:

- The 63 currently materialised positive weather clips are not a large supervised dataset. They are
  enough for smoke/MVP calibration, but not enough to claim a robust trained
  classifier.
- The current site-only policy snapshot expands the candidate policy pool to
  113 Site257 rows, but those rows still need materialised WAV access and E-B
  audit labels before they can replace the current 63-clip executable test set.
- Client-facing E-B validation should not use sound-library rain/wind/thunder.
  Library rows may remain in historical asset indexes, but they are excluded
  from E-B calibration/evaluation unless explicitly labelled as non-client demo
  fallback.
- Thunder is not validated in the current site-derived pool. The detector may
  keep a thunder field for schema stability, but it should report
  `status=insufficient_site_data` or conservative `none`.
- The shared site-only policy reference is used only to avoid contradicting the
  current team weather taxonomy. The training/evaluation claim should come from
  the E-B-owned manifest once its Site257 clips are materialised and audited.

---

## 5. Model / Training Target

### 5.1 PANNs primary signal

Use PANNs CNN14 as a frozen pre-trained AudioSet tagger. Weather evidence is
read from AudioSet labels:

| E-B component | PANNs labels |
|---|---|
| rain | `Rain`, `Raindrop`, `Rain on surface` |
| wind | `Wind`, `Rustling leaves`, `Wind noise` |
| thunder | Reserved only. PANNs labels exist (`Thunderstorm`, `Thunder`), but MVP output is disabled until a Site257 thunder policy exists. |

PANNs output probabilities are converted into conservative intensity buckets.
For the final MVP, this should be calibrated on E-B-owned Site257 train/validation
splits, not only on the weather-positive pool.

### 5.2 Site257 spectral support

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

### 5.3 Fusion

Current `mvp_1` rule:

1. Run Site257 spectral detector for a stable baseline.
2. Run PANNs if dependency + checkpoint are available.
3. Use PANNs as the primary model signal.
4. Keep spectral output in `supporting_detector` for transparency and fallback.
5. Report per-component intensity and confidence for wind/rain.
6. Keep thunder as a reserved field with `insufficient_site_data` unless a
   site-derived thunder example is audited and accepted.

### 5.4 Training target

The model should be trained/calibrated to this standard:

```text
Given arbitrary Site257 audio, classify audible weather layer presence and
intensity while controlling false positives on ordinary no-weather site clips.
```

Minimum MVP model target:

| Component | Target |
|---|---|
| Feature extractor | Frozen PANNs CNN14 logits + explainable DSP features. |
| Trainable/calibrated part | Threshold table, logistic regression, or small MLP head over frozen features. |
| Training data | E-B-owned Site257 manifest with positives, mixed clips, and no-weather negatives. |
| Validation data | Held-out Site257 clips, including random site-wide uploads. |
| Output | Wind/rain summary objects with continuous intensity, label, coverage, variability, and confidence; thunder disabled unless validated. |

Do **not** fine-tune the full PANNs backbone for the first MVP. The training
objective is calibration and small-head learning over frozen features.

---

## 6. Server / Training Runtime Plan

### 6.1 Team policy: formal MVP attempts run on Server B

Every E-B MVP attempt should have a Server B run, even when the model backbone
is frozen. The local Mac can be used for code edits, dry-runs, and quick CLI
smoke checks, but the official MVP evidence should come from a Server B
training/evaluation job.

E-B should be treated as an independent Layer E-B workstream. Shared Layer B
weather assets are useful for temporary policy alignment, but the MVP training
and validation path should build an E-B-owned Site257 weather dataset from the
same site source, with its own manifest, labels, audit notes, and holdout split.

### 6.2 Current `mvp_1`: Server B calibration/evaluation job

`liting__mvp_1__panns_weather_baseline` does **not** fine-tune the full PANNs
model. It runs a Server B job that materialises Site257 candidate clips,
extracts PANNs/DSP evidence, builds the E-B-owned manifest, fits or confirms
thresholds, and writes an evaluation report.

Model path:

- frozen PANNs CNN14 AudioSet inference
- Site257 spectral calibration / thresholding
- optional threshold fitting over E-B-owned Site257 weather labels
- Server B evaluation against materialised Site257 weather assets

Expected runtime:

| Task | Machine | Expected time | Notes |
|---|---|---:|---|
| Server B setup check | Server B | ~5-15 min | Confirm branch, venv, PANNs checkpoint/cache, DVC/S3 access. |
| Materialise Site257 E-B candidate pool | Server B | ~15-60 min | Include positives, mixed clips, no-weather negatives, and random site-wide holdout clips. |
| Build E-B-owned manifest + split | Server B | ~10-30 min | Create train/validation/holdout rows and audit status fields. |
| Extract PANNs + DSP features | Server B CPU/GPU | ~5-20 min | 113 short clips; GPU helps but is not mandatory. |
| Fit/confirm thresholds or small head | Server B | < 5 min | Calibration/head training, not full PANNs training. |
| Run evaluation + write report | Server B | ~3-10 min | Produces report JSON/markdown for team review. |
| Upload/API demo check | Server A API + Server B artifacts | ~5-15 min | Confirms frontend/API output format after server-side attempt succeeds. |

Conclusion:

```text
For mvp_1, the formal Server B run is required for team review.
The job is calibration/evaluation training over frozen features, not full
PANNs fine-tuning.
Expected total: ~30-90 minutes including data materialisation.
```

Operational checklist:

```text
acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/SERVER_B_RUNBOOK.md
```

### 6.3 Future `mvp_2`: small calibration-head training

Move to `liting__mvp_2__calibrated_weather_head` only if `mvp_1` needs better
confidence calibration or fewer boundary failures. This stage should still keep
PANNs frozen and train only a small head over PANNs logits + DSP features.

Expected runtime:

| Task | Machine | Expected time | Notes |
|---|---|---:|---|
| Build feature table from 100-300 labelled clips | Server B or local | ~5-20 min | Mostly audio loading + PANNs/DSP feature extraction. |
| Train small calibration head | Server B preferred, local acceptable | < 5 min | Logistic regression / small MLP; no full PANNs fine-tune. |
| Evaluate holdout + write report | Server B or local | ~5-15 min | Includes confusion summary, calibration report, and threshold table. |

Expected total:

```text
~30-90 minutes including data materialisation and evaluation.
Actual model training should be only a few minutes.
```

### 6.4 Future `mvp_3`: Site257 CLAP data expansion

`liting__mvp_3__site257_clap_weather_expansion` is a data-expansion path, not
the first MVP training path. It should be used only if the current Site257 pool
is too small or too imbalanced for E-B validation.

Expected runtime:

| Task | Machine | Expected time | Notes |
|---|---|---:|---|
| Pull/download additional Site257 clips | Server A/B or DVC-enabled local | ~30 min-2 hr | Depends on DVC/S3 and clip count. |
| Run CLAP candidate scoring | Server B GPU preferred | ~30 min-2 hr | Needed only for larger candidate search. |
| Manual audit / label cleanup | human review | ~1-3 hr | This is likely the slowest part. |

Conclusion:

```text
MVP should not wait for mvp_3 unless the 113-row Site257 pool cannot support
reliable calibration/evaluation.
```

---

## 7. Evaluation Design

### 7.1 Smoke baseline

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

### 7.2 Current local MVP baseline

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
- The policy-aligned rate is high enough for a baseline on the current
  materialised weather-positive test set.
- More negative examples and better thunder validation are still required.
- Current positive set is still based on the older 63 local materialised clips.
  The Liting attempt now includes a 113-row Site257 policy snapshot for
  alignment, but the executable test should be refreshed only after those
  referenced WAV files are materialised through DVC/Server A or copied into the
  attempt's data area and audited as E-B labels.

### 7.3 Site-wide success bar for `mvp_1`

`mvp_1` remains the preferred candidate if:

- PANNs is available in the demo/server environment.
- `policy_aligned_rate >= 0.90`.
- A site-wide holdout report is generated on Server B.
- False positives on no-weather/random Site257 clips are low enough for demo
  use.
- Rain primary examples are detected as non-none rain with high confidence.
- Wind primary examples are detected as non-none wind with acceptable boundary
  tolerance.
- No-weather examples mostly stay `rain=none`, `wind=none`.
- The output includes per-component confidence for wind and rain.
- Thunder is either omitted from the claim or marked as insufficient site data.

---

## 8. Phase 2 / Next Attempt Triggers

Move to `liting__mvp_2__calibrated_weather_head` only if one of these happens:

- PANNs logits are present but intensity buckets are unstable.
- False positives on site-wide no-weather/random clips are too high after
  adding a real ambient negative set.
- Confidence values do not correlate with correctness.
- Mixed rain+wind boundary clips keep failing in a way the demo cannot explain.
- The site-wide validation split shows that the 63-clip weather-positive
  baseline is no longer representative.

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

## 9. Risks / Open Questions

- **Thunder data is weak.** The current Site257 pool does not contain enough
  validated thunder examples for a stable E-B class. Thunder should remain
  conservative until real thunder examples exist.
- **Sound-library exclusion.** Client-facing E-B calibration and evaluation
  must use Site257-derived assets only. If a shared asset index includes
  `source_type=sound_library`, those rows are ignored for E-B MVP claims.
- **E-B data ownership.** Shared weather indexes can provide policy references,
  but E-B should build its own Site257 training/evaluation manifest before
  claiming MVP performance.
- **Weather-pool overfitting.** Training only on curated weather positives would
  make the feature look accurate on the pool while failing ordinary uploaded
  Site257 clips. The MVP must include no-weather and random site-wide holdouts.
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

## 10. Definition of Done

This MVP attempt is done when:

- `liting__mvp_1__panns_weather_baseline` is registered as `head: weather` in
  `acoustic_ai/registry.yaml`.
- The attempt supports registry-based upload analysis:
  `POST /layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/analyze`.
- `e_b_weather_mvp_test.py` passes with PANNs available.
- Server B produces an E-B-owned Site257 training/evaluation manifest.
- Evaluation uses site-derived weather positives, mixed clips, no-weather clips,
  and random site-wide holdout clips for client-facing metrics.
- README documents:
  - model used
  - whether training happened
  - data source
  - upload duration assumptions
  - success/failure interpretation
- The report exposes per-component intensity + confidence for wind, rain, and
  a conservative thunder status.

---

## 11. Client-Facing Explanation

This MVP uses a pre-trained environmental audio tagger, PANNs CNN14, to detect
weather-like audio evidence in an uploaded Site257 soundscape. We combine the
tagger's AudioSet weather evidence with site-specific spectral calibration and
an E-B-owned Site257 validation set. The result is an explainable E-B analysis
head that reports audible wind and rain intensity with confidence for ordinary
Site257 uploads, not only curated weather examples.

At this stage, the model is not trained from scratch. The MVP training target is
a Server B calibration/head-training job over frozen PANNs/DSP features using
Site257 positives, no-weather negatives, mixed clips, and random holdout clips.
Thunder is intentionally not claimed as a validated class until site-derived
thunder audio is found and audited.
