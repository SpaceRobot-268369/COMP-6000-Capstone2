# Server B Runbook - E-B Weather Analysis MVP 1

Attempt:

```text
liting__mvp_1__panns_weather_baseline
```

Branch:

```text
feat/liting/e-b-weather-analysis-main-sync
```

Purpose:

```text
Run the formal Server B calibration/evaluation pass for E-B weather-layer
analysis. The target is not a curated weather-pool classifier. The target is
Site257 upload analysis: given an arbitrary Site257 clip, estimate audible
wind/rain intensity and confidence while controlling false positives on
ordinary no-weather site audio.
```

## 1. Pre-Run Checks

Run on Server B from the repo root:

```bash
pwd
git branch --show-current
git status --short
```

Expected branch:

```text
feat/liting/e-b-weather-analysis-main-sync
```

If the branch is not current:

```bash
git fetch origin
git checkout feat/liting/e-b-weather-analysis-main-sync
git pull --ff-only
```

Check Python and DVC:

```bash
./acoustic_ai/.venv/bin/python --version
dvc --version
```

Expected:

- AI code uses `./acoustic_ai/.venv/bin/python`.
- DVC can run from user/system path, as documented in the repo workflow.

## 2. Required Data Access

The run needs Site257-derived audio only. Sound-library assets are excluded from
client-facing E-B claims.

Required sources:

| Source | Purpose |
|---|---|
| `resources/site_257_bowra-dry-a/` | Site-wide candidate universe and random holdout source. |
| `data/site257_weather_policy_snapshot.csv` | Policy reference for known Site257 weather-positive candidates. |
| Current materialised smoke assets | Runnable seed positives while the larger Site257 set is materialised. |
| E-B no-weather candidates | Required false-positive control group. |

Pull DVC assets if needed:

```bash
dvc pull
```

Do not add or push large raw audio files to git. If new binary artifacts are
created for the formal run, track them through DVC according to the project DVC
workflow.

## 3. Build E-B-Owned Manifest

The formal MVP evidence should use an E-B-owned manifest, not only the curated
weather-positive pool.

Template:

```text
acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/data/e_b_site257_weather_manifest_template.csv
```

Target output:

```text
acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/data/e_b_site257_weather_training_manifest.csv
```

Minimum required groups:

| Group | Requirement |
|---|---|
| Weather positives | Audited Site257 wind/rain examples. |
| Mixed clips | Site257 clips where weather overlaps with birds, insects, or ambience. |
| No-weather negatives | Ambient/event clips with no audible rain or wind. |
| Random holdout | Site-wide random clips not used for fitting thresholds. |

Split policy:

| Split | Purpose |
|---|---|
| `train` | Fit thresholds or a small calibration head. |
| `validation` | Tune confidence/intensity decisions. |
| `holdout` | Estimate real upload behaviour and false positives. |

## 4. Run Current Local-Equivalent MVP Test

Use this first to confirm the branch behaves the same on Server B:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/tests/e_b_weather_mvp_test.py
```

Expected report:

```text
debug/e_b_weather_mvp/report.json
```

The report should expose:

- wind intensity and confidence
- rain intensity and confidence
- thunder status, normally `insufficient_site_data`
- method used
- whether PANNs was available
- policy-aligned pass rate
- no-weather false-positive checks

## 5. PANNs Calibration Evidence

Run:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/layers/layer_e/attempts/liting__mvp_1__panns_weather_baseline/code/calibrate_panns_weather.py
```

Expected output folder:

```text
debug/e_b_weather_mvp/panns_calibration/
```

This stage is calibration/evaluation over frozen PANNs/DSP features. It is not
full PANNs backbone fine-tuning.

## 6. Formal MVP Success Bar

The attempt is ready for team/client review only when Server B produces an
evaluation report that includes:

| Check | Required result |
|---|---|
| Weather-positive detection | Wind/rain positives are detected with reasonable intensity and confidence. |
| Mixed audio robustness | Weather can still be detected in clips containing other ecoacoustic content. |
| No-weather false positives | No-weather clips return `wind=none` and `rain=none`. |
| Random holdout behaviour | Site-wide random clips do not produce inflated weather claims. |
| Thunder discipline | Thunder is not claimed unless Site257-derived examples are audited. |
| Artifact traceability | Report lists manifest path, split counts, method, and limitations. |

## 7. Expected Runtime

| Task | Expected time |
|---|---:|
| Server B setup check | 5-15 min |
| DVC pull / materialise required clips | 15-60 min |
| Build E-B manifest and split | 10-30 min |
| Extract PANNs/DSP features | 5-20 min |
| Fit thresholds or small head | <5 min |
| Run evaluation and write report | 3-10 min |
| API/frontend demo sanity check | 5-15 min |

Expected total:

```text
30-90 minutes, depending mostly on data materialisation time.
```

## 8. What To Report Back

After the run, send the team:

```text
Branch:
feat/liting/e-b-weather-analysis-main-sync

Attempt:
liting__mvp_1__panns_weather_baseline

Server B report:
<report path>

Manifest:
<manifest path>

Summary:
- split counts
- model/method used
- wind/rain metrics
- no-weather false-positive result
- thunder status
- known limitations
```

