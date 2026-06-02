# Layer E Smoke 1 Detectors

## Scope

This attempt is the first end-to-end smoke test for Layer E analysis detectors.
The branch currently wires the E-B weather head into both CLI and web upload
flows. E-A ambient context and E-C event detection are placeholders in the API
response.

## E-B Weather Smoke Test

E-B estimates audible weather evidence from an uploaded ecoacoustic mixture:

- rain intensity
- wind intensity
- thunder intensity
- per-component confidence
- explainable spectral features

The detector does not perform source separation. It scores the raw mixture
directly, matching the analysis-mode architecture.

## Data Source

The smoke test uses Murphy's latest Layer B site-weather output from:

```text
origin/feat/murphy/generative-layer-b-from-site
```

The source assets were produced on Server A:

```text
/home/ubuntu/layer_b_site_weather_job/runs/mvp_pool_20260530_001_layer_d_assets_accept_only/
```

Server source copied from:

```text
/home/ubuntu/layer_b_site_weather_job/runs/mvp_pool_20260530_001_layer_d_assets_accept_only/
```

The local smoke-test copy is owned by this Layer E attempt:

```text
acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/
```

Important files:

```text
layer_d_ready_manifest.csv
summary.json
policy_version.txt
assets_wav_22050_mono/
```

The WAV assets are ignored by git. They were copied from Server A for local
demo/testing. The manifest records the Server A CLAP-first candidate policy
outputs and points to 22.05 kHz mono WAV assets.

Asset summary:

| Category | Count |
|---|---:|
| rain_primary | 4 |
| rain_wind_mixed | 6 |
| wind_primary | 53 |
| Total | 63 |

## Method

There are two stages in the current smoke path:

1. Murphy's Server A pipeline selected site 257 candidate clips using a
   CLAP-first weather policy. CLAP weather scores, contamination scores, and
   environmental priors were used to promote accepted rain/wind assets.
2. This E-B detector loads those promoted labels as calibration references,
   extracts spectral/audio features from the uploaded clip, and predicts the
   nearest calibrated rain/wind/thunder intensity bucket.

Current method name in API output:

```text
site257_clap_promoted_calibrated_spectral_nearest_centroid
```

This means the calibration labels are CLAP-first site-weather labels, while the
live detector is currently spectral calibration. It is not yet a live CLAP pass
on every uploaded file.

## CLI Smoke Test

Run from the repository root:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/tests/e_b_weather_smoke_test.py
```

Current result:

```text
Report written to: debug/e_b_weather_smoke/report.json
Summary: pass=28, partial=25, fail=10, pass_or_partial_rate=0.84
PASS
```

Interpretation:

- `pass`: predicted bucket matches the expected label.
- `partial`: predicted bucket is adjacent or close enough for smoke testing.
- `fail`: predicted component or intensity is not aligned.

This is a smoke-test agreement rate, not a production accuracy score.

## Web Demo

The FastAPI `/analysis` endpoint loads the same site257 promoted calibration
assets. The frontend can upload an audio file and show:

- mel spectrogram
- rain intensity
- wind intensity
- thunder intensity
- confidence
- raw feature report

Demo URLs when the local E-B stack is running:

```text
http://localhost:5174/analysis
http://localhost:5174/dev/analysis
```

Example rain file:

```text
acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/rain_primary/rain_primary__rain__site257_1313196_001218_001233.wav
```

Example wind file:

```text
acoustic_ai/layers/layer_e/attempts/liting__smoke_1__e_b_weather_analysis/data/analysis/site257_clap_promoted/assets_wav_22050_mono/wind_primary/wind_primary__wind__site257_214837_001286_001301.wav
```

## Limitations

- E-B is currently a smoke baseline, not a production classifier.
- Live analysis does not yet run CLAP prompt scoring per upload.
- Rain coverage is limited compared with wind.
- Site-derived thunder is not promoted as an MVP primary pool; thunder should
  remain conservative or use a library fallback.
- Mixed rain/wind clips can confuse component intensity estimates.

## Next Steps

1. Add live CLAP zero-shot scoring for uploaded audio.
2. Combine CLAP scores with the current spectral features.
3. Add PANNs or YAMNet as an independent weather-tagging baseline.
4. Add clean negative/no-weather site examples for better calibration.
5. Surface top calibration neighbours and CLAP prompt scores in the analysis
   report for better explainability.
