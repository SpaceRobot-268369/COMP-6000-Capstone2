# DEVLOG

## 2026-06-03

### Step 1 — Schema and policy scaffold

Created the independent E-B weather direct-detection attempt:

`murphy__mvp_1__weather_direct_detection`

Decision:

- E-B is an analysis feature, not a Layer B retrieval/generation feature.
- The target output is weather element presence, intensity, and confidence for
  `rain`, `wind`, and `thunder`.
- If no clear weather layer is detected, output `overall_label: none`.
- Direct detection on the uploaded mixture is the MVP path.
- Source separation is explicitly out of scope.

Files added:

- `README.md`
- `schema.md`
- `weather_analysis_policy.md`
- `params.yaml`
- `__init__.py`

### Step 2 — Offline CLI skeleton

Added `code/run_weather_analysis.py`.

Current behavior:

- reads WAV input
- converts to mono
- resamples to 22,050 Hz
- splits into 5 s windows with 2.5 s hop
- computes lightweight audio features
- emits schema-shaped JSON

Validation:

- Python compile passed.
- CLI smoke output succeeded on a local rain WAV.

Important limitation:

- No CLAP/PANNs/YAMNet scores yet.
- Placeholder feature scores are conservative and should not be interpreted as
  real detection accuracy.

### Step 3 — Model scorer boundary

Added `code/model_scores.py` and wired it into the CLI.

Current behavior:

- defines a `WindowScorer` interface
- defines prompt groups for weather and contamination
- exposes a CLAP scorer boundary
- safely returns `model_scores_unavailable` until a concrete CLAP backend is
  implemented

Decision:

- Do not pretend handcrafted feature scores are model confidence.
- Keep model scores, feature scores, and fused scores separate in the JSON.

Validation:

- Python compile passed.
- CLI smoke output includes `model_scores`, `feature_scores`, and
  `model_scores_unavailable`.

### Step 4 — CLAP backend discovery and adapter

Inspected existing Layer E ambient CLAP attempts and found the reusable contract:

- `laion/clap-htsat-unfused`
- `transformers.ClapModel` + `ClapProcessor`
- CLAP audio input at 48 kHz mono
- L2-normalised audio/text embeddings
- cosine similarity for ranking

Decision:

- Keep Murphy's attempt self-contained per repo convention, but follow the
  existing CLAP contract instead of inventing a new one.
- Add a real `TransformersClapScorer` behind the existing scorer boundary.
- Preserve safe degradation: if CLAP dependencies or model files are missing,
  the CLI still emits schema-shaped JSON with `model_scores_unavailable`.

Files added/updated:

- `code/clap_backbone.py`
- `code/model_scores.py`

### Step 5 — ServerB CLAP smoke

Ran one real CLAP smoke on serverB in an isolated workdir:

`~/murphy/analysis-layer-E-B-smoke`

The live `~/shiny-pikachu` service clone was not touched. The existing dirty
Layer B personal clone was not modified; its venv and one debug WAV were only
used as inputs for this smoke run.

Input:

`/home/ubuntu/murphy/COMP-6000-Capstone2/debug/site_weather_nov2019_storm_scout_001/windows_wav/0654_site257_214823_002249_002279.wav`

Output:

`acoustic_ai/layers/layer_e/attempts/murphy__mvp_1__weather_direct_detection/dev-artifacts-self-testing/serverb_clap_smoke_001.json`

Result summary:

```json
{
  "model_scores_available": true,
  "overall_label": "none",
  "warnings": ["unsupported_sample_rate_resampled"],
  "first_window_scores": {
    "rain": 0.368483,
    "wind": 0.510325,
    "thunder": 0.267831,
    "none": 0.372418,
    "bio_contamination": 0.219299,
    "human_machine_contamination": 0.2013
  }
}
```

Interpretation:

- The real CLAP backend loads and scores successfully on serverB.
- This sample is below current `wind_present` threshold, so the aggregate label
  remains `none`.
- The next step is threshold/calibration smoke on a tiny balanced set, not API
  or frontend integration yet.

### Step 6 — AudioSet/PANNs model boundary

Checked repo and serverB dependency state for a second analysis model.

Findings:

- Project docs recommend PANNs CNN14 as the strongest off-the-shelf weather
  AudioSet baseline.
- Existing repo code does not include a PANNs implementation yet.
- serverB venv has `torch` and `transformers`.
- serverB venv does not currently have `panns_inference`, `torchaudio`,
  `tensorflow`, or `tensorflow_hub`.

Decision:

- Add an AudioSet/PANNs scorer boundary now.
- Do not install packages or wire PANNs implementation in this step.
- CLI now reports `audioset_scores_unavailable` separately from CLAP scores.
- PANNs will be added as a cross-check model after dependency/runtime choice.

Files added/updated:

- `code/audioset_scores.py`
- `code/run_weather_analysis.py`
- `schema.md`

### Step 7 — ServerB PANNs smoke

Installed `panns-inference` in the personal serverB experiment venv and cached
the PANNs CNN14 checkpoint:

`/home/ubuntu/panns_data/Cnn14_mAP=0.431.pth`

AudioSet labels relevant to E-B exist directly:

- `Rain`
- `Raindrop`
- `Rain on surface`
- `Wind`
- `Wind noise (microphone)`
- `Thunder`
- `Thunderstorm`
- `Bird`
- `Insect`
- `Speech`

Smoke-tested three human-reviewed site weather clips from the Nov 2019 storm
scout page.

Result summary:

| sample | human label | PANNs strongest weather labels |
| --- | --- | --- |
| `001_site257_214657_001545_001575` | rain+wind medium | Thunder `0.699`, Thunderstorm `0.562`, Rain `0.428`, Wind `0.109` |
| `006_site257_214871_006725_006755` | wind heavy | Thunder `0.530`, Thunderstorm `0.477`, Rain `0.436`, Wind `0.127` |
| `012_site257_214872_001700_001730` | thunder maybe | Thunder `0.700`, Thunderstorm `0.599`, Rain `0.433`, Wind `0.060` |

Interpretation:

- PANNs detects broad weather-like texture better than the first random smoke,
  and its labels are useful evidence.
- PANNs is not safe as the sole primary element classifier for E-B because it
  strongly overcalls `Thunder` / `Thunderstorm` on heavy wind and rain+wind.
- This confirms the user-observed ambiguity: wind overload and thunder-like
  low-frequency bursts are hard to separate by a single off-the-shelf model.
- Next model step should treat PANNs as one evidence channel, then test a third
  model or calibrated rules before making final E-B decisions.
