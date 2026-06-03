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

### Step 8 — ServerB AST AudioSet smoke

Tested a third off-the-shelf AudioSet model on serverB:

`MIT/ast-finetuned-audioset-10-10-0.4593`

Runtime:

- `transformers`
- `torch`
- GPU: `cuda:0`
- input resampled to 16 kHz by the test script

Smoke-tested the same three human-reviewed site weather clips used for PANNs.

Result summary:

| sample | human label | AST strongest relevant labels |
| --- | --- | --- |
| `001_site257_214657_001545_001575` | rain+wind medium | Wind `0.395`, Thunderstorm `0.270`, Wind noise `0.264`, Thunder `0.253`, Rain `0.067` |
| `006_site257_214871_006725_006755` | wind heavy | Wind `0.462`, Wind noise `0.365`, Thunderstorm `0.128`, Rain `0.091`, Thunder `0.085` |
| `012_site257_214872_001700_001730` | thunder maybe | Thunder `0.129`, Wind noise `0.124`, Wind `0.111`, Thunderstorm `0.103`, Rain `0.039` |

Interpretation:

- AST is more conservative than PANNs.
- AST handles the `wind heavy` sample better: it ranks `Wind` / `Wind noise`
  above `Thunder` / `Thunderstorm`.
- AST may be useful as a guard against PANNs thunder overcalls.
- AST rain scores are low on these mixed site clips, so it should not be the
  only rain detector.
- AST thunder scores are also low on the ambiguous thunder sample, so it should
  not be the only thunder detector.
- Best next direction: use AST as a calibration/confirmation channel in a
  multi-evidence fusion policy, not as the sole classifier.

### Step 9 — ServerB BEATs AudioSet smoke

Tested Microsoft BEATs with an AudioSet-finetuned checkpoint on serverB.

Runtime setup in the personal experiment venv only:

- installed `torchaudio==2.11.0+cu128`
- downloaded official BEATs source files to `~/beats_data/code`
- downloaded checkpoints to `~/beats_data/checkpoints`
- used AudioSet-finetuned checkpoint:
  `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`

Notes:

- The first downloaded `BEATs_iter3_plus_AS2M.pt` checkpoint was encoder-only
  and did not include a classifier head, so it was not useful for E-B label
  smoke testing.
- The fine-tuned checkpoint includes `label_dict` and a 527-class classifier.

Smoke-tested the same three human-reviewed site weather clips.

Result summary:

| sample | human label | BEATs strongest relevant labels |
| --- | --- | --- |
| `001_site257_214657_001545_001575` | rain+wind medium | Rain `0.233`, Thunder `0.182`, Wind `0.176`, Wind noise `0.153`, Thunderstorm `0.145` |
| `006_site257_214871_006725_006755` | wind heavy | Rustling leaves `0.243`, Wind `0.149`, Wind noise `0.138`, Rain `0.098`, Thunder `0.012`, Thunderstorm `0.015` |
| `012_site257_214872_001700_001730` | thunder maybe | Wind `0.209`, Wind noise `0.174`, Thunder `0.058`, Rain `0.035`, Thunderstorm `0.033` |

Interpretation:

- BEATs is the most conservative of the tested AudioSet-style models so far.
- BEATs does not strongly overcall thunder on the wind-heavy clip, which is an
  improvement over PANNs.
- BEATs correctly keeps bird/insect/speech scores very low on these samples.
- BEATs absolute weather probabilities are low, so thresholds cannot be copied
  from PANNs or AST.
- BEATs may be useful as a robust evidence channel for reducing false thunder
  positives and for confirming broad rain/wind texture, but it is not enough as
  a sole classifier.

### Step 10 — Composite element-level aggregation

Updated the CLI fusion/aggregation path so E-B makes independent element-level
decisions before deriving the top-level weather label.

Changes:

- `combine_scores` now accepts CLAP scores, AudioSet scores, and feature
  support.
- Fusion weights are normalised based on which channels are available.
- `aggregate_weather` now computes per-element:
  - `confidence`
  - `coverage`
  - `present`
  - `intensity`
- Composite labels are built from present elements instead of forcing a single
  winning class.
- Mixed outputs can now naturally become:
  - `rain+wind`
  - `rain+thunder`
  - `wind+thunder`
  - `rain+thunder+wind`
- Added warning support for:
  - `weather_mixed_with_ambient`
  - `possible_rain_under_wind`
  - `possible_wind_overload`

Validation:

- Local Python compile passed.
- Local CLI smoke on a rain WAV with both model backends disabled completed and
  emitted schema-shaped JSON with `confidence` and `coverage` for all elements.

Interpretation:

- This does not solve model accuracy yet.
- It does establish the correct E-B decision shape for mixed weather: detect
  rain/wind/thunder separately, then derive the composite label.

### Step 11 — ServerB CLAP-only composite smoke

Synced the current local attempt to the isolated serverB smoke workdir and ran
the composite CLI on three human-reviewed Nov 2019 site clips.

Inputs were audit MP3 previews converted to temporary 22.05 kHz mono WAV files
under `/tmp/e_b_composite_smoke_wav/`.

Outputs were written under:

`dev-artifacts-self-testing/composite_smoke_clap_001/`

Result summary:

| sample | human label | E-B CLAP-only label | rain | wind | thunder |
| --- | --- | --- | --- | --- | --- |
| `rain_wind_001` | rain+wind medium | `none` | `0.478` | `0.537` | `0.514` |
| `wind_heavy_006` | wind heavy | `none` | `0.479` | `0.542` | `0.489` |
| `thunder_maybe_012` | thunder maybe | `none` | `0.389` | `0.504` | `0.524` |

Interpretation:

- Composite aggregation executes successfully on serverB.
- CLAP-only evidence is close to thresholds but not enough to mark elements
  present under current conservative settings.
- This confirms that composite logic is structurally ready, but threshold tuning
  should wait until AudioSet channels are integrated into the CLI.
- Do not solve this by simply lowering all thresholds for CLAP-only; that would
  likely increase false positives. The next implementation step should add
  PANNs/AST/BEATs score channels or a small calibration harness.

### Step 12 — ServerB CLAP + BEATs one-off fusion smoke

Ran a one-off serverB experiment combining:

- CLAP max window scores from `composite_smoke_clap_001`
- BEATs full-clip AudioSet scores from the same three human-reviewed clips

Tested two weighted-average settings:

- current documented weights: CLAP `0.65`, BEATs `0.25`, features `0.10`
- exploratory weights: CLAP `0.55`, BEATs `0.35`, features `0.10`

Result summary:

| sample | human label | CLAP max rain/wind/thunder | BEATs rain/wind/thunder | current weighted output |
| --- | --- | --- | --- | --- |
| `rain_wind_001` | rain+wind medium | `0.552 / 0.590 / 0.569` | `0.232 / 0.177 / 0.182` | `none` |
| `wind_heavy_006` | wind heavy | `0.553 / 0.595 / 0.535` | `0.096 / 0.149 / 0.015` | `none` |
| `thunder_maybe_012` | thunder maybe | `0.449 / 0.547 / 0.573` | `0.036 / 0.210 / 0.058` | `none` |

Interpretation:

- Naive weighted averaging with BEATs is not useful because BEATs absolute
  scores are much lower than CLAP scores.
- BEATs is still useful, but not as a simple probability averaged with CLAP.
- Better next fusion design: use CLAP as the sensitive detector and BEATs/AST as
  a relative gate or guard, especially to suppress false thunder positives.
- Example: do not accept thunder unless CLAP thunder is high and BEATs/AST does
  not strongly prefer wind/noise over thunder.

### Step 13 — Rule-based multi-model gate smoke

Ran a second one-off fusion experiment using all four evidence channels:

- CLAP
- PANNs
- AST
- BEATs

Important design change:

- Do not average raw model probabilities directly.
- Use model rankings and relative support as gates.
- Treat PANNs thunder as sensitive but noisy.
- Treat AST/BEATs as conservative guards against false thunder.
- Treat BEATs rain support as a stronger confirmation for rain than PANNs rain
  alone, because PANNs rain is broad on storm/wind textures.

Rule v2 results:

| sample | human label | rule v2 output | notes |
| --- | --- | --- | --- |
| `rain_wind_001` | rain+wind medium | `rain+wind` | correct composite, low confidence, possible wind overload |
| `wind_heavy_006` | wind heavy | `wind` | fixed the previous false rain positive |
| `thunder_maybe_012` | thunder maybe | `wind+thunder` | plausible but low confidence; flagged wind overload |

Rule v2 confidence values:

| sample | rain | wind | thunder |
| --- | ---: | ---: | ---: |
| `rain_wind_001` | `0.475` | `0.477` | `0.517` |
| `wind_heavy_006` | `0.446` | `0.488` | `0.412` |
| `thunder_maybe_012` | `0.372` | `0.402` | `0.483` |

Interpretation:

- The element-level composite approach is viable.
- Raw model scores are not calibrated probabilities; confidence must be
  calibrated after rule decisions, likely by mapping evidence patterns to
  confidence bands instead of reporting raw weighted averages.
- Rule v2 is a better MVP direction than naive weighted averaging.
- Next implementation step: encode rule-style gate fusion in the CLI behind a
  transparent calibration policy, then run a tiny 15-25 clip calibration set.

### Step 14 — Gate fusion module scaffold

Added:

`code/gate_fusion.py`

Purpose:

- preserve the rule v2 fusion behavior in reusable code
- keep composite weather decisions element-level
- map raw model scores into honest MVP confidence bands
- avoid treating raw CLAP/PANNs/AST/BEATs scores as calibrated probabilities

Current function:

`decide_weather_from_evidence(evidence, coverage=None)`

Expected optional evidence channels:

- `clap`
- `panns`
- `ast`
- `beats`
- `features`

Smoke input used the recorded score triplets from Step 13.

Smoke output:

| sample | human label | gate module output |
| --- | --- | --- |
| `rain_wind_001` | rain+wind medium | `rain+wind` |
| `wind_heavy_006` | wind heavy | `wind` |
| `thunder_maybe_012` | thunder maybe | `wind+thunder` |

Notes:

- Confidence is now banded after the rule decision rather than reported as raw
  weighted model score.
- Example: `rain_wind_001` outputs rain confidence `0.606` and wind confidence
  `0.606`, even though raw evidence confidence is around `0.47`.
- This is intentionally more readable for users while still exposing raw scores
  in debug output.

Validation:

- Local smoke of the module passed.
- Python compile passed.

Remaining work:

- Wire real PANNs/AST/BEATs channels into the CLI.
- Use this fusion module for final `weather` output once model channels are
  available.
- Run a tiny 15-25 clip calibration set before API/frontend work.

### Step 15 — Real PANNs scorer wired into CLI

Implemented the real PANNs CNN14 AudioSet scorer in:

`code/audioset_scores.py`

Behavior:

- loads `panns_inference.AudioTagging`
- uses 32 kHz mono audio
- maps AudioSet labels to E-B groups:
  - rain: `Rain`, `Raindrop`, `Rain on surface`
  - wind: `Wind`, `Wind noise (microphone)`
  - thunder: `Thunder`, `Thunderstorm`
  - bio contamination: `Bird`, `Bird vocalization, bird call, bird song`,
    `Insect`
  - human/machine contamination: `Speech`, `Vehicle`, `Engine`, `Machinery`
- stores raw weather labels and top labels in `audioset_scores.raw`
- degrades safely when `panns_inference` is unavailable

Validation:

- Local compile passed.
- Local fallback smoke passed without PANNs installed.
- ServerB CLAP+PANNs CLI smoke ran on the three human-reviewed WAV-converted
  clips.

ServerB CLAP+PANNs CLI smoke summary:

| sample | human label | CLI output | rain | wind | thunder |
| --- | --- | --- | ---: | ---: | ---: |
| `rain_wind_001` | rain+wind medium | `none` | `0.405` | `0.433` | `0.535` |
| `wind_heavy_006` | wind heavy | `none` | `0.457` | `0.433` | `0.504` |
| `thunder_maybe_012` | thunder maybe | `none` | `0.368` | `0.398` | `0.551` |

Interpretation:

- PANNs is now technically available to the CLI.
- The current CLI still uses weighted aggregation, so output remains too
  conservative and not useful as final E-B judgment.
- PANNs scores differ between full-clip one-off tests and 5 s window CLI tests;
  calibration must account for windowing.
- Next step should route CLAP/PANNs evidence into `gate_fusion.py` instead of
  relying on weighted-average aggregation.

### Step 16 — CLI routed through gated fusion, AST guard added

Implemented the final `weather` decision path through `code/gate_fusion.py`
instead of the old weighted-average threshold.

Changes:

- `run_weather_analysis.py` now aggregates per-window evidence peaks and calls
  `decide_weather_from_evidence(...)`.
- Output debug now exposes evidence channels:
  - `clap`
  - `panns`
  - `ast`
  - `features`
- Added optional `--guard-backend ast`.
- `audioset_scores.py` now includes a Hugging Face AST AudioSet scorer:
  `MIT/ast-finetuned-audioset-10-10-0.4593`.
- Rain confirmation now uses:
  - BEATs when available,
  - otherwise PANNs or AST as conservative support.

Validation:

- Local Python compile passed.
- Local `none` smoke passed.
- ServerB CLAP+PANNs+AST smoke passed on three human-reviewed samples.

ServerB CLAP+PANNs+AST gate summary:

| sample | human label | CLI output | key interpretation |
| --- | --- | --- | --- |
| `rain_wind_001` | rain+wind medium | `rain+wind` | AST top rain confirms rain while CLAP top wind keeps wind present. |
| `wind_heavy_006` | wind heavy | `wind` | AST top wind prevents PANNs thunder false positive from becoming thunder. |
| `thunder_maybe_012` | thunder maybe | `thunder` | CLAP/PANNs/AST all top thunder, so gate promotes thunder. |

Current judgment rule:

- CLAP is the sensitive candidate detector.
- PANNs is useful but noisy, especially for thunder on wind-like textures.
- AST is now the main conservative guard for `wind` vs `thunder`, and also
  helps confirm rain when BEATs is not wired into CLI.
- `possible_wind_overload` remains as a warning when thunder-like evidence is
  present but may be wind overload.

Remaining work:

- Run a slightly larger calibration set before exposing this as an API feature.
- Decide whether BEATs should be wired into CLI or kept as a research-only
  cross-check.
- Tune confidence/intensity mapping once calibration data is larger than three
  examples.

### Step 17 — 12-sample calibration smoke

Ran a small human-reviewed calibration set on ServerB with:

```bash
--model-backend clap --audioset-backend panns --guard-backend ast
```

Temporary input/output location:

- local: `/tmp/e_b_calibration_001`
- ServerB: `/tmp/e_b_calibration_001`

Calibration mix:

- `rain`: 2
- `rain+wind`: 4
- `wind`: 3
- `thunder`: 2
- `none`: 1

Summary:

| metric | result |
| --- | --- |
| exact label match | 8 / 12 |
| expected elements contained in output | 8 / 12 |

Per-sample summary:

| sample | expected | output | note |
| --- | --- | --- | --- |
| `rain_heavy_site` | rain | rain | correct |
| `rain_light_site` | rain | rain | correct |
| `rain_wind_storm_001` | rain+wind | wind | missed rain |
| `rain_wind_storm_006` | rain+wind | wind | missed rain |
| `rain_wind_storm_015` | rain+wind | wind | missed rain |
| `rain_wind_nov_001` | rain+wind | rain+wind | correct |
| `wind_nov_006` | wind | wind | correct |
| `wind_nov_009` | wind | wind | correct |
| `wind_storm_022` | wind | wind | correct |
| `thunder_nov_012` | thunder | thunder | correct |
| `thunder_nov_017` | thunder | wind | uncertain thunder missed |
| `none_quiet_091` | none | none | correct |

Interpretation:

- Pure rain and wind are usable for MVP-level analysis.
- `rain+wind` recall is still too conservative: three mixed clips become
  `wind` only.
- Thunder remains deliberately conservative. One maybe-thunder clip became
  wind, which is acceptable if we prefer avoiding false thunder, but should be
  documented.
- The next useful change is not another model test. It is a small gate-tuning
  pass for rain-under-wind:
  - allow weak rain when CLAP rain is close to CLAP wind and PANNs/AST has any
    rain support,
  - keep the confidence low and add `possible_rain_under_wind`,
  - do not change pure wind behavior.

### Step 18 — Rain-under-wind gate tuning

Implemented a small rule change in:

`code/gate_fusion.py`

Change:

- Keep the existing strict rain path for clearer rain.
- Add a weaker `rain_under_wind_candidate` path when:
  - CLAP rain is not dominant but is close to CLAP wind,
  - CLAP wind is clearly present,
  - PANNs or AST has at least some rain support.
- Promote that rain element with `weak` confidence only.
- Always attach `possible_rain_under_wind` so downstream consumers know the
  mixed label is cautious.

Local rule-only checks:

| case | expected | output | notes |
| --- | --- | --- | --- |
| synthetic mixed rain-under-wind | `rain+wind` | `rain+wind` | rain confidence stays low (`0.42`) and warning is present |
| synthetic pure wind | `wind` | `wind` | no false rain added |

Validation:

- `PYTHONPYCACHEPREFIX=/tmp/e_b_pycache python3 -m py_compile ...` passed for
  `gate_fusion.py` and `run_weather_analysis.py`.
- This is a local rule check only. The next step should rerun the 12-sample
  ServerB calibration set with `CLAP + PANNs + AST` and compare exact match plus
  `rain+wind` recall before API/frontend work.

### Step 19 — 12-sample ServerB calibration after gate tuning

Synced the updated `gate_fusion.py` to the personal ServerB smoke workdir:

`~/murphy/analysis-layer-E-B-smoke`

Reran the same 12-sample calibration set:

```bash
--model-backend clap --audioset-backend panns --guard-backend ast
```

Temporary output location:

`/tmp/e_b_calibration_001/results_step18`

First pass result after a too-wide rain-under-wind rule:

| metric | result |
| --- | --- |
| exact label match | 8 / 12 |
| rain+wind | 4 / 4 |
| wind | 0 / 3 |

Interpretation: rain+wind recall improved, but pure wind was over-promoted to
`rain+wind`. The weak gate was too permissive.

Adjusted the weak rain-under-wind path to require:

- `CLAP wind >= 0.45`
- `CLAP rain >= 0.49`
- `CLAP rain >= CLAP wind - 0.04`
- `CLAP rain >= CLAP thunder - 0.15`
- PANNs or AST weak rain support

Second pass result:

| metric | result |
| --- | --- |
| exact label match | 10 / 12 |
| rain | 2 / 2 |
| rain+wind | 3 / 4 |
| wind | 3 / 3 |
| thunder | 1 / 2 |
| none | 1 / 1 |

Per-sample summary:

| sample | expected | output | note |
| --- | --- | --- | --- |
| `rain_heavy_site` | rain | rain | correct |
| `rain_light_site` | rain | rain | correct |
| `rain_wind_storm_001` | rain+wind | rain+wind | correct; weak rain-under-wind |
| `rain_wind_storm_006` | rain+wind | wind | missed rain; gate prefers wind precision |
| `rain_wind_storm_015` | rain+wind | rain+wind | correct; weak rain-under-wind |
| `rain_wind_nov_001` | rain+wind | rain+wind | correct |
| `wind_nov_006` | wind | wind | fixed previous false rain |
| `wind_nov_009` | wind | wind | fixed previous false rain |
| `wind_storm_022` | wind | wind | fixed previous false rain |
| `thunder_nov_012` | thunder | thunder | correct |
| `thunder_nov_017` | thunder | wind | still conservative; possible thunder missed |
| `none_quiet_091` | none | none | correct |

Current interpretation:

- This is a better MVP tradeoff than the previous gate: wind precision is
  preserved while rain+wind recall improves from 1/4 to 3/4.
- The remaining `rain_wind_storm_006` miss has weak independent rain support,
  so keeping it as wind is acceptable for an honest MVP.
- The remaining thunder miss is consistent with the project policy to avoid
  false thunder when wind overload is plausible.

### Step 20 — Gate v1.1 frozen; holdout sanity check started

Froze the current rules as **gate v1.1** for the next sanity check.

Gate v1.1 tradeoff:

- preserve pure wind precision,
- allow cautious low-confidence `rain+wind` when rain is close under wind,
- keep `possible_rain_under_wind` on weak mixed-weather promotions,
- keep thunder conservative when wind overload is plausible.

Prepared an 8-sample holdout set outside the repo:

`/tmp/e_b_holdout_001`

Composition:

- `rain`: 2 site-derived clips
- `rain+wind`: 2 site-derived clips
- `wind`: 2 site-derived clips
- `thunder`: 1 pure thunder sanity clip
- `none`: 1 quiet site ambience control

Copied the holdout set to ServerB and ran:

```bash
--model-backend clap --audioset-backend panns --guard-backend ast
```

ServerB output directory:

`/tmp/e_b_holdout_001/results_gate_v1_1`

Status:

- The 8 jobs completed on ServerB.
- Result summarisation completed after the session resumed.

Holdout result:

| metric | result |
| --- | --- |
| exact label match | 6 / 8 |
| rain | 2 / 2 |
| rain+wind | 0 / 2 |
| wind | 2 / 2 |
| thunder | 1 / 1 |
| none | 1 / 1 |

Per-sample summary:

| sample | expected | output | note |
| --- | --- | --- | --- |
| `rain_holdout_001` | rain | rain | correct |
| `rain_holdout_002` | rain | rain | correct |
| `rain_wind_holdout_001` | rain+wind | wind | missed rain; rain confidence `0.268` |
| `rain_wind_holdout_002` | rain+wind | wind | missed rain; rain confidence `0.274` |
| `wind_holdout_001` | wind | wind | correct |
| `wind_holdout_002` | wind | wind | correct |
| `thunder_holdout_001` | thunder | thunder | correct; resampled library sanity clip |
| `none_holdout_001` | none | none | correct |

Interpretation:

- Gate v1.1 is precise and stable for pure rain, wind, thunder sanity, and none.
- It remains conservative for some `rain+wind` site clips.
- Do not loosen the gate blindly: the previous wide rain-under-wind rule damaged
  wind precision. The next useful step is to inspect whether these two holdout
  mixed clips are truly rain+wind to human ears, then decide whether E-B should
  report them as `wind` with a warning or adjust a narrow mixed-weather rule.

Human follow-up listen:

- The two missed holdout `rain+wind` clips were replayed in
  `debug/e_b_holdout_rain_wind_listen_001/listen.html`.
- User judgment: rain is extremely subtle and weak in both clips.
- Decision: gate v1.1 conservative `wind` output is acceptable for MVP. Do not
  loosen the mixed-weather gate based on these two clips.

### Step 21 — Human-audited regression spot-check for gate v1.1

Created a small regression manifest from previously human-audited Layer B site
weather clips:

`calibration/gate_v1_1_regression_spotcheck.csv`

Composition:

- `rain`: 3
- `wind`: 3
- `rain+wind`: 4
- `thunder`: 2

All 12 WAV paths exist locally. This is not a broad benchmark; it is a small
guardrail to catch obvious gate regressions using clips the user already
reviewed.

Server B run:

```bash
cd /tmp/e_b_gate_regression_001
/home/ubuntu/murphy/COMP-6000-Capstone2/acoustic_ai/.venv/bin/python \
  acoustic_ai/layers/layer_e/attempts/murphy__mvp_1__weather_direct_detection/code/run_weather_analysis.py \
  <audio.wav> \
  --model-backend clap \
  --audioset-backend panns \
  --guard-backend ast \
  --out outputs/<audio_id>.json
```

Summary from `evaluate_weather_outputs.py`:

| metric | result |
| --- | --- |
| exact label match | 7 / 12 |
| rain | 3 / 3 |
| wind | 2 / 3 |
| rain+wind | 1 / 4 |
| thunder | 1 / 2 |

Mismatch diagnosis:

- `rain+wind -> wind` cases have rain confidence around `0.27` and are
  consistent with prior listening feedback that site rain under wind is often
  extremely subtle.
- `rain+wind -> rain` had strong CLAP/PANNs/AST rain support but weak wind
  confirmation. This is a known limitation of the current wind gate.
- `wind -> rain+wind` fired `possible_rain_under_wind`; this is a mild false
  positive but remains explicitly warned.
- `thunder -> wind` remains the expected thunder/wind-overload ambiguity.

Decision:

- Keep gate v1.1 frozen for now.
- Do not loosen rain-under-wind or thunder gates based on this spot-check.
- Treat this manifest as a regression set for future gate edits.
- Next useful improvement is operational, not policy: run calibration in one
  batch process so CLAP/PANNs/AST load once instead of reloading for each clip.
