# Layer E-B Weather Analysis Handoff

This file is the single handoff document for a new Codex session. Read this
first, then read `README.md`, `DEVLOG.md`, `schema.md`, and
`weather_analysis_policy.md` in this same attempt folder.

## Current Branch and Attempt

- Branch: `feat/murphy/analysis-layer-E-B`
- Last pushed commit with this attempt: `c14d433`
- Attempt folder:
  `acoustic_ai/layers/layer_e/attempts/murphy__mvp_1__weather_direct_detection`
- ServerB personal smoke workdir:
  `~/murphy/analysis-layer-E-B-smoke`
- ServerB live service clone must not be touched:
  `~/shiny-pikachu`

## User Goal

Build Layer E-B as an independent analysis feature.

Given an uploaded audio clip, output which weather elements are present and how
strong/confident they are:

- `rain`
- `wind`
- `thunder`

If no clear weather layer is detected, output `none`.

Important: E-B is analysis only. Do not turn this into Layer B retrieval,
generation, source separation, or Layer D mixing.

## Desired Output

The MVP output should include:

- overall label:
  - `none`
  - `rain`
  - `wind`
  - `thunder`
  - `rain+wind`
  - `rain+thunder`
  - `wind+thunder`
  - `rain+thunder+wind`
- per-element result for rain/wind/thunder:
  - `present`
  - `intensity`: `none`, `light`, `medium`, `heavy`
  - `confidence`: number in `[0, 1]`
- warnings:
  - `low_confidence`
  - `possible_bio_overlap`
  - `possible_human_or_machine_overlap`
  - `possible_wind_overload`
  - `possible_clipping`
  - `weather_mixed_with_ambient`
  - `short_audio`
  - `unsupported_sample_rate_resampled`
  - `model_scores_unavailable`
  - `audioset_scores_unavailable`
- optional per-window evidence for audit/debug.

See `schema.md` for the exact contract.

## Completed

### 1. Attempt scaffold

Created:

- `README.md`
- `DEVLOG.md`
- `schema.md`
- `weather_analysis_policy.md`
- `params.yaml`
- `__init__.py`
- `code/`
- `dev-artifacts-self-testing/`

The attempt follows repo convention: work stays under the Layer E attempt
folder, and smoke outputs are ignored except `.gitkeep`.

### 2. CLI skeleton

Implemented:

`code/run_weather_analysis.py`

Current behavior:

- reads WAV input
- converts to mono
- resamples to 22,050 Hz
- splits into 5 s windows with 2.5 s hop
- computes lightweight acoustic features:
  - RMS
  - peak
  - clipping ratio
  - spectral centroid
  - spectral flatness
  - spectral entropy
  - low 20-700 Hz energy ratio
  - high 2-8 kHz energy ratio
- emits schema-shaped JSON
- works even when model dependencies are missing

Validation:

- Local Python compile passed using:
  `PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 -m compileall -q ...`

### 3. CLAP scorer boundary and backend

Implemented:

- `code/model_scores.py`
- `code/clap_backbone.py`

Current CLAP backend:

- model: `laion/clap-htsat-unfused`
- implementation: `transformers.ClapModel` + `ClapProcessor`
- audio input: 48 kHz mono
- scoring: L2-normalised audio/text embeddings + cosine similarity
- safe fallback when CLAP dependencies/model are unavailable

ServerB smoke result:

- CLAP loaded successfully.
- It produced real scores.
- On one debug weather clip, it returned:
  - rain: `0.368`
  - wind: `0.510`
  - thunder: `0.268`
  - none: `0.372`
- The sample stayed below current wind threshold, so output was `none`.

Conclusion:

- CLAP backend works technically.
- CLAP should not be trusted as the only decision source because earlier Layer B
  retrieval work showed many weather confusions.

### 4. AudioSet/PANNs boundary

Implemented:

- `code/audioset_scores.py`
- CLI now reports `audioset_scores` separately from CLAP/model scores.
- If PANNs is unavailable, JSON still completes with
  `audioset_scores_unavailable`.

### 5. PANNs dependency smoke on ServerB

Installed in the personal experiment venv only:

- `panns-inference`

Cached checkpoint on ServerB:

- `/home/ubuntu/panns_data/Cnn14_mAP=0.431.pth`

Important AudioSet labels exist:

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

## Tried but Not Good Enough

### CLAP as primary judge

Why tried:

- CLAP is available in the repo's Layer E context.
- It can score open-vocabulary prompts.

Problem:

- Prior Layer B retrieval work showed CLAP often confuses wind, rain, thunder,
  and contamination.
- It is useful as one evidence channel, but not enough as final primary judge.

Decision:

- Keep CLAP as auxiliary evidence.
- Do not make CLAP the sole primary detector.

### PANNs as primary judge

Why tried:

- PANNs CNN14 is a strong AudioSet model.
- It has direct weather labels, unlike CLAP prompt matching.

Smoke samples from human-reviewed site clips:

| sample | human label | PANNs strongest weather labels |
| --- | --- | --- |
| `001_site257_214657_001545_001575` | rain+wind medium | Thunder `0.699`, Thunderstorm `0.562`, Rain `0.428`, Wind `0.109` |
| `006_site257_214871_006725_006755` | wind heavy | Thunder `0.530`, Thunderstorm `0.477`, Rain `0.436`, Wind `0.127` |
| `012_site257_214872_001700_001730` | thunder maybe | Thunder `0.700`, Thunderstorm `0.599`, Rain `0.433`, Wind `0.060` |

Problem:

- PANNs strongly overcalls `Thunder` / `Thunderstorm` on heavy wind and
  rain+wind.
- This matches the user observation that wind overload can sound like thunder.

Decision:

- Keep PANNs as useful AudioSet evidence.
- Do not make PANNs the sole final classifier.
- Thunder needs extra rules or another model before final decisions.

### 6. AST AudioSet smoke on ServerB

Tested:

- `MIT/ast-finetuned-audioset-10-10-0.4593`
- runtime: `transformers` + `torch`
- GPU: `cuda:0`

Smoke samples:

| sample | human label | AST strongest relevant labels |
| --- | --- | --- |
| `001_site257_214657_001545_001575` | rain+wind medium | Wind `0.395`, Thunderstorm `0.270`, Wind noise `0.264`, Thunder `0.253`, Rain `0.067` |
| `006_site257_214871_006725_006755` | wind heavy | Wind `0.462`, Wind noise `0.365`, Thunderstorm `0.128`, Rain `0.091`, Thunder `0.085` |
| `012_site257_214872_001700_001730` | thunder maybe | Thunder `0.129`, Wind noise `0.124`, Wind `0.111`, Thunderstorm `0.103`, Rain `0.039` |

Interpretation:

- AST is more conservative than PANNs.
- AST is better than PANNs for the wind-heavy case because it does not strongly
  overcall thunder.
- AST is weak for rain on mixed site clips.
- AST is also weak for ambiguous thunder.
- AST should be used as a confirmation/calibration channel, especially to reduce
  false thunder positives, not as the only detector.

### 7. BEATs AudioSet smoke on ServerB

Tested Microsoft BEATs with an AudioSet-finetuned checkpoint.

ServerB scratch/runtime details:

- code: `~/beats_data/code`
- checkpoints: `~/beats_data/checkpoints`
- useful checkpoint:
  `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`
- unused encoder-only checkpoint:
  `BEATs_iter3_plus_AS2M.pt`
- installed in personal experiment venv:
  `torchaudio==2.11.0+cu128`

Important setup note:

- Default `pip install torchaudio` installed a mismatched wheel requiring
  `libcudart.so.13`.
- It was replaced with the CUDA 12.8 wheel from PyTorch:
  `torchaudio==2.11.0+cu128`.

Smoke samples:

| sample | human label | BEATs strongest relevant labels |
| --- | --- | --- |
| `001_site257_214657_001545_001575` | rain+wind medium | Rain `0.233`, Thunder `0.182`, Wind `0.176`, Wind noise `0.153`, Thunderstorm `0.145` |
| `006_site257_214871_006725_006755` | wind heavy | Rustling leaves `0.243`, Wind `0.149`, Wind noise `0.138`, Rain `0.098`, Thunder `0.012`, Thunderstorm `0.015` |
| `012_site257_214872_001700_001730` | thunder maybe | Wind `0.209`, Wind noise `0.174`, Thunder `0.058`, Rain `0.035`, Thunderstorm `0.033` |

Interpretation:

- BEATs is more conservative than PANNs and AST.
- BEATs does not strongly overcall thunder on wind-heavy audio.
- This makes BEATs useful as a false-thunder guard.
- Its absolute weather scores are low, so it needs its own thresholds.
- It should not be the sole detector.

## Current Technical State

The E-B CLI currently has these layers:

1. audio preprocessing and windowing
2. feature extraction
3. CLAP scorer channel
4. AudioSet/PANNs scorer channel boundary
5. tested AST AudioSet as a third evidence model
6. tested BEATs AudioSet as a fourth evidence model
7. added composite element-level aggregation
8. conservative fused output

### 8. Composite element-level aggregation

The CLI now detects weather elements independently and derives the top-level
label from the present elements.

Current behavior:

- each element has:
  - `present`
  - `intensity`
  - `confidence`
  - `coverage`
- if rain and wind are both present, output becomes `rain+wind`
- if rain, wind, and thunder are present, output becomes `rain+thunder+wind`
- if no element is present, output remains `none`

Important:

- This is not source separation.
- This is not perfect attribution of how much rain vs wind is in the audio.
- It is element-level direct detection with explicit uncertainty.

Warnings now include:

- `weather_mixed_with_ambient`
- `possible_rain_under_wind`
- `possible_wind_overload`

Validation:

- Local Python compile passed.
- Local CLI smoke with model backends disabled emitted valid JSON with
  `confidence` and `coverage` fields.

### 9. Gate fusion module

Added:

`code/gate_fusion.py`

This module preserves the best current fusion direction from ServerB one-off
tests.

Core API:

```python
decide_weather_from_evidence(evidence, coverage=None)
```

Expected optional evidence channels:

- `clap`
- `panns`
- `ast`
- `beats`
- `features`

The module does not average raw probabilities. It uses rule-style gates:

- CLAP is the sensitive candidate detector.
- PANNs is weather-sensitive but noisy for thunder.
- AST and BEATs are conservative guards, especially against false thunder.
- BEATs rain support is stronger rain confirmation than PANNs rain alone.

Recorded score smoke:

| sample | human label | gate module output |
| --- | --- | --- |
| `rain_wind_001` | rain+wind medium | `rain+wind` |
| `wind_heavy_006` | wind heavy | `wind` |
| `thunder_maybe_012` | thunder maybe | `wind+thunder` |

Important:

- The module maps raw score patterns into confidence bands.
- Final `confidence` is not raw CLAP/PANNs/AST/BEATs probability.
- Raw confidence and model tops are exposed in debug.

But final calibrated fusion is not done yet.

The current code is best understood as an MVP analysis scaffold plus model
evidence probes, not a finished accurate detector.

## Unfinished

### 1. Real PANNs scorer integration in code

`audioset_scores.py` currently has the boundary/fallback. It still needs the
real implementation that imports `panns_inference.AudioTagging` when available.

Suggested mapping:

- rain:
  - max of `Rain`, `Raindrop`, `Rain on surface`
- wind:
  - max of `Wind`, `Wind noise (microphone)`
- thunder:
  - max of `Thunder`, `Thunderstorm`
- bio contamination:
  - max of `Bird`, `Bird vocalization, bird call, bird song`, `Insect`
- human/machine contamination:
  - max of speech, vehicle, engine, machinery-like labels

Important:

- PANNs expects 32 kHz mono audio.
- Do not use PANNs output directly as final labels.
- Store raw labels in `audioset_scores.raw` for calibration/debug.

### 2. AST scorer integration

AST and BEATs have now been smoke-tested manually, and rule-style gate fusion
has a reusable module. PANNs/AST/BEATs are still not integrated into the CLI.

Next implementation step:

- add AST/BEATs scorer boundaries or extend `audioset_scores.py`
- store AST raw AudioSet labels separately from PANNs raw labels
- store BEATs raw AudioSet labels separately from PANNs/AST raw labels
- avoid replacing final decisions with AST alone
- use AST and BEATs mostly as guards against PANNs thunder overcalls
- then route those evidence channels into `code/gate_fusion.py`

Possible fallback if AST is not enough:

- YAMNet, but it likely needs TensorFlow/TensorFlow Hub and may add heavier
  dependency friction.

### 3. Calibration set

Need a tiny, human-reviewed calibration set, not a huge audit.

Suggested initial set:

- 5 clear rain clips
- 5 clear wind clips
- 5 likely thunder/storm clips
- 5 no-weather/ambient clips
- 5 bio-dominant or human/machine contamination clips

Use existing debug/audit clips where possible. Do not start another large
manual audit.

### 4. Fusion rules

Need calibrated fusion that combines:

- CLAP scores
- PANNs/AudioSet scores
- third-model scores if useful
- acoustic features
- warning rules

Likely rules:

- `rain` requires rain evidence plus not too much bio/human contamination.
- `wind` requires wind evidence or wind-like low-frequency/broadband feature
  evidence.
- `thunder` must not be accepted from PANNs alone.
- `thunder` should require stronger transient/low-frequency evidence or
  agreement from a second model.
- if all element confidences are weak, output `none`.
- if wind and thunder conflict, prefer `possible_wind_overload` until calibrated.
- composite labels should be derived from element decisions, not from a separate
  single-class classifier.

### 5. API/frontend integration

Not started.

Do not start this until CLI accuracy is reasonable on the tiny calibration set.

Eventually E-B should expose:

- upload audio
- run analysis
- display overall weather label
- display rain/wind/thunder strength and confidence
- display warnings
- optionally display per-window evidence or timeline

## Recommended Next Steps

1. Implement real PANNs scorer in `code/audioset_scores.py`.
2. Run E-B CLI on 5-10 known human-reviewed clips and save JSON under
   `dev-artifacts-self-testing/`.
3. Add a short note to `DEVLOG.md` summarising whether PANNs helps as an
   evidence channel after code integration.
4. Integrate AST and/or BEATs scoring, or run them on the same tiny calibration set.
5. Compare CLAP / PANNs / AST / BEATs against human labels.
6. Decide fusion thresholds and confidence bands.
7. Only then connect to API/frontend.

## Do Not Do

- Do not continue Layer B retrieval work inside this E-B attempt.
- Do not use Layer B pool labels as the only truth source.
- Do not run large audits before a tiny calibration set.
- Do not make CLAP the primary final judge.
- Do not make PANNs the primary final judge.
- Do not touch `~/shiny-pikachu` on ServerB.
- Do not auto-commit immediately after every change unless the user asks.

## Useful Commands

Local compile:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 -m compileall -q \
  acoustic_ai/layers/layer_e/attempts/murphy__mvp_1__weather_direct_detection/code
```

ServerB smoke clone:

```bash
ssh -i ~/.ssh/shinypokemon.pem ubuntu@shinypokemon.adelaideuni.cloud
cd ~/murphy/analysis-layer-E-B-smoke
```

ServerB venv:

```bash
/home/ubuntu/murphy/COMP-6000-Capstone2/acoustic_ai/.venv/bin/python
```

PANNs checkpoint:

```bash
/home/ubuntu/panns_data/Cnn14_mAP=0.431.pth
```

Example CLI:

```bash
./acoustic_ai/.venv/bin/python \
  acoustic_ai/layers/layer_e/attempts/murphy__mvp_1__weather_direct_detection/code/run_weather_analysis.py \
  /path/to/input.wav \
  --out acoustic_ai/layers/layer_e/attempts/murphy__mvp_1__weather_direct_detection/dev-artifacts-self-testing/weather_smoke.json
```

## One-Sentence Summary for New Codex

Layer E-B should analyze uploaded audio and report rain/wind/thunder presence,
intensity, and confidence; the scaffold and CLAP/PANNs/AST/BEATs probes are
done, but none is accurate enough alone, so the next work is model-channel
integration and calibrated fusion on a tiny human-reviewed set before
API/frontend work.
