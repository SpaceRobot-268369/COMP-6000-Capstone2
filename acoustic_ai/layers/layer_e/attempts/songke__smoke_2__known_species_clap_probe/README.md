# songke__smoke_2__known_species_clap_probe

E-C known-species detector using frozen CLAP embeddings plus a small
classifier head. The current shared candidate checkpoint is
`model/candidates/songke/mvp_1__layer_e_species_event_detector/`.

## Goal

Use a frozen pretrained audio encoder as the E-C known-species event detector
baseline. Current label set:

- `ninox_boobook`
- `laughing_kookaburra`
- `rhipidura_leucophrys`
- `psophodes_cristatus`
- `cincloramphus_mathewsi`
- `podargus_strigoides`
- `red_capped_robin`
- `anas_superciliosa`
- `australian_raven`
- `peaceful_dove`
- `galah`
- `crested_bellbird`
- `rainbow_bee_eater`

## Method

```
5 s WAV -> frozen LAION-CLAP audio encoder -> 512-d embedding -> MLP probe
```

Generated training intermediates stay under `local_data/` and are gitignored:

```
local_data/ec_species/embeddings/clap_4class/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_4class/
local_data/ec_species/embeddings/clap_5class/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_5class/
local_data/ec_species/embeddings/clap_4class_no_magpie/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_4class_no_magpie/
local_data/ec_species/embeddings/clap_5class_no_magpie/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_5class_no_magpie/
local_data/ec_species/embeddings/clap_6class_no_magpie/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_6class_no_magpie/
local_data/ec_species/embeddings/clap_7class_no_magpie/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_7class_no_magpie/
local_data/ec_species/embeddings/clap_8class_no_magpie/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_8class_no_magpie/
local_data/ec_species/embeddings/clap_9class_no_magpie/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_9class_no_magpie/
local_data/ec_species/embeddings/clap_10class_no_magpie/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_10class_no_magpie/
local_data/ec_species/embeddings/clap_11class_no_magpie/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_11class_no_magpie/
local_data/ec_species/embeddings/clap_12class_no_magpie/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_12class_no_magpie/
local_data/ec_species/embeddings/clap_13class_no_magpie/
local_data/ec_species/models/songke__smoke_2__known_species_clap_probe_13class_no_magpie/
```

The current shared checkpoint artifact is:

```
model/candidates/songke/mvp_1__layer_e_species_event_detector/
```

It contains the DVC-tracked `best_probe.pt` plus git-tracked metadata
(`README.md`, `params.yaml`, and `metrics.json`).

## Reproduce

From the repo root:

```powershell
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\embed_clips.py
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\train_probe.py
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\eval_probe.py
```

Single 5 s clip prediction:

```powershell
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\predict.py local_data\ec_species\clips\ninox_boobook_positive\ninox_boobook__XC936351__s000000_e005000__clip001.wav
```

Sliding-window prediction over a longer recording:

```powershell
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\detect.py "C:\path\to\recording.mp3" --output local_data\ec_species\detections\recording_detect.json
```

Use `--summary-only` to print only counts and merged events while still writing
the full window-level JSON to `--output`.

Registry-facing handler smoke test:

```powershell
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\handler.py "C:\path\to\recording.mp3"
```

The attempt is registered as the Layer E `events` head:

```text
layer_e / songke__smoke_2__known_species_clap_probe
```

`predict.py` prints JSON with:

- `top_label`: highest-scoring trained species.
- `confidence`: softmax score for `top_label`.
- `detected`: whether `confidence >= threshold`.
- `scores`: per-label probabilities.

`detect.py` prints the same prediction fields for each sliding window, plus
`start_s`, `end_s`, `window_index`, `num_windows`, `detected_windows`, and
merged `events`. Defaults are `window_s=5.0`, `hop_s=1.0`, `threshold=0.55`,
`merge_gap_s=1.0`, and `min_event_windows=7`. For very short audio, the
effective minimum is capped to the number of available windows.

Each event contains:

- `label`: detected species.
- `onset_s` / `offset_s`: merged event time range.
- `confidence_mean` / `confidence_max`: confidence summary across merged windows.
- `window_count`: number of detected windows that support the event.
- `species_matches`: event-level mean score for every trained species, sorted high to low.

## Results

Local CLAP probe runs:

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| smoke-2 CLAP probe, 3 classes | 0.765 | 0.775 |
| smoke-2 CLAP probe, 4 classes | 0.798 | 0.803 |
| smoke-2 CLAP probe, 5 classes | 0.730 | 0.737 |
| smoke-2 CLAP probe, no-magpie 4 classes | 0.806 | 0.826 |
| smoke-2 CLAP probe, no-magpie 5 classes | 0.870 | 0.881 |
| smoke-2 CLAP probe, no-magpie 6 classes | 0.872 | 0.873 |
| smoke-2 CLAP probe, no-magpie 7 classes | 0.838 | 0.843 |
| smoke-2 CLAP probe, no-magpie 8 classes | 0.848 | 0.847 |
| smoke-2 CLAP probe, no-magpie 9 classes | 0.814 | 0.812 |
| smoke-2 CLAP probe, no-magpie 10 classes | 0.799 | 0.787 |
| smoke-2 CLAP probe, no-magpie 11 classes | 0.834 | 0.826 |
| smoke-2 CLAP probe, no-magpie 12 classes | 0.847 | 0.841 |
| smoke-2 CLAP probe, no-magpie 13 classes | 0.817 | 0.811 |

Current no-magpie 13-class per-class test recall:

| Label | Recall |
|---|---:|
| `ninox_boobook` | 0.571 |
| `laughing_kookaburra` | 0.741 |
| `rhipidura_leucophrys` | 0.697 |
| `psophodes_cristatus` | 0.889 |
| `cincloramphus_mathewsi` | 0.826 |
| `podargus_strigoides` | 1.000 |
| `red_capped_robin` | 0.697 |
| `anas_superciliosa` | 0.970 |
| `australian_raven` | 0.788 |
| `peaceful_dove` | 0.861 |
| `galah` | 1.000 |
| `crested_bellbird` | 1.000 |
| `rainbow_bee_eater` | 0.545 |

Current no-magpie 13-class main confusions (`rows=true`, selected non-zero errors):

| True label | Most common wrong predictions |
|---|---|
| `ninox_boobook` | `peaceful_dove` 6, `podargus_strigoides` 2, `anas_superciliosa` 1, `galah` 1 |
| `laughing_kookaburra` | `peaceful_dove` 4, `cincloramphus_mathewsi` 1, `red_capped_robin` 1, `australian_raven` 1 |
| `rhipidura_leucophrys` | `red_capped_robin` 8, `psophodes_cristatus` 2 |
| `red_capped_robin` | `rainbow_bee_eater` 6, `rhipidura_leucophrys` 3, `galah` 1 |
| `australian_raven` | `laughing_kookaburra` 2, `anas_superciliosa` 2, `crested_bellbird` 1, `rainbow_bee_eater` 1 |
| `crested_bellbird` | No test split errors |
| `rainbow_bee_eater` | `rhipidura_leucophrys` 8, `peaceful_dove` 4, `ninox_boobook` 1, `psophodes_cristatus` 1, `galah` 1 |

The pretrained CLAP embedding is the current E-C event detector baseline. The
no-magpie thirteen-class model is the active local model for frontend/API testing.

## Single-clip smoke check

Single-clip prediction is useful for checking the classifier API, but it is
not the final E-C user workflow. The user workflow runs overlapping windows
over a longer audio file and merges repeated high-confidence windows into
onset/offset events.

## Sliding-window smoke check

Command run locally on a 444.518 s raw boobook MP3 with a sparse smoke-test hop:

```powershell
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\detect.py "C:\Users\SONGKE HE\Desktop\Ninox boobook\XC1085051 - 布克鹰鸮 - Ninox boobook.mp3" --hop-s 30 --merge-gap-s 30 --threshold 0.6 --output local_data\ec_species\detections\boobook_smoke_events.json
```

Result:

- `num_windows`: 16
- `num_detected_windows`: 16
- `num_events`: 1
- all windows predicted `ninox_boobook`
- confidence range: 0.769 to 0.865
- merged event: `ninox_boobook`, `onset_s=0.0`, `offset_s=444.518`,
  `confidence_mean=0.832051`

Default dense-window checks (`window_s=5`, `hop_s=1`, `threshold=0.6`,
`merge_gap_s=1`) on raw species recordings:

| Source | Duration | Windows | Detected windows | Events | Notes |
|---|---:|---:|---:|---:|---|
| `XC936351` boobook | 24.102 s | 21 | 14 | 1 | `ninox_boobook`, 0.0-24.102 s, mean 0.678 |
| `XC1104895` kookaburra | 28.920 s | 25 | 20 | 1 | `laughing_kookaburra`, 3.0-28.920 s, mean 0.709 |

Calibrated dense-window checks (`threshold=0.55`, `min_event_windows=7`):

| Source | Windows | Detected windows | Events | Notes |
|---|---:|---:|---:|---|
| `XC936351` boobook | 21 | 17 | 1 | `ninox_boobook`, 0.0-24.102 s, mean 0.659 |
| `XC1104895` kookaburra | 25 | 23 | 1 | `laughing_kookaburra`, 1.0-28.920 s, mean 0.692 |
| `XC1069588` rhipidura | 11 | 11 | 1 | `rhipidura_leucophrys`, 0.0-13.0 s, mean 0.757 |
| `XC334404` psophodes | 120 | 109 | 1 | `psophodes_cristatus`, 1.0-121.0 s, mean 0.839 |
| `XC608494` cincloramphus | 22 | 21 | 1 | `cincloramphus_mathewsi`, 5.0-25.479 s, mean 0.932 |
| `XC1048190` podargus | 70 | 54 | 1 | `podargus_strigoides`, 0.0-73.0 s, mean 0.894 |
| `XC1033968` red-capped robin | 21 | 19 | 1 | `red_capped_robin`, 0.0-24.152 s, mean 0.879 |
| `XC1025910` anas | 20 | 19 | 1 | `anas_superciliosa`, 1.0-23.019 s, mean 0.888 |
| `XC1085023` raven | 17 | 16 | 1 | `australian_raven`, 0.0-20.375 s, mean 0.945 |
| `XC1025935` peaceful dove | 17 | 11 | 1 | `peaceful_dove`, 0.0-16.0 s, mean 0.666 |
| `XC328084` galah | 59 | 47 | 2 | `galah`, 0.0-41.0 s mean 0.884 and 43.0-61.0 s mean 0.893 |
| `XC1133174` crested bellbird | 32 | 26 | 1 | `crested_bellbird`, 9.0-35.888 s, mean 0.919 |
| `XC1066693` rainbow bee-eater | 23 | 23 | 1 | `rainbow_bee_eater`, 0.0-26.640 s, mean 0.945 |

This calibrated rule improves recall while requiring at least seven supporting
windows per event. That removes isolated or very short weak events from the
final report without hiding the full window-level diagnostics.

Registry dispatch smoke check on `XC936351` boobook:

- `attempt.head`: `events`
- `report.head`: `events`
- `num_windows`: 21
- `num_detected_windows`: 17
- `num_events`: 1
- event: `ninox_boobook`, 0.0-24.102 s, mean confidence 0.659

## Status

No-magpie thirteen-class embedding cache, probe training, evaluation, single-clip prediction,
and long-audio event detection are working locally. `handler.py` and the
`registry.yaml` entry are in place for the E-C events analysis head. Restart the
AI server before frontend/API testing so it loads the updated no-magpie 13-class
checkpoint and registry params.
