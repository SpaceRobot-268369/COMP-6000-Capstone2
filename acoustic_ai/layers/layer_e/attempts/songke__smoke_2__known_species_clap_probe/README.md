# E-C Known Species Event Detector

## Status

- Layer/head: Layer E, E-C events analysis
- Attempt id: `songke__smoke_2__known_species_clap_probe`
- Registry status: `candidate`
- Shared checkpoint: `model/candidates/songke/mvp_1__layer_e_species_event_detector/`
- Model artifact: DVC-tracked `best_probe.pt`
- Current label set: 13 known Australian species

This attempt does not generate audio. It detects known species events in an
uploaded recording and returns time ranges, confidence values, species match
scores, ecological metadata, and an aggregator-ready `analysis_report`.

## What It Does

Pipeline:

```text
uploaded audio -> 5 s sliding windows -> frozen LAION-CLAP audio encoder
               -> 512-d embeddings -> MLP probe -> merged species events
```

The detector returns:

- `events`: detected species time ranges.
- `species_matches`: per-event average match scores for all known species.
- `phenology`: common name, scientific name, diel signal, season signal, habitat signal, and source URL.
- `analysis_report`: observations and inferred context for future Analysis Mode aggregation.
- `diagnostics.detected_windows`: supporting window-level evidence.

## Known Species

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

## Model Artifact

The shared candidate model lives here:

```text
model/candidates/songke/mvp_1__layer_e_species_event_detector/
```

Files:

- `best_probe.pt.dvc`: git-tracked DVC pointer for the checkpoint binary.
- `best_probe.pt`: DVC-managed binary, not committed to git.
- `README.md`: checkpoint card.
- `params.yaml`: frozen training/inference parameters.
- `metrics.json`: evaluation metrics.

Restore the checkpoint after checking out a branch that contains the `.dvc`
pointer:

```powershell
dvc pull model/candidates/songke/mvp_1__layer_e_species_event_detector/best_probe.pt.dvc
```

If `dvc` is not on PATH on Windows, locate `dvc.exe` in the user Python scripts
directory and run the same command through the full path.

## How To Run In The Dev UI

Start the local services from separate PowerShell windows.

Postgres:

```powershell
cd D:\COMP-6000-Capstone2
docker compose -f services/dev/docker-compose.yml up -d postgres
```

AI server:

```powershell
cd D:\COMP-6000-Capstone2
acoustic_ai\.venv\Scripts\python.exe -m uvicorn acoustic_ai.server.server:app --host 127.0.0.1 --port 8000
```

Backend:

```powershell
cd D:\COMP-6000-Capstone2\backend
$env:DATABASE_URL="postgresql://capstone_user:BKlBXA_pz3uSnjWUb3-hRCh2Wk4fxf0taauc1RxxDD8@localhost:5432/capstone_dev"
$env:PORT="4000"
$env:SESSION_SECRET="oXRnjaEmiSs2jcn-im6ndS5-_IhJqXG2rmlcpINQ6B4"
$env:FRONTEND_URL="http://localhost:5173,http://127.0.0.1:5173"
$env:AI_SERVER_URL="http://127.0.0.1:8000"
$env:AI_REQUEST_TIMEOUT_MS="180000"
npm run dev
```

Frontend:

```powershell
cd D:\COMP-6000-Capstone2\frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Then upload an audio file, find `E-C - Events`, select `Known species events -
CLAP probe`, and click `Run E-C`.

Expected UI sections:

- Detected species timeline
- Species match
- Ecological signal
- Report-ready summary
- Supporting window segments
- Raw report

## API And Output Format

This attempt is registered as:

```text
layer_e / songke__smoke_2__known_species_clap_probe
```

The frontend calls:

```text
POST /api/layers/layer_e/attempts/songke__smoke_2__known_species_clap_probe/analyze
```

Core event shape:

```json
{
  "label": "australian_raven",
  "onset_s": 0.0,
  "offset_s": 29.792,
  "confidence_mean": 0.95066,
  "confidence_max": 0.990115,
  "window_count": 26,
  "species_matches": [
    { "label": "australian_raven", "score": 0.95066 }
  ],
  "phenology": {
    "common_name": "Australian Raven",
    "scientific_name": "Corvus coronoides",
    "diel_signal": "day",
    "diel_confidence": 0.55,
    "season_signal": "weak",
    "season_confidence": 0.2,
    "habitat_signal": "open woodland/farmland/urban"
  }
}
```

Aggregator-ready report shape:

```json
{
  "analysis_report": {
    "schema_version": "analysis_report.v0",
    "scope": "layer_e_events_only",
    "observations": [
      {
        "type": "species_event",
        "source_head": "events",
        "species_label": "australian_raven",
        "common_name": "Australian Raven",
        "time_range_s": [0.0, 29.792],
        "confidence": 0.95066
      }
    ],
    "inferred_context": [
      {
        "type": "diel_signal",
        "source_head": "events",
        "value": "day",
        "confidence": 0.522863
      },
      {
        "type": "habitat_signal",
        "source_head": "events",
        "value": "open woodland/farmland/urban",
        "confidence": 0.95066
      }
    ],
    "disagreements": []
  }
}
```

`observations` are direct model outputs. `inferred_context` is derived from the
species phenology table. `disagreements` is empty for this E-C-only adapter and
is reserved for future E-A/E-B/E-C fusion.

## Phenology Metadata

Phenology metadata lives in:

```text
data/species_phenology.csv
```

Each row provides:

- model label
- common name
- scientific name
- diel signal and confidence
- season signal and confidence
- habitat signal
- short inference note
- source URL

This CSV is report metadata, not model training data. It exists so the final
Analysis Mode report can separate observations from ecological inferences.

## Metrics

Current shared checkpoint:

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| CLAP probe, no-magpie 13 classes | 0.817 | 0.811 |

Per-class test recall:

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

Selected test-split confusions:

| True label | Most common wrong predictions |
|---|---|
| `ninox_boobook` | `peaceful_dove` 6, `podargus_strigoides` 2, `anas_superciliosa` 1, `galah` 1 |
| `laughing_kookaburra` | `peaceful_dove` 4, `cincloramphus_mathewsi` 1, `red_capped_robin` 1, `australian_raven` 1 |
| `rhipidura_leucophrys` | `red_capped_robin` 8, `psophodes_cristatus` 2 |
| `red_capped_robin` | `rainbow_bee_eater` 6, `rhipidura_leucophrys` 3, `galah` 1 |
| `australian_raven` | `laughing_kookaburra` 2, `anas_superciliosa` 2, `crested_bellbird` 1, `rainbow_bee_eater` 1 |
| `crested_bellbird` | No test split errors |
| `rainbow_bee_eater` | `rhipidura_leucophrys` 8, `peaceful_dove` 4, `ninox_boobook` 1, `psophodes_cristatus` 1, `galah` 1 |

## Smoke Checks

Calibrated dense-window checks use:

```text
threshold=0.55, window_s=5.0, hop_s=1.0, merge_gap_s=1.0, min_event_windows=7
```

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

Frontend validation on an Australian Raven clip produced:

- 1 detected event
- 26 / 26 windows hit
- event: `australian_raven`, 0.0-29.792 s, mean 0.951
- report-ready summary: 1 observation, 2 inferences, 0 disagreements

## Reproduce Training

Training intermediates stay under `local_data/` and are gitignored:

```text
local_data/ec_species/clips/
local_data/ec_species/manifests/
local_data/ec_species/embeddings/
local_data/ec_species/models/
```

From the repo root:

```powershell
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\embed_clips.py
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\train_probe.py
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\eval_probe.py
```

Single-clip classifier check:

```powershell
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\predict.py local_data\ec_species\clips\ninox_boobook_positive\ninox_boobook__XC936351__s000000_e005000__clip001.wav
```

Long-audio event check:

```powershell
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\detect.py "C:\path\to\recording.mp3" --output local_data\ec_species\detections\recording_detect.json
```

Registry-facing handler check:

```powershell
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__smoke_2__known_species_clap_probe\code\handler.py "C:\path\to\recording.mp3"
```

## Limitations

- This is a known-species detector. It only knows the 13 labels listed above.
- There is no explicit unknown/background rejection model yet.
- If an uploaded recording contains an unseen species, the model may assign it to the nearest known label.
- Confidence values are model scores, not ecological certainty.
- Seasonal inference is deliberately conservative; weak season signals are not promoted into `analysis_report.inferred_context`.
- E-C does not fuse with E-A or E-B yet. Cross-head disagreements are reserved for the future Analysis Mode aggregator.
