# songke__smoke_3__analysis_aggregator

Layer E Aggregator smoke attempt.

This attempt will combine the three Layer E analysis heads into one final
report:

- E-A ambient context
- E-B weather detection
- E-C species/event detection

The aggregator does not run a new model. It accepts the existing head reports,
keeps direct observations separate from inferred context, and applies
deterministic fusion rules for season and diel estimates.

## Current status

Adapter, deterministic fusion code, a local report-fusion handler, and the
FastAPI analysis orchestrator are implemented and covered by unit tests. The
attempt is registered as the Layer E `aggregator` head and is used by
`POST /analysis/run`.

## Scope for v1

- Pass through E-B weather observations.
- Pass through E-C species/event observations.
- Pass through E-A similar clips as ambient observations.
- Fuse season and diel evidence from E-C and E-A.
- Record disagreements when heads conflict.
- Prefer `undetermined` over false precision when evidence is weak.

## Non-goals

- No new detector training.
- No source separation.
- No LLM weighting.
- No generated audio.

## Files

- `schema.md` documents the v1 fused report contract.
- `params.yaml` will hold thresholds and weights once fusion code starts.
- `code/adapters.py` normalizes E-A / E-B / E-C report shapes.
- `code/aggregator.py` fuses season and diel evidence into the v1 report.
- `code/handler.py` exposes `load()` + `aggregate()` and a local JSON CLI.

## Smoke tests

```powershell
python -m unittest acoustic_ai.tests.test_analysis_aggregator_adapters acoustic_ai.tests.test_analysis_aggregator_fusion
```

Full local test set:

```powershell
python -m unittest acoustic_ai.tests.test_analysis_aggregator_adapters acoustic_ai.tests.test_analysis_aggregator_fusion acoustic_ai.tests.test_analysis_aggregator_handler
```

Orchestrator test:

```powershell
python -m unittest acoustic_ai.tests.test_analysis_orchestrator
```

## Local CLI

Create an input JSON file shaped like:

```json
{
  "ambient_report": {},
  "weather_report": {},
  "events_report": {}
}
```

Then run:

```powershell
python acoustic_ai\layers\layer_e\attempts\songke__smoke_3__analysis_aggregator\code\handler.py input.json --out fused.json
```

## FastAPI endpoint

The full Analysis stack is available inside the AI service at:

```text
POST /analysis/run
```

The request is a multipart upload with a `file` field. The endpoint runs the
registered Layer E ambient, weather, and events heads, then fuses their reports
through this aggregator.

Optional multipart fields can override the registry defaults:

```text
ambient_attempt
weather_attempt
events_attempt
aggregator_attempt
```

The Dev Analysis page sends the currently selected E-A/E-B/E-C/Aggregator
attempts when running Full Analysis, so future tuned models can be tested
without changing code.

The browser-facing Express proxy is:

```text
POST /api/analysis
```

## Sample contract

The aggregator input is three JSON reports produced by the registered E-A,
E-B, and E-C models. These samples use the exact field names from those head
contracts.

### Sample input 1: E-A ambient JSON

```json
{
  "estimated_conditions": {
    "season": "autumn",
    "diel_bin": "afternoon",
    "hour": 15.4,
    "month": 4.1
  },
  "season_source": "probe",
  "similar_clips": [
    {
      "segment_id": "seg_00417",
      "source_clip": "1535532_clip001",
      "similarity": 0.71
    }
  ],
  "confidence": 0.35,
  "season_confidence": 0.35,
  "head_agreement": true,
  "ood_flag": false,
  "k": 5,
  "tau": 0.1
}
```

### Sample input 2: E-B weather JSON

```json
{
  "attempt_id": "murphy__mvp_1__weather_direct_detection",
  "analysis_version": "e_b_weather_mvp_1",
  "audio": {
    "duration_s": 30.04,
    "sample_rate": 22050,
    "channels": 1
  },
  "observations": {
    "weather": {
      "wind": {
        "summary": {
          "intensity": 0.073,
          "variability": 0.0,
          "coverage": 0.0,
          "label": "none",
          "confidence": 0.209
        }
      },
      "rain": {
        "summary": {
          "intensity": 0.073,
          "variability": 0.0,
          "coverage": 0.0,
          "label": "none",
          "confidence": 0.208
        }
      },
      "thunder": {
        "summary": {
          "intensity": 0.072,
          "variability": 0.0,
          "coverage": 0.0,
          "label": "none",
          "confidence": 0.204
        },
        "events": [],
        "mean_interval_s": null
      },
      "confidence": 0.21,
      "derived_label": "none",
      "warnings": []
    }
  },
  "weather": {
    "overall_label": "none",
    "none": true,
    "elements": {
      "rain": { "present": false, "intensity": "none", "confidence": 0.208, "coverage": 0.0 },
      "wind": { "present": false, "intensity": "none", "confidence": 0.209, "coverage": 0.0 },
      "thunder": { "present": false, "intensity": "none", "confidence": 0.204, "coverage": 0.0 }
    },
    "warnings": []
  },
  "window_results": [],
  "debug": {}
}
```

Thunder timeline fields are reserved for future development. `events` may be
`[]` or `null` when E-B is summary-only; when present, event rows should carry
timestamp fields such as `onset_s`/`offset_s` or `start_s`/`end_s`.

### Sample input 3: E-C events JSON

```json
{
  "head": "events",
  "detector": "known_species_clap_probe",
  "known_species": 13,
  "duration_s": 30.04,
  "window_s": 5.0,
  "hop_s": 1.0,
  "threshold": 0.55,
  "merge_gap_s": 1.0,
  "min_event_windows": 7,
  "effective_min_event_windows": 7,
  "num_windows": 27,
  "num_detected_windows": 27,
  "num_events": 1,
  "events": [
    {
      "label": "ninox_boobook",
      "onset_s": 12.4,
      "offset_s": 17.1,
      "confidence_mean": 0.91,
      "confidence_max": 0.98,
      "window_count": 27,
      "phenology": {
        "common_name": "Southern Boobook",
        "scientific_name": "Ninox boobook",
        "diel_signal": "night",
        "diel_confidence": 0.85,
        "season_signal": "weak",
        "season_confidence": 0.2,
        "habitat_signal": "woodland/open woodland",
        "inference_notes": "Nocturnal species; strong diel cue, weak seasonal cue."
      }
    }
  ],
  "analysis_report": {
    "schema_version": "analysis_report.v0",
    "scope": "layer_e_events_only",
    "observations": [
      {
        "id": "ec_event_001",
        "type": "species_event",
        "source_head": "events",
        "species_label": "ninox_boobook",
        "common_name": "Southern Boobook",
        "scientific_name": "Ninox boobook",
        "time_range_s": [12.4, 17.1],
        "confidence": 0.91,
        "confidence_max": 0.98,
        "window_count": 27,
        "evidence": "Southern Boobook detected from 12.40s to 17.10s."
      }
    ],
    "inferred_context": [
      {
        "type": "diel_signal",
        "source_head": "events",
        "value": "night",
        "confidence": 0.7735,
        "evidence_observation_id": "ec_event_001",
        "evidence": "Southern Boobook has a night activity signal and was detected from 12.40s to 17.10s."
      },
      {
        "type": "habitat_signal",
        "source_head": "events",
        "value": "woodland/open woodland",
        "confidence": 0.91,
        "evidence_observation_id": "ec_event_001",
        "evidence": "Southern Boobook is associated with woodland/open woodland."
      }
    ],
    "disagreements": []
  },
  "diagnostics": {
    "detected_windows": []
  }
}
```

Expected output shape:

```json
{
  "schema_version": "analysis_aggregator.v1",
  "mode": "analysis",
  "observations": {
    "ambient": {},
    "weather": {},
    "events": []
  },
  "inferred_context": {
    "diel": {
      "estimate": "night",
      "posterior": 0.88,
      "distribution": {},
      "primary_evidence": "E-C: Southern Boobook supports night",
      "evidence": []
    },
    "season": {
      "estimate": "undetermined",
      "posterior": 0.0,
      "distribution": {},
      "primary_evidence": "E-A: ambient head estimated autumn",
      "evidence": []
    }
  },
  "disagreements": [
    {
      "field": "diel",
      "ambient": "afternoon",
      "events": "night",
      "resolution": "events_preferred"
    }
  ],
  "overall_confidence": 0.0,
  "limitations": [],
  "decision": {},
  "narration": {
    "schema_version": "analysis_narration.v1",
    "source": "deterministic_fallback",
    "summary": "The recording is best described as night with none weather. The season is undetermined. The detected call evidence includes Southern Boobook.",
    "bullets": []
  },
  "llm_input": {
    "schema_version": "analysis_llm_input.v1",
    "task": "Render this ecoacoustic analysis decision JSON as immersive, third-person perspective narration with an analytical tone. Narrate only the provided observations, inferred context, disagreements, limitations, timestamps, and confidence values; do not invent species, season, time of day, weather, certainty, or causes beyond the JSON.",
    "decision": {}
  },
  "model_lineage": {
    "ambient": { "id": "lucas__mvp_2__clap_knn_probe_enlarged", "head": "ambient" },
    "weather": { "id": "murphy__mvp_1__weather_direct_detection", "head": "weather" },
    "events": { "id": "songke__smoke_2__known_species_clap_probe", "head": "events" },
    "aggregator": { "id": "songke__smoke_3__analysis_aggregator", "head": "aggregator" }
  }
}
```
