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

The browser-facing Express proxy is:

```text
POST /api/analysis
```

## Sample contract

Small input sketch:

```json
{
  "ambient_report": {
    "estimated_conditions": { "season": "autumn", "diel_bin": "afternoon" },
    "confidence": 0.35,
    "season_confidence": 0.35
  },
  "weather_report": {
    "observations": {
      "weather": { "derived_label": "none", "confidence": 0.8 }
    }
  },
  "events_report": {
    "events": [
      {
        "label": "ninox_boobook",
        "confidence_mean": 0.91,
        "phenology": {
          "common_name": "Southern Boobook",
          "diel_signal": "night",
          "diel_confidence": 0.85,
          "season_signal": "weak",
          "season_confidence": 0.2
        }
      }
    ]
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
  "llm_input": {}
}
```
