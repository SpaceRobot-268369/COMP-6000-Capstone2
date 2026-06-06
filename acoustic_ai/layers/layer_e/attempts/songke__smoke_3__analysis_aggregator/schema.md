# Analysis Aggregator v1 Schema

This schema is the target output for Layer E Aggregator v1.

The aggregator receives three existing head reports:

```json
{
  "ambient_report": {},
  "weather_report": {},
  "events_report": {}
}
```

It returns one fused report:

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
    "diel": {},
    "season": {}
  },
  "decision": {},
  "llm_input": {},
  "disagreements": [],
  "confidence": 0.0,
  "limitations": []
}
```

## Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Fixed schema name for API and frontend compatibility. |
| `mode` | string | Always `analysis` for this report. |
| `observations` | object | Direct things the heads heard in the audio. |
| `inferred_context` | object | Aggregator guesses from evidence, not direct detections. |
| `decision` | object | Compact machine-readable final decision for downstream narration. |
| `llm_input` | object | Wrapper payload for a future LLM narration step. |
| `disagreements` | array | Conflicts or weak evidence that affected the final answer. |
| `confidence` | number | Overall fused confidence in `[0, 1]`. |
| `limitations` | array | Human-readable caveats. |

## Observations

Observations are direct outputs from the detector heads. The aggregator should
not reinterpret them.

### `observations.ambient`

Source: E-A ambient head.

```json
{
  "similar_clips": [
    {
      "segment_id": "seg_00417",
      "source_clip": "1535532_clip001",
      "similarity": 0.71
    }
  ],
  "estimated_conditions": {
    "season": "autumn",
    "diel_bin": "night",
    "hour": 22.4,
    "month": 4.1
  },
  "confidence": 0.35,
  "season_confidence": 0.42,
  "ood_flag": false
}
```

Required for v1:

- `similar_clips`
- `estimated_conditions`
- `confidence`
- `season_confidence`
- `ood_flag`

### `observations.weather`

Source: E-B weather head.

This block should be copied from `weather_report.observations.weather`.

```json
{
  "wind": {
    "summary": {
      "intensity": 0.62,
      "variability": 0.40,
      "coverage": 0.95,
      "label": "moderate",
      "confidence": 0.83
    }
  },
  "rain": {
    "summary": {
      "intensity": 0.10,
      "variability": 0.70,
      "coverage": 0.20,
      "label": "light",
      "confidence": 0.55
    }
  },
  "thunder": {
    "summary": {
      "intensity": 0.00,
      "variability": 0.00,
      "coverage": 0.00,
      "label": "none",
      "confidence": 0.90
    },
    "events": [],
    "mean_interval_s": null
  },
  "confidence": 0.80,
  "derived_label": "rain+wind",
  "warnings": []
}
```

Required for v1:

- `wind.summary`
- `rain.summary`
- `thunder.summary`
- `confidence`
- `derived_label`
- `warnings`

### `observations.events`

Source: E-C events head.

The aggregator normalizes each detected event into this report-facing shape:

```json
[
  {
    "label": "ninox_boobook",
    "common_name": "Southern Boobook",
    "scientific_name": "Ninox boobook",
    "confidence": 0.91,
    "onset_s": 12.4,
    "offset_s": 17.1,
    "phenology": {
      "diel_signal": "night",
      "diel_confidence": 0.85,
      "season_signal": "weak",
      "season_confidence": 0.20,
      "habitat_signal": "woodland/open woodland"
    }
  }
]
```

Required for v1:

- `label`
- `confidence`
- `onset_s`
- `offset_s`
- `phenology.diel_signal`
- `phenology.diel_confidence`
- `phenology.season_signal`
- `phenology.season_confidence`

## Inferred context

Inferred context is the aggregator's answer. These are not direct observations.

### `inferred_context.diel`

```json
{
  "estimate": "night",
  "posterior": 0.88,
  "distribution": {
    "dawn": 0.02,
    "morning": 0.04,
    "afternoon": 0.06,
    "night": 0.88
  },
  "primary_evidence": "E-C: Southern Boobook has a night activity signal",
  "evidence": [
    {
      "source_head": "events",
      "value": "night",
      "weight": 0.77,
      "reason": "Southern Boobook detected with high confidence"
    },
    {
      "source_head": "ambient",
      "value": "night",
      "weight": 0.12,
      "reason": "Ambient head estimated night with low confidence"
    }
  ]
}
```

Allowed `estimate` values:

- `dawn`
- `morning`
- `afternoon`
- `night`
- `undetermined`

### `inferred_context.season`

```json
{
  "estimate": "undetermined",
  "posterior": 0.40,
  "distribution": {
    "spring": 0.15,
    "summer": 0.40,
    "autumn": 0.35,
    "winter": 0.10
  },
  "primary_evidence": "No strongly seasonal species detected",
  "evidence": [
    {
      "source_head": "ambient",
      "value": "autumn",
      "weight": 0.10,
      "reason": "Ambient head estimated autumn, but confidence was weak"
    }
  ]
}
```

Allowed `estimate` values:

- `spring`
- `summer`
- `autumn`
- `winter`
- `undetermined`

## Disagreements

Disagreements explain conflicts or weak evidence. They are part of the report,
not debug-only output.

```json
[
  {
    "field": "diel",
    "ambient": "afternoon",
    "events": "night",
    "resolution": "events_preferred",
    "reason": "Southern Boobook is a stronger time-of-day cue than ambient texture"
  }
]
```

Allowed `resolution` values for v1:

- `events_preferred`
- `ambient_used_as_fallback`
- `direct_observation_kept`
- `low_confidence_range_reported`
- `undetermined`

Conflict policy:

- E-B weather is a direct acoustic observation. The aggregator keeps it as the
  weather decision and does not let E-A or E-C overwrite it.
- E-C species/events are direct call observations. Strong species phenology can
  override E-A for `time_of_day` / `diel`.
- E-A ambient context is a weaker prior. It can fill `time_of_day` or `season`
  only when E-C has no stronger evidence, and its fusion weight remains capped
  below E-C so weak ambient estimates do not become hard claims.
- If there is no clear winner, the field stays `undetermined` and the
  disagreement is recorded with `low_confidence_range_reported`.
- The LLM must not resolve conflicts. It receives the aggregator's decision and
  the recorded disagreements, then translates them into readable language.

## Decision JSON

`decision` is the compact result that should be passed to a later LLM step. It
keeps the fields that humans usually ask for: when the recording sounds like it
was made, what season is most likely, what weather was detected, and what calls
or acoustic events were heard.

```json
{
  "schema_version": "analysis_decision.v1",
  "time_of_day": {
    "value": "night",
    "confidence": 0.88,
    "distribution": {
      "dawn": 0.02,
      "morning": 0.04,
      "afternoon": 0.06,
      "night": 0.88
    },
    "evidence": "E-C: Southern Boobook supports night"
  },
  "season": {
    "value": "undetermined",
    "confidence": 0.40,
    "distribution": {
      "spring": 0.15,
      "summer": 0.40,
      "autumn": 0.35,
      "winter": 0.10
    },
    "evidence": "No reliable season evidence"
  },
  "weather": {
    "label": "wind",
    "confidence": 0.80,
    "rain": { "label": "none", "confidence": 0.90, "intensity": 0.0, "coverage": 0.0 },
    "wind": { "label": "moderate", "confidence": 0.83, "intensity": 0.62, "coverage": 0.95 },
    "thunder": { "label": "none", "confidence": 0.90, "intensity": 0.0, "coverage": 0.0 },
    "warnings": []
  },
  "detected_calls": [
    {
      "label": "ninox_boobook",
      "common_name": "Southern Boobook",
      "scientific_name": "Ninox boobook",
      "confidence": 0.91,
      "onset_s": 12.4,
      "offset_s": 17.1,
      "diel_signal": "night",
      "diel_confidence": 0.85,
      "season_signal": "weak",
      "season_confidence": 0.20,
      "habitat_signal": "woodland/open woodland"
    }
  ],
  "disagreements": [
    {
      "field": "diel",
      "ambient": "afternoon",
      "events": "night",
      "resolution": "events_preferred",
      "reason": "Event phenology is stronger context evidence than ambient texture."
    }
  ],
  "overall_confidence": 0.72,
  "limitations": []
}
```

## LLM input

`llm_input` wraps `decision` with a task instruction. It is intentionally
grounded: the LLM should convert the JSON into readable language, not invent
extra species, seasons, weather, or certainty.

```json
{
  "schema_version": "analysis_llm_input.v1",
  "task": "Convert this ecoacoustic analysis decision JSON into concise, human-readable language. Do not invent species, season, time of day, weather, or confidence beyond the provided fields.",
  "decision": {}
}
```

## Confidence

`confidence` is a simple summary for v1. It should be conservative:

- high when E-C and E-A agree, or E-C has strong phenology evidence;
- medium when only E-A contributes useful context;
- low when evidence is weak or conflicting.

The exact formula belongs in `params.yaml` and the future aggregator code, not
in this schema.

## Limitations

Always include limitations. Common v1 limitations:

```json
[
  "Season is difficult to infer from audio alone at this site.",
  "Ambient context is a weak prior, not ground truth.",
  "The species detector only covers the known species in its checkpoint."
]
```

## Minimal successful v1 response

```json
{
  "schema_version": "analysis_aggregator.v1",
  "mode": "analysis",
  "observations": {
    "ambient": {
      "similar_clips": [],
      "estimated_conditions": null,
      "confidence": 0.0,
      "season_confidence": 0.0,
      "ood_flag": false
    },
    "weather": {
      "wind": { "summary": { "intensity": 0.0, "variability": 0.0, "coverage": 0.0, "label": "none", "confidence": 0.0 } },
      "rain": { "summary": { "intensity": 0.0, "variability": 0.0, "coverage": 0.0, "label": "none", "confidence": 0.0 } },
      "thunder": {
        "summary": { "intensity": 0.0, "variability": 0.0, "coverage": 0.0, "label": "none", "confidence": 0.0 },
        "events": [],
        "mean_interval_s": null
      },
      "confidence": 0.0,
      "derived_label": "none",
      "warnings": []
    },
    "events": []
  },
  "inferred_context": {
    "diel": {
      "estimate": "undetermined",
      "posterior": 0.0,
      "distribution": { "dawn": 0.25, "morning": 0.25, "afternoon": 0.25, "night": 0.25 },
      "primary_evidence": "No reliable diel evidence",
      "evidence": []
    },
    "season": {
      "estimate": "undetermined",
      "posterior": 0.0,
      "distribution": { "spring": 0.25, "summer": 0.25, "autumn": 0.25, "winter": 0.25 },
      "primary_evidence": "No reliable season evidence",
      "evidence": []
    }
  },
  "disagreements": [],
  "confidence": 0.0,
  "limitations": [
    "No reliable context evidence was available."
  ]
}
```
