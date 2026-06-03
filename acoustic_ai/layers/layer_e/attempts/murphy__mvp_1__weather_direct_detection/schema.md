# E-B Weather Analysis Schema

This schema describes the MVP output for Layer E-B weather analysis. It is
independent of Layer B generation/retrieval schemas.

## Top-Level Result

```json
{
  "attempt_id": "murphy__mvp_1__weather_direct_detection",
  "analysis_version": "e_b_weather_mvp_1",
  "audio": {
    "duration_s": 10.0,
    "sample_rate": 22050,
    "channels": 1
  },
  "observations": {
    "weather": {
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
  },
  "weather": {
    "overall_label": "rain+wind",
    "none": false,
    "elements": {
      "rain": {
        "present": true,
        "intensity": "medium",
        "confidence": 0.78,
        "coverage": 0.64
      },
      "wind": {
        "present": true,
        "intensity": "light",
        "confidence": 0.62,
        "coverage": 0.38
      },
      "thunder": {
        "present": false,
        "intensity": "none",
        "confidence": 0.18,
        "coverage": 0.0
      }
    },
    "warnings": [
      "possible_bio_overlap"
    ]
  },
  "window_results": [],
  "debug": {}
}
```

`observations.weather` is the aggregator-facing contract. It follows the
project-wide analysis spec: E-B reports continuous 0-1 summaries for wind,
rain, and thunder. The legacy `weather` block is retained during MVP
development for current tests, debug pages, and gate calibration scripts.

## Labels

Allowed `overall_label` values:

- `none`
- `rain`
- `wind`
- `thunder`
- `rain+wind`
- `rain+thunder`
- `wind+thunder`
- `rain+thunder+wind`

The overall label is derived from the elements marked `present: true`. If no
element is present, `overall_label` must be `none` and `none` must be `true`.

## Elements

The MVP tracks exactly three weather elements:

- `rain`
- `wind`
- `thunder`

The compatibility `weather.elements` block contains:

| Field | Type | Meaning |
|---|---|---|
| `present` | boolean | Whether this element is clearly detected. |
| `intensity` | string | One of `none`, `light`, `medium`, `heavy`. |
| `confidence` | number | Calibrated confidence in `[0, 1]`. |
| `coverage` | number | Fraction of analysis windows where the element passes its presence threshold. |

If `present` is false, `intensity` must be `none`.

Composite weather labels are derived from element-level presence. For example,
if rain and wind are present but thunder is absent, the top-level label is
`rain+wind`. E-B does not claim to source-separate mixed weather; confidence,
coverage, and warnings communicate ambiguity.

## Aggregator Weather Observations

The compulsory E-B observation contract is:

```json
{
  "observations": {
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
    }
  }
}
```

Summary fields:

| Field | Type | Meaning |
|---|---|---|
| `intensity` | number | Continuous weather-element magnitude in `[0, 1]`. |
| `variability` | number | Fluctuation over analysis windows: steady `0`, highly variable `1`. |
| `coverage` | number | Fraction of the clip where the element is audibly present. |
| `label` | string | Derived bucket from intensity: `none`, `light`, `moderate`, or `heavy`. |
| `confidence` | number | Confidence in the element summary. |

For MVP, E-B may be summary-only. Wind/rain `segments` and thunder `events` are
optional advanced timeline fields; consumers must handle their absence.

## Warnings

Warnings do not change the label by themselves. They explain why confidence may
be lower or why the output should be read carefully.

Allowed MVP warning values:

- `low_confidence`
- `possible_bio_overlap`
- `possible_human_or_machine_overlap`
- `possible_wind_overload`
- `possible_rain_under_wind`
- `rain_confirmed_without_beats_guard`
- `possible_clipping`
- `weather_mixed_with_ambient`
- `short_audio`
- `unsupported_sample_rate_resampled`
- `model_scores_unavailable`
- `audioset_scores_unavailable`

## Window Results

For audio longer than one analysis window, the implementation should include
per-window evidence:

```json
{
  "start_s": 0.0,
  "end_s": 5.0,
  "scores": {
    "rain": 0.72,
    "wind": 0.48,
    "thunder": 0.10,
    "none": 0.18,
    "bio_contamination": 0.22,
    "human_machine_contamination": 0.08
  },
  "model_scores": {
    "available": false,
    "backend": "clap",
    "scores": {
      "rain": 0.0,
      "wind": 0.0,
      "thunder": 0.0,
      "none": 0.0,
      "bio_contamination": 0.0,
      "human_machine_contamination": 0.0
    },
    "raw": {}
  },
  "audioset_scores": {
    "available": false,
    "backend": "audioset_unavailable",
    "scores": {
      "rain": 0.0,
      "wind": 0.0,
      "thunder": 0.0,
      "bio_contamination": 0.0,
      "human_machine_contamination": 0.0
    },
    "raw": {}
  },
  "guard_scores": {
    "available": false,
    "backend": "ast",
    "scores": {
      "rain": 0.0,
      "wind": 0.0,
      "thunder": 0.0,
      "bio_contamination": 0.0,
      "human_machine_contamination": 0.0
    },
    "raw": {}
  },
  "feature_scores": {
    "rain": 0.12,
    "wind": 0.08,
    "thunder": 0.04,
    "none": 0.45,
    "bio_contamination": 0.0,
    "human_machine_contamination": 0.0
  },
  "features": {
    "rms_dbfs": -28.4,
    "peak_dbfs": -6.1,
    "clipping_ratio": 0.0,
    "spectral_centroid_hz": 3100.0,
    "spectral_flatness": 0.42,
    "spectral_entropy": 0.71,
    "low_20_700_ratio": 0.88,
    "high_2000_8000_ratio": 0.21
  },
  "warnings": []
}
```

Window results are optional for production UI display but useful for audit and
threshold tuning.
