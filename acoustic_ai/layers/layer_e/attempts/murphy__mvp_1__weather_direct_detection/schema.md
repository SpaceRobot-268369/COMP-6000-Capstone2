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
  "weather": {
    "overall_label": "rain+wind",
    "none": false,
    "elements": {
      "rain": {
        "present": true,
        "intensity": "medium",
        "confidence": 0.78
      },
      "wind": {
        "present": true,
        "intensity": "light",
        "confidence": 0.62
      },
      "thunder": {
        "present": false,
        "intensity": "none",
        "confidence": 0.18
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

Each element must contain:

| Field | Type | Meaning |
|---|---|---|
| `present` | boolean | Whether this element is clearly detected. |
| `intensity` | string | One of `none`, `light`, `medium`, `heavy`. |
| `confidence` | number | Calibrated confidence in `[0, 1]`. |

If `present` is false, `intensity` must be `none`.

## Warnings

Warnings do not change the label by themselves. They explain why confidence may
be lower or why the output should be read carefully.

Allowed MVP warning values:

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
