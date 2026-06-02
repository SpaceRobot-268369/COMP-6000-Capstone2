# Layer B Weather Retrieval API Contract

## Purpose

This document defines the suggested API interface for calling Layer B weather retrieval from the backend or frontend.

Layer B provides semantic weather asset retrieval using pretrained CLAP embeddings.

## Python Entry Point

```python
from acoustic_ai.modules.weather.retriever import retrieve_weather_asset

results = retrieve_weather_asset(
    query_text="strong forest wind ambience",
    weather_type="wind",
    top_k=3,
)
```

## Suggested HTTP Endpoint

```http
GET /api/layer-b/weather/retrieve
```

## Query Parameters

| Name | Type | Required | Example |
|---|---|---|---|
| query | string | yes | strong forest wind ambience |
| type | string | yes | wind |
| top_k | integer | no | 3 |

Allowed `type` values:

```text
wind
rain
thunder
```

## Example Request

```http
GET /api/layer-b/weather/retrieve?query=strong%20forest%20wind%20ambience&type=wind&top_k=3
```

## Example Success Response

```json
{
  "ok": true,
  "query": "strong forest wind ambience",
  "type": "wind",
  "top_k": 3,
  "results": [
    {
      "file": "/workspace/acoustic_ai/data/weather/wind/wind_trees_rustling.wav",
      "score": 0.503,
      "weather_type": "wind",
      "query": "strong forest wind ambience"
    }
  ]
}
```

## Example Error Response

```json
{
  "ok": false,
  "error": "Weather embedding index not found"
}
```

## Notes

- Backend should call the Python function, not CLAP directly.
- Frontend should call the backend endpoint only.
- Current MVP supports curated weather assets only.
- Generated embedding indexes should not be committed to git.

## Segment Selection Endpoint

Layer B should also expose a segment-selection interface for Layer D.

This endpoint returns metadata only. It does not render audio, mix layers, or
place clips on the final timeline.

```http
POST /layer_b/select_segments
```

Example request:

```json
{
  "query": "moderate forest rain with distant thunder",
  "weather_types": ["rain", "thunder"],
  "wind_speed_ms": 1.0,
  "precipitation_mm": 3.0,
  "include_thunder": true,
  "target_duration": 60,
  "top_assets": 3,
  "segments_per_type": 2,
  "window_seconds": 10,
  "overlap_seconds": 2
}
```

Example response shape:

```json
{
  "ok": true,
  "query": "moderate forest rain with distant thunder",
  "target_duration": 60,
  "weather_types": ["rain", "thunder"],
  "results": [
    {
      "weather_type": "rain",
      "file": "acoustic_ai/data/weather/rain/forest_rain_canopy.wav",
      "score": 0.53,
      "retrieval_score": 0.48,
      "segment": {
        "start_time": 12.0,
        "duration": 10.0,
        "fade_in": 1.0,
        "fade_out": 1.0,
        "role": "base_rain"
      },
      "validation": {
        "quality_score": 0.05,
        "silence_ratio": 0.01,
        "clipping_ratio": 0.0,
        "stability": 0.82
      },
      "reason": "CLAP-matched rain asset with stable texture suitable for Layer D."
    }
  ],
  "warnings": [],
  "layer_d_contract": {
    "layer_b_selects": "weather asset files and candidate segment metadata",
    "layer_d_owns": "timeline placement, crossfades, gain staging, and final mix"
  }
}
```

If local WAV files are missing, Layer B may still return CLAP-matched asset
metadata with a validation warning, but real segment validation requires the
weather assets to exist on disk.
