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