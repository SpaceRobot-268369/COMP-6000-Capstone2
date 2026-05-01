# Layer Verification & Handoff Formats — Modules A, B, C, D

## Overview

Two distinct concerns:

1. **Scrutinization** — verifying each layer in isolation before trusting it downstream.
   Each module writes a self-contained inspection bundle (WAV + visual + JSON trace).
2. **Synthesis handoff** — once verified, passing layer outputs to Module D as a `LayerBundle`.
   Audio lives in memory as numpy arrays; no disk round-tripping at runtime.

---

## Part 1 — Scrutinization Output Per Layer

### Module A — Ambient Bed

```json
{
  "layer": "ambient",
  "audio_path": "debug/layer_a_bed.wav",
  "spectrogram_path": "debug/layer_a_spec.png",
  "retrieved_clips": [
    {
      "clip_id": "1234567_012",
      "cosine_similarity": 0.912,
      "env": { "temp_c": 18.2, "wind_ms": 1.4, "hour": 6 }
    }
  ],
  "requested_env": { "temp_c": 19.0, "wind_ms": 1.2, "hour": 6 },
  "top_k": 5,
  "blend_weights": [0.35, 0.25, 0.20, 0.12, 0.08]
}
```

**What to check:**
- Retrieved clips are from the correct diel bin and season.
- Cosine similarity scores are plausible (> 0.7 is good for MVP).
- WAV sounds like the expected ambient texture for those conditions.

---

### Module B — Weather

```json
{
  "layer": "weather",
  "audio_path": "debug/layer_b_weather.wav",
  "spectrogram_path": "debug/layer_b_spec.png",
  "wind": {
    "intensity_label": "light",
    "asset_used": "wind/light/clip_003.wav",
    "gain_db": -6.0,
    "highpass_hz": 200
  },
  "rain": {
    "intensity_label": "none",
    "asset_used": null,
    "gain_db": null
  },
  "input_params": { "wind_speed_ms": 3.1, "precipitation_mm": 0.0 }
}
```

**What to check:**
- Intensity label matches the lookup table (wind: none <2, light 2–6, moderate 6–10, strong ≥10 m/s; rain: none=0, light 0–2, moderate 2–5, heavy ≥5 mm).
- Audio character matches the label (light wind should sound light).
- No clipping or artefacts.

---

### Module C — Events

```json
{
  "layer": "events",
  "audio_path": "debug/layer_c_events.wav",
  "timeline_path": "debug/layer_c_timeline.png",
  "events": [
    {
      "label": "Pied Butcherbird",
      "onset_s": 12.4,
      "offset_s": 15.1,
      "confidence": 0.87,
      "snippet_path": "data/events/event_snippets/pied_butcherbird_003.wav",
      "selection_reason": "dawn bin, temp 18°C, site 257 annotation match"
    }
  ],
  "scheduling_params": {
    "season": "winter",
    "diel_bin": "dawn",
    "target_duration_s": 120
  }
}
```

**What to check:**
- Events are ecologically plausible for the given season, diel bin, and conditions.
- All onset/offset values fall within `[0, target_duration_s]`.
- Snippets sound like the labelled species.
- No temporal overlap between events (unless intentional).

---

## Part 2 — Synthesis Handoff to Module D

### `LayerResult` dataclass

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class LayerResult:
    audio: np.ndarray   # float32, shape (n_samples,), 22050 Hz, range [-1, 1]
    sample_rate: int    # always 22050 — enforced, not optional
    gain_db: float      # layer's recommended gain in the final mix
    metadata: dict      # feeds the explanation JSON in Module D
```

### `LayerBundle` — what Module D receives

```python
@dataclass
class LayerBundle:
    ambient: LayerResult        # Module A
    weather: LayerResult        # Module B — zeros array if no weather
    events: LayerResult         # Module C — sparse array (silence between events)
    target_duration_s: float
    requested_env: dict         # original user request, passed through for explanation
```

**Invariants Module D can rely on:**
- All three `audio` arrays have the same `sample_rate` (22050 Hz).
- All three arrays have the same length (`target_duration_s * 22050` samples).
- `weather.audio` and `events.audio` may be all-zeros (no weather / no events scheduled) — D must handle this gracefully without branching.
- `gain_db` is a *recommendation*; D still applies final gain staging and normalisation before export.

### Starting gain values (tune during mix stage)

| Layer | `gain_db` starting point |
|-------|--------------------------|
| A — Ambient | −3 dB |
| B — Weather | −6 dB |
| C — Events  | −9 dB |

---

## Summary Table

| Layer | Scrutinization artifacts | Runtime handoff |
|-------|--------------------------|-----------------|
| A | WAV + mel spec PNG + JSON (clips + similarity scores) | `LayerResult(audio, gain_db=-3, metadata={clips, weights})` |
| B | WAV + mel spec PNG + JSON (asset selection trace) | `LayerResult(audio, gain_db=-6, metadata={wind, rain})` |
| C | WAV + timeline PNG + JSON (event list with onsets) | `LayerResult(audio, gain_db=-9, metadata={events})` |
| D input | — | `LayerBundle(ambient, weather, events, duration, env)` |

---

## Design Notes

- **Module D is a deterministic function.** It never re-runs inference — it only mixes. This hard separation means bugs in D cannot taint per-module debugging.
- **Scrutinization bundles write to `debug/`.** They are never loaded at runtime; they exist only for inspection during development and verification.
- **`metadata` in each `LayerResult` maps directly to the explanation JSON fields** defined in `pipeline_design.md` (`ambient_source_clips`, `weather_layers`, `event_layers`, etc.). D assembles the final explanation by merging all three `metadata` dicts.
