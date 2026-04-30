# Analysis Mode — Component Design

Summarised from `ai_mvp_decision_log_and_new_architecture.md`.

## Pipeline

```
Uploaded audio clip
    → Preprocessing (mel spectrogram + waveform features)
    → Component A: Ambient similarity encoder
    → Component B: Weather detector
    → Component C: Species/event detector
    → Analysis report (estimated conditions + layer breakdown + confidence)
```

---

## Component A — Ambient Similarity Encoder

**Purpose:** locate the uploaded clip in soundscape space; estimate broad context
(season, diel bin, similar training recordings, plausible env ranges).

**MVP implementation:** VAE latent nearest-neighbour (already working in `inference.py`).
- `encode_clip()` → latent `mu` (256-dim)
- Compare against `data/module_a/latents/latent_clips.npy`
- Average top-k neighbours → estimated env conditions

Optional enhancement: add acoustic indices (ACI, entropy, spectral centroid) as supporting evidence.

**Code:** `inference.py` — `encode_clip()`, `estimate_env_conditions()`

---

## Component B — Weather Detector

**Purpose:** detect audible wind and rain intensity in the uploaded clip.

**Output:**
```json
{
  "wind_intensity": "none | light | moderate | strong",
  "rain_intensity": "none | light | moderate | heavy",
  "confidence": 0.0–1.0
}
```

**MVP implementation:** start with curated labels (same clips used for Layer B asset library).
- Manually tag a small set of clips with wind/rain intensity.
- Use spectral heuristics first (broadband energy, low-freq modulation, high-freq texture).
- Upgrade to a small classifier after labels accumulate.

**Code:** `modules/analysis/weather_detector.py` [PLACEHOLDER]
**Shared data:** `data/module_b/asset_index.csv` (weather intensity labels)

---

## Component C — Species and Event Detector

**Purpose:** identify biologically meaningful events in the uploaded clip.

**Output:**
```json
[
  {"label": "str", "confidence": 0.0–1.0, "onset_s": float, "offset_s": float}
]
```

**MVP implementation:** BirdNET pseudo-labels or high-confidence A2O annotations.
- Run BirdNET over the uploaded audio.
- Cross-reference with `data/module_c/annotation_event_index.csv` where available.
- Use existing A2O annotations as validation or overrides.

**Code:** `modules/analysis/event_detector.py` [PLACEHOLDER]
**Shared data:** `data/module_c/annotation_event_index.csv`

---

## Analysis Report Fields

| Field | Source |
|---|---|
| `estimated_conditions` | Component A — top-k NN average |
| `similar_clips` | Component A — top-k clip IDs and similarity scores |
| `wind_intensity`, `rain_intensity` | Component B |
| `detected_events` | Component C |
| `confidence` | Per-component confidence scores |
| `limitations` | Notes on model limitations and dataset coverage |
