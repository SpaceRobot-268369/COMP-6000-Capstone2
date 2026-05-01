# Pipeline Design — Generation & Analysis Modes

---

## Generation Mode — Layer Design

### Soundscape Equation

```
speculative soundscape = ambient site bed
                       + weather layer
                       + biological/event layer
                       + final mix and explanation
```

---

### Layer A — Ambient Site Bed

**Purpose:** continuous ecoacoustic background texture (insects, low-level ambience, site tone). **Must not contain events** — bird calls, vehicles, helicopters belong in Layer C, weather in Layer B. Mixing events into the bed double-counts them and breaks layer separation.

**MVP implementation:** retrieval-first over a *cleaned ambient-only segment pool*.

Stage 1 — offline data cleaning (precompute), **audio-only and content-agnostic**:

Two principles drive the design:

1. **Annotations are unreliable as a negative signal.** Sparse coverage means *absence of annotation does not mean absence of event*. The only trustable signal is the audio itself.
2. **Events are an open class — ambient is locally stationary.** We cannot enumerate every event type (birds, helicopters, vehicles, branch snaps, frogs, distant voices, unknown insects). But every event has the same property: it deviates from its own clip's stationary baseline. So the gate detects *anomalies relative to the local baseline*, not specific event categories.

Per-clip pipeline:

  1. **Frame features** along the whole clip: mel spectrogram (128 bins), RMS, spectral centroid, spectral flatness, spectral flux, zero-crossing rate.
  2. **Per-clip rolling baseline:** rolling median + MAD over a 30 s window for each feature.
  3. **Anomaly mask:** mark a frame anomalous if *any* feature deviates > 3·MAD from its rolling median. (3·MAD ≈ 3σ but robust to outliers — the events themselves.)
  4. **Dilation:** extend each anomalous frame by ± 0.5 s so partial onsets/offsets are masked.
  5. **Contiguous spans:** invert the mask, find unmasked stretches, **keep only spans ≥ 20 s**.
  6. **Span verification:** within each kept span, require RMS within [p20, p80] of the clip *and* low frame-to-frame mel variance (re-confirms the span is itself stationary, not just non-anomalous on average).
  7. **Slice** each verified span into segments of **20–60 s** (target 30 s). Long segments mean the runtime crossfade is at most ~1 s and inaudible — generation does not depend on stitch quality.

Outputs: `data/ambient/ambient_segments/*.wav` plus `ambient_index.csv` with columns (`segment_id`, `source_clip`, `t_start`, `t_end`, `diel_bin`, `season`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`).

**Validation, not gating:** A2O annotations and BirdNET are run *over retained segments* as audits, not as gates. Targets: <1% of retained-segment-seconds overlap any annotated event; BirdNET fire-rate above conf 0.3 stays below an acceptable threshold (tune-up signal, not pass/fail). If either audit fails, tighten MAD threshold and re-run.

**Why no neural detector in the gate:** BirdNET only knows species in its training set, is deaf to everything else, and is the wrong failure mode for a permissive gate. The 3·MAD anomaly check is content-agnostic, self-calibrating per clip (cicada-rich clips get a higher noise floor automatically), and runs on CPU in minutes for the whole 6,148-clip pool.

Stage 2 — runtime retrieval:
- **Hard filter:** restrict to segments matching the requested `diel_bin` and `season`. Categorical mismatch sounds wrong regardless of numeric proximity.
- **Soft rank:** cosine similarity on `[hour_sin, hour_cos, month_sin, month_cos]` only — the four features that describe time/seasonal *position* within the bin. Take top-k=5.
- **Blend:** `blend_weights = softmax(sim / τ)` with τ=0.1, crossfade-mix the k segments, RMS-match, tile/loop to `target_duration_s`.
- If hard filter returns <k segments, relax to neighbouring diel bin and flag `low_confidence: true` in metadata.

**Why temp/humidity/wind/rain are excluded from retrieval:** wind and rain are direct acoustic signals owned by Layer B; temperature and humidity affect species/insect *behaviour* and so flow through Layer C. The ambient bed itself — site tone, low-level texture — is driven by time of day and seasonal position, not weather. Including weather variables in the Layer A key would either double-count (with B) or pull in irrelevant similarity (with C).

VAE reconstruction is **not** part of the MVP path for Layer A — the existing VAE was trained on event-contaminated full clips, so its latent space mixes ambient with events and is the wrong tool for this layer. Keep VAE for transformation mode and Module E analysis.

**Code:** `modules/ambient/retrieval.py` [PLACEHOLDER]
**Data:** `data/ambient/ambient_segments/`, `data/ambient/ambient_index.csv` (cleaned ambient-only pool — to be built)
**Legacy data:** `data/ambient/latents/latent_clips.npy` (5,318 per-clip latents over uncleaned clips — retained for transformation/analysis, not used for Layer A retrieval)

---

### Layer B — Weather Sound Engine

**Purpose:** generate or mix direct weather sounds (wind, rain).

**MVP implementation:** curated asset library + parameter-controlled mixing.

| Condition | Behaviour |
|---|---|
| `wind_speed_ms < 2` | No wind layer |
| `2 ≤ wind_speed_ms < 6` | Light wind, high-pass filtered |
| `6 ≤ wind_speed_ms < 10` | Moderate wind |
| `wind_speed_ms ≥ 10` | Strong wind, more low-mid energy |
| `precipitation_mm == 0` | No rain layer |
| `0 < precipitation_mm < 2` | Sparse light rain |
| `2 ≤ precipitation_mm < 5` | Moderate rain |
| `precipitation_mm ≥ 5` | Dense heavy rain |

**Code:** `modules/weather/asset_index.py`, `modules/weather/mixer.py` [PLACEHOLDERS]
**Data:** `data/weather/weather_assets/wind/{none,light,moderate,strong}/` and `rain/{none,light,moderate,heavy}/`

---

### Layer C — Species and Annotated Event Layer

**Purpose:** add biologically meaningful events plausible for the requested env/time context.

**Pre-condition:** annotation audit must complete before event model design.
Run `modules/events/annotation_audit.py` and review `data/events/annotation_label_report.md`.

**Decision gates after audit:**

| Audit result | Layer C design |
|---|---|
| Many reliable species labels | Train species/event classifier |
| Few species labels, many BirdNET events | BirdNET pseudo-label retrieval |
| Mostly score-only events | Binary activity layer only |
| Labels highly imbalanced | Retrieval + rule-based scheduling |

**Code:** `modules/events/annotation_audit.py`, `modules/events/event_index.py`, `modules/events/scheduler.py` [PLACEHOLDERS]
**Data:** `data/events/annotation_event_index.csv`, `data/events/event_snippets/`

---

### Layer D — Mixer and Output Explanation

**Purpose:** combine A+B+C into one coherent audio file and produce an explanation.

**Mixer responsibilities:**
- Match sample rate (22,050 Hz throughout)
- Trim or loop layers to requested duration
- Apply fades (avoid clicks at layer boundaries)
- Control gain staging (avoid clipping)
- Optionally apply light per-layer EQ
- Export final WAV
- Generate mel spectrogram preview (PNG base64)
- Return explanation JSON

**Explanation JSON fields:**

| Field | Meaning |
|---|---|
| `ambient_source_clips` | Real clips used as the background bed |
| `weather_layers` | Wind/rain assets used and intensity mapping |
| `event_layers` | Species/events selected and ecological reason |
| `env_match_score` | Similarity between request and retrieved clips |
| `limitations` | Notes about speculative nature and dataset gaps |

**Code:** `modules/mixer/audio_mixer.py` [PLACEHOLDER]

---

### MVP Build Priority

1. Ambient retrieval function (Layer A) — highest immediate audio realism
2. Mixer/export pipeline (Layer D) — needed to combine later layers
3. Wind/rain asset curation + mixing (Layer B) — direct env → audio link
4. Annotation audit and event index (Layer C prerequisite)
5. Event planner and scheduler (Layer C)
6. Optional: VAE reconstruction/transformation for Layer A variation

---

## Analysis Mode — Component Design

### Pipeline

```
Uploaded audio clip
    → Preprocessing (mel spectrogram + waveform features)
    → Component A: Ambient similarity encoder
    → Component B: Weather detector
    → Component C: Species/event detector
    → Analysis report (estimated conditions + layer breakdown + confidence)
```

---

### Component A — Ambient Similarity Encoder

**Purpose:** locate the uploaded clip in soundscape space; estimate broad context
(season, diel bin, similar training recordings, plausible env ranges).

**MVP implementation:** VAE latent nearest-neighbour (already working in `inference.py`).
- `encode_clip()` → latent `mu` (256-dim)
- Compare against `data/ambient/latents/latent_clips.npy`
- Average top-k neighbours → estimated env conditions

Optional enhancement: add acoustic indices (ACI, entropy, spectral centroid) as supporting evidence.

**Code:** `inference.py` — `encode_clip()`, `estimate_env_conditions()`

---

### Component B — Weather Detector

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
**Shared data:** `data/weather/asset_index.csv` (weather intensity labels)

---

### Component C — Species and Event Detector

**Purpose:** identify biologically meaningful events in the uploaded clip.

**Output:**
```json
[
  {"label": "str", "confidence": 0.0–1.0, "onset_s": float, "offset_s": float}
]
```

**MVP implementation:** BirdNET pseudo-labels or high-confidence A2O annotations.
- Run BirdNET over the uploaded audio.
- Cross-reference with `data/events/annotation_event_index.csv` where available.
- Use existing A2O annotations as validation or overrides.

**Code:** `modules/analysis/event_detector.py` [PLACEHOLDER]
**Shared data:** `data/events/annotation_event_index.csv`

---

### Analysis Report Fields

| Field | Source |
|---|---|
| `estimated_conditions` | Component A — top-k NN average |
| `similar_clips` | Component A — top-k clip IDs and similarity scores |
| `wind_intensity`, `rain_intensity` | Component B |
| `detected_events` | Component C |
| `confidence` | Per-component confidence scores |
| `limitations` | Notes on model limitations and dataset coverage |
