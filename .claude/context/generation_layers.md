# Generation Mode — Layer Design

Summarised from `ai_mvp_decision_log_and_new_architecture.md`.

## Soundscape Equation

```
speculative soundscape = ambient site bed
                       + weather layer
                       + biological/event layer
                       + final mix and explanation
```

---

## Layer A — Ambient Site Bed

**Purpose:** continuous ecoacoustic background texture (insects, low-level ambience, site tone).

**MVP implementation:** retrieval-first.
- Given user env conditions, find similar real clips by env vector cosine similarity.
- Use retrieved audio directly as the bed.
- VAE reconstruction is a secondary option only when retrieval quality is insufficient.

**Code:** `modules/ambient/retrieval.py` [PLACEHOLDER]
**Data:** `data/module_a/latents/latent_clips.npy` (5,318 per-clip latents + env vectors)

---

## Layer B — Weather Sound Engine

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
**Data:** `data/module_b/weather_assets/wind/{none,light,moderate,strong}/` and `rain/{none,light,moderate,heavy}/`

---

## Layer C — Species and Annotated Event Layer

**Purpose:** add biologically meaningful events plausible for the requested env/time context.

**Pre-condition:** annotation audit must complete before event model design.
Run `modules/events/annotation_audit.py` and review `data/module_c/annotation_label_report.md`.

**Decision gates after audit:**

| Audit result | Layer C design |
|---|---|
| Many reliable species labels | Train species/event classifier |
| Few species labels, many BirdNET events | BirdNET pseudo-label retrieval |
| Mostly score-only events | Binary activity layer only |
| Labels highly imbalanced | Retrieval + rule-based scheduling |

**Code:** `modules/events/annotation_audit.py`, `modules/events/event_index.py`, `modules/events/scheduler.py` [PLACEHOLDERS]
**Data:** `data/module_c/annotation_event_index.csv`, `data/module_c/event_snippets/`

---

## Layer D — Mixer and Output Explanation

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

## MVP Build Priority

1. Ambient retrieval function (Layer A) — highest immediate audio realism
2. Mixer/export pipeline (Layer D) — needed to combine later layers
3. Wind/rain asset curation + mixing (Layer B) — direct env → audio link
4. Annotation audit and event index (Layer C prerequisite)
5. Event planner and scheduler (Layer C)
6. Optional: VAE reconstruction/transformation for Layer A variation
