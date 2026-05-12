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

**Current implementation under validation:** AudioLDM2 LoRA generation. Current working checkpoint: `acoustic_ai/checkpoints/audioldm2-lora-raw-smoke`, with base model `cvssp/audioldm2`. User validation on 2026-05-06: works well for quiet environmental ambience with only minor issues. Keep generated beds low-volume, mostly stationary, and event-free; sample with low guidance around `2.0` and the fixed smoke-test prompt that excludes foreground events, music, and machinery. Because the smoke dataset is tiny, the dev frontend/backend path must not accept user prompts yet: expose only a non-negative integer seed, and let FastAPI own the prompt/checkpoint/settings. Different seeds produce different variations from the same model/prompt/settings; same seed should reproduce effectively the same result on the same code path. Seed is not temperature, and temperature is not exposed. Do not use the deprecated `audioldm2-lora-rms005-smoke` checkpoint for quality testing. CLI and frontend Layer A spectrograms should use the shared log-mel renderer in `modules.ambient.diffusion.layer_a_visualization`.

**Branch note:** this is one attempted Layer A implementation that succeeded for the smoke test on this branch. If merged into `main`, align all architecture, pipeline, and handoff docs so AudioLDM2 LoRA is described consistently as the main Layer A path.

**Previous MVP implementation:** retrieval-first over a *cleaned ambient-only segment pool*.

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

**Approach: Generative — AudioGen LoRA, per species (and optionally per diel/seasonal context).**

We fine-tune `facebook/audiogen-medium` with LoRA adapters on per-species snippet manifests built from A2O / BirdNET annotations. AudioGen is chosen over AudioLDM2 for events specifically because its token-based EnCodec representation preserves transients (the leading edge of a call), its training corpus already contains AudioSet animal/environmental labels, and its native operating range matches event clip durations. Retrieval and DSP variation are kept as fallbacks if a given species LoRA fails the smoke-test bar.

**Pre-condition:** annotation audit must complete before training any LoRA.
Run `modules/events/annotation_audit.py` and review `data/events/annotation_label_report.md` to produce per-species manifests filtered by score, duration, and diel context.

**Per-species selection policy (apply in order):**

| Filter | Rule |
|---|---|
| 1. Species | `common_name_tags` or `other_tags` matches target species |
| 2. Confidence | `score ≥ 0.85` (BirdNET imports only; manual Raven annotations always kept) |
| 3. Duration | `1.0 ≤ event_duration_seconds ≤ 6.0` |
| 4. Diel context | event-start hour falls in the target diel bin (e.g. 22:00–05:00 AEST for nocturnal species) |
| 5. Year | exclude 2021–2022 (A2O archive gap); prefer 2023–2024 |
| 6. Per-recording cap | ≤ 3 snippets per `audio_recording_id` to avoid overfitting to one individual |
| 7. Random sample | seed=42, sample ~150 candidates |
| 8. Manual audit | listen + reject overlap, wind, multi-species → target **40–80 keepers** |

**Smoke test (single LoRA, Southern Boobook nocturnal):**

| Setting | Value |
|---|---|
| Base model | `facebook/audiogen-medium` |
| Dataset size | 60–100 audited clips, 16 kHz mono, 3–6 s, single isolated calls |
| Captions | short natural language, varied (e.g. "Southern Boobook owl two-note call at night, distant", "close Boobook call quiet woodland night") |
| LoRA rank `r` | 8 (alpha 16); target modules `q_proj`, `v_proj` |
| Epochs | 10–15 |
| Batch size | 1 (gradient accumulation 4–8) |
| Learning rate | 1e-4 (warmup ~50 steps) |
| Precision | fp16 on CUDA, fp32 on MPS (MPS fp16 unreliable) |
| Inference duration | 3–5 s |
| Sampling | top-k 250, temperature 1.0, CFG 3.0 |
| Seeds for audit | 42–51 (10 seeds; higher rejection rate than ambient → cherry-pick) |
| Output checkpoint | `acoustic_ai/checkpoints/audiogen-lora-boobook-smoke/` |
| Output samples | `debug/layer_c/audiogen/samples/audiogen-lora-boobook-smoke/boobook_smoke_seed{42..51}/` |

**Pass/fail criterion (smoke):** at least 4 of 10 seeds produce a clip in which the two-note "boo-book" structure is identifiable in the first 3 s with no obvious EnCodec warble; end-to-end training + sampling completes without intervention.

**MVP scope:** 5–15 per-species LoRAs across diel bins (target ~10 species covering ~95% of high-confidence detections at this site). Storage budget ≈ 50 MB per LoRA → ~500 MB on DVC.

**Sample-rate boundary:** AudioGen output is **16 kHz mono**. Module D mixer must resample 16 → 22,050 Hz before overlaying on the ambient bed. Apply a fixed event-layer attenuation (≈ −12 dB) at the mixer because AudioGen output is naturally hotter than the ambient bed.

**Prompt style:** AudioGen responds to short, declarative captions — not to AudioLDM2-style "no X, no Y" negation lists. Use negative prompts via classifier-free guidance instead.

**Tooling:** Meta's `audiocraft` + PEFT. Use a separate environment at `acoustic_ai/.venv-audiogen` to avoid torch/torchaudio conflicts with the AudioLDM2 stack.

**Code:** `modules/events/annotation_audit.py`, `modules/events/dataset.py`, `modules/events/train_audiogen.py`, `modules/events/sample_audiogen.py`, `modules/events/scheduler.py` [PLACEHOLDERS]
**Data:** `data/events/<species>/manifest.csv` + extracted snippets per species (DVC-tracked)
**Checkpoints:** `checkpoints/audiogen-lora-<species>-<context>/` per LoRA (DVC-tracked)

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
