# Pipeline Design — Generation & Analysis Modes

> **Layout note (post-restructure):** code lives under
> `acoustic_ai/layers/layer_<X>/attempts/<member>__<stage>__<slug>/`.
> Available attempts are declared in `acoustic_ai/registry.yaml` and
> served via `GET /layers` for the frontend dropdown. Naming rules:
> [../conventions.md](../conventions.md). Per-layer
> "Module" sections describe the *role*; concrete implementations live
> across one or more attempts per layer.



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

### Prompt Parsing Layer (Parser)

**Purpose:** Turn a raw user prompt into complete, validated, layer-specific inputs before routing them to individual layers.

> Canonical policy: [prompt_parser_policy.md](prompt_parser_policy.md). This
> section is the overview; the policy owns the default table, the
> block/suggest gate rules, and the parse-result schema.

The **Prompt Parser** is **LLM-OSS powered** (the generation-side mirror of the Layer E report writer) and governed by a written policy — not a single regex pass. A lightweight rule-based fast-path may short-circuit trivially unambiguous prompts, but the LLM + policy is the source of truth. It runs **three stages**:

1. **Pre-process & default-fill** — normalise the prompt and supply explicit defaults for anything left unspecified. Ambient (Layer A) is always on; **weather (Layer B) is off by default — no rain unless requested**; events (Layer C) start as an empty checklist. Silence is a recorded decision, not a gap.
2. **Validity / coherence gate** — reject requests our site/models can't voice (e.g. dense city noise over a remote dry-woodland site, climatically implausible weather, fauna that doesn't occur in the requested season). Wherever possible the parser *corrects and continues* — it rewrites the prompt and explains the swap — rather than hard-failing.
3. **Decode into layer contracts** — only a complete, validated request is translated into the three aligned inputs below.

Because we divide soundscape generation into independent, modular layers, we cannot feed a raw user prompt directly into the model of each layer. After Stage 3 each downstream model receives exactly what its API expects:

1. **Layer A (Ambient Bed) Input**:
   - For cell-based generative banks: Resolves keywords/context to a valid `(season, diel)` cell tuple (e.g., `"autumn_dawn"`).
   - For open-prompt generative models: Rewrites the prompt to focus strictly on background texture (e.g. insects, low-level wind/foliage, distant birds) while stripping out specific dynamic foreground events (like loud vehicles or specific bird calls).
2. **Layer B (Weather) Input**:
   - Because Layer B is a retrieval-based asset mixing engine, it requires structured JSON query parameters: `weather_type` (`rain`, `wind`, etc.), `intensity` (`light`, `medium`, `heavy`), and `duration_s`. The decoder extracts these from weather descriptors in the user prompt (e.g. `"pouring rain"` -> `weather_type: rain, intensity: heavy`).
3. **Layer C (Events/Species) Input**:
   - Resolves mentions of fauna or specific events into species common names (`"splendid fairywren"`, `"boobook owl"`) and timeline density parameters.
   - Maps species to their respective generative LoRA weights or queries the audited retrieval database using structured JSON specifying species labels and target diel/season boundaries.

This parsing layer acts as the orchestrator at the front of the generation workflow, guaranteeing that each downstream model receives inputs that are complete, in-domain, and matching its specific API contract.

---

### Layer A — Ambient Site Bed

**Purpose:** continuous ecoacoustic background texture (insects, low-level ambience, site tone). **Must not contain events** — bird calls, vehicles, helicopters belong in Layer C, weather in Layer B. Mixing events into the bed double-counts them and breaks layer separation.

**Current implementation under validation:** AudioLDM2 LoRA generation. Current working checkpoint: `model/candidates/lucas/layer-a-audioldm2-raw-smoke`, with base model `cvssp/audioldm2`. User validation on 2026-05-06: works well for quiet environmental ambience with only minor issues. Keep generated beds low-volume, mostly stationary, and event-free; sample with low guidance around `2.0` and the fixed smoke-test prompt that excludes foreground events, music, and machinery. Because the smoke dataset is tiny, the dev frontend/backend path must not accept user prompts yet: expose only a non-negative integer seed, and let FastAPI own the prompt/checkpoint/settings. Different seeds produce different variations from the same model/prompt/settings; same seed should reproduce effectively the same result on the same code path. Seed is not temperature, and temperature is not exposed. Do not use the deprecated `audioldm2-lora-rms005-smoke` checkpoint for quality testing. CLI and frontend Layer A spectrograms should use the shared log-mel renderer in `modules.ambient.diffusion.layer_a_visualization`.

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

**Code:** `layers/layer_a/attempts/lucas__smoke_4__vae_baseline/retrieval.py` [PLACEHOLDER]
**Data:** `data/ambient/ambient_segments/`, `data/ambient/ambient_index.csv` (cleaned ambient-only pool — to be built)
**Legacy data:** `data/ambient/latents/latent_clips.npy` (5,318 per-clip latents over uncleaned clips — retained for transformation/analysis, not used for Layer A retrieval)

---

### Layer B — Weather Sound Engine

**Purpose:** generate or mix direct weather sounds (wind, rain, thunder).

**MVP implementation:** `murphy__mvp_1__weather_stem_selector` returns a short
weather-only stem. The frontend supplies `weather_type`, `intensity`,
`duration_s`, and `seed`; the handler selects from the curated asset index,
uses the seed for both asset choice and start offset, applies basic loudness
normalization, and returns WAV + metadata for Layer D.

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
| thunder / storm requested | Select thunder or storm asset from library fallback |

**Code:** `layers/layer_b/attempts/murphy__mvp_1__weather_stem_selector/code/handler.py`
**Asset index:** `layers/layer_b/attempts/lucas__smoke_1__curated_assets/data/weather/asset_index.csv`

---

### Layer C — Species and Annotated Event Layer

**Purpose:** add biologically meaningful events plausible for the requested env/time context.

**Approach: Generative — AudioGen LoRA, per species (and optionally per diel/seasonal context).**

We fine-tune `facebook/audiogen-medium` with LoRA adapters on per-species snippet manifests built from A2O / BirdNET annotations. AudioGen is chosen over AudioLDM2 for events specifically because its token-based EnCodec representation preserves transients (the leading edge of a call), its training corpus already contains AudioSet animal/environmental labels, and its native operating range matches event clip durations. Retrieval and DSP variation are kept as fallbacks if a given species LoRA fails the smoke-test bar.

**Pre-condition:** annotation audit must complete before training any LoRA.
Run `layers/layer_c/attempts/lucas__smoke_1__audiogen_boobook/annotation_audit.py` and review `layers/layer_c/attempts/lucas__smoke_1__audiogen_boobook/data/events/annotation_label_report.md` to produce per-species manifests filtered by score, duration, and diel context.

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
| Output checkpoint | `model/candidates/lucas/layer-c-audiogen-boobook-smoke/` |
| Output samples | `debug/layer_c/audiogen/samples/audiogen-lora-boobook-smoke/boobook_smoke_seed{42..51}/` |

**Pass/fail criterion (smoke):** at least 4 of 10 seeds produce a clip in which the two-note "boo-book" structure is identifiable in the first 3 s with no obvious EnCodec warble; end-to-end training + sampling completes without intervention.

**MVP scope:** 5–15 per-species LoRAs across diel bins (target ~10 species covering ~95% of high-confidence detections at this site). Storage budget ≈ 50 MB per LoRA → ~500 MB on DVC.

**Sample-rate boundary:** AudioGen output is **16 kHz mono**. Module D mixer must resample 16 → 22,050 Hz before overlaying on the ambient bed. Apply a fixed event-layer attenuation (≈ −12 dB) at the mixer because AudioGen output is naturally hotter than the ambient bed.

**Prompt style:** AudioGen responds to short, declarative captions — not to AudioLDM2-style "no X, no Y" negation lists. Use negative prompts via classifier-free guidance instead.

**Tooling:** Meta's `audiocraft` + PEFT. Use a separate environment at `acoustic_ai/.venv-audiogen` to avoid torch/torchaudio conflicts with the AudioLDM2 stack.

**Code:** `layers/layer_c/attempts/lucas__smoke_1__audiogen_boobook/annotation_audit.py`, `layers/layer_c/attempts/lucas__smoke_1__audiogen_boobook/dataset.py`, `layers/layer_c/attempts/lucas__smoke_1__audiogen_boobook/train_audiogen_lora.py`, `layers/layer_c/attempts/lucas__smoke_1__audiogen_boobook/sample_audiogen_lora.py`, `layers/layer_c/attempts/lucas__smoke_1__audiogen_boobook/scheduler.py` [PLACEHOLDERS]
**Data:** `data/events/<species>/manifest.csv` + extracted snippets per species (DVC-tracked)
**Checkpoints:** `model/candidates/<member>/layer-c-audiogen-<species>-<context>/` per LoRA (DVC-tracked)

---

### Layer D — Mixer and Output Explanation

**Purpose:** combine A+B+C into one coherent audio file and produce an explanation.

**Mixer responsibilities:**
- Match sample rate (22,050 Hz throughout)
- Trim or loop **bed** layers (ambient, continuous weather) to requested duration
- Place **discrete** clips (thunder, species calls) at caller-supplied onset times
- Apply fades (avoid clicks at layer/clip boundaries)
- Control gain staging (avoid clipping)
- Optionally apply light per-layer EQ
- Export final WAV
- Generate mel spectrogram preview (PNG base64)
- Return explanation JSON

**Multi-clip arrangement (target contract).** The MVP mixer
(`songke__mvp_1__layered_mix`) takes one stem per layer. The next iteration
(`songke__mvp_2__multi_clip_mix`, design only) accepts **lists of placed clips**:
Layer B may pass several weather clips (continuous beds vs. discrete thunder),
and Layer C may pass several event clips (multiple species, or repeated calls).
Each clip carries an explicit `onsets_s` list — repetition ("frequency") is
expanded **LLM-side** into concrete times, never reasoned about by the mixer.
Onsets may overlap (a call during thunder is natural); the mixer just sums them.
`null` onsets trigger a seeded random fallback inside Layer D (`placement_seed`,
the one case where Layer D consumes a seed). Weather *transitions* are a reserved placeholder, not
built yet. Full contract:
[`layer_d/.../songke__mvp_2__multi_clip_mix/README.md`](../../../acoustic_ai/layers/layer_d/attempts/songke__mvp_2__multi_clip_mix/README.md).

**Explanation JSON fields:**

| Field | Meaning |
|---|---|
| `ambient_source_clips` | Real clips used as the background bed |
| `weather_layers` | Wind/rain assets used and intensity mapping |
| `event_layers` | Species/events selected and ecological reason |
| `env_match_score` | Similarity between request and retrieved clips |
| `limitations` | Notes about speculative nature and dataset gaps |

**Code:** `layers/layer_d/attempts/songke__mvp_1__layered_mix/code/audio_mixer.py` (single-stem MVP, implemented) · `layers/layer_d/attempts/songke__mvp_2__multi_clip_mix/` (multi-clip contract, design only)

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

### Design principle: direct detection, no decomposer

Analysis does **not** mirror generation. A "reverse architecture" that first
decomposes the uploaded clip into ambient / weather / event stems and then
runs a detector on each stem is a trap for ecoacoustic mixtures:

- **No pre-trained decomposer exists** for this domain. Music source
  separation (Demucs, Spleeter) does not transfer — natural soundscapes are
  diffuse, overlapping, and unlike vocals+drums.
- **Training one is infeasible here:** it would require source-isolated
  ground truth (clean ambient, clean wind, clean species) which we do not
  have and cannot collect at the scale a separator needs.
- **Errors compound:** a bad separation poisons every downstream head.
- **Pre-trained detectors already work on mixed audio.** BirdNET, PANNs,
  CLAP — all were trained on real-world mixtures. They are designed to find
  their target signal *in the presence of* everything else, not after it has
  been removed.

So analysis runs **three detector heads in parallel on the raw mixture**,
each owning its own question. There is no shared "separated stem" — only
labels, embeddings, and confidences.

```
   Uploaded clip ─► Preprocess ──┬──► E-A  Ambient context (similarity / k-NN)
   (mel + waveform)   (shared)   │       "what kind of bed is this?"
                                 │
                                 ├──► E-B  Weather detector
                                 │       "wind / rain intensity?"
                                 │
                                 └──► E-C  Event detector
                                         "which species, when?"
                                              │
                                              ▼
                                      Layer E aggregator
                                   (deterministic fusion of
                                    latent context — see below)
                                              │
                                              ▼
                                       Report JSON
                              (observations + inferred_context
                                     + disagreements)
                                              │
                                              ▼
                                     LLM-OSS narration
                              (renders, does not decide — two
                               registers: analytical / immersive)
```

**Tradeoff to flag:** direct detection cannot produce a *playable* decomposed
stem — the user gets **labels + context**, not three isolated waveforms. If
the analysis UX later requires playback of "the ambient layer we extracted,"
the decision above must be revisited.

---

### Design principle: observations vs. inferences

Every analysis output is one of two kinds, and conflating them is the root of
the "which head decides the season?" confusion:

- **Observation** — something a head *directly detects*: wind/rain/thunder, a
  species call at 0:12, the acoustic character of the bed. The owning head is
  **authoritative** — no fusion, no second-guessing.
- **Inference (latent context)** — season, diel, plausible env ranges. *No
  head observes these directly.* They are latent variables **owned by nobody**
  and produced only by the aggregator's fusion step, where the *event* head
  usually carries the strongest evidence (species phenology) and the ambient
  head is a weak prior.

So a head's output below is split into what it **observes** (authoritative)
and what it **contributes as evidence** toward latent context. The fusion
rules, trust hierarchy, report-writing registers, and per-head pass standards
live in the companion doc:
[analysis_synthesis_policy.md](analysis_synthesis_policy.md).

> **Future step:** the per-head time spans (E-B segments, E-C onsets) naturally
> merge onto **one shared time axis** the aggregator assembles — driving both
> immersive narration and a future waveform-overlay UI. Land per-head spans
> first; the unified timeline comes next.

---

### Reuse from generation models

The Layer A and Layer C generation paths already load large base models.
Their **encoder halves transfer to analysis for free** — same model, no
retraining; only the decoder halves stay locked to generation.

```
   Generation path (text → audio)
   ─────────────────────────────────────────────
                   ┌─────────────┐
   "spring night"──► Text encoder│──┐
                   └─────────────┘  │
                                    ▼
                              ┌──────────┐    ┌──────────┐
                              │  joint   │───►│ Decoder  │──► audio
                              │ embedding│    │ (U-Net / │
                              └──────────┘    │   AR)    │      ✗ one-way only
                                    ▲         └──────────┘
                   ┌─────────────┐  │
   audio ─────────►│ Audio enc.  │──┘
                   └─────────────┘
                          ▲
                          │
              ┌───────────┴──────────────────────┐
              │  REUSABLE for analysis:          │
              │  audio↔text encoder gives        │
              │  zero-shot tagging + similarity  │
              │  with no further training        │
              └──────────────────────────────────┘
```

| Asset | Source | Reuse in analysis? | Why |
|---|---|---|---|
| **CLAP audio+text encoder** | inside `cvssp/audioldm2` (Layer A base) | ✅ as-is, no retraining | shared embedding space; audio↔text both directions; powers E-A similarity + zero-shot E-B/E-C |
| **EnCodec audio tokenizer** | inside `facebook/audiogen-medium` (Layer C base) | ✅ as-is, no retraining | discrete-token features for an event classifier; optional |
| AudioLDM2 diffusion U-Net | Layer A base | ❌ | generator only — no audio→labels path |
| AudioGen AR transformer | Layer C base | ❌ | generator only — no audio→labels path |
| **Project LoRA checkpoints** (`layer-a-audioldm2-raw-smoke`, future Layer C LoRAs) | trained on this project | ❌ | LoRA fine-tunes decoder attention; analysis path does not touch the decoder |

**Bottom line:** the *base* generation models contribute their encoders to
analysis for free; the *fine-tuned LoRAs* do not.

---

### E-A — Ambient context (similarity, not detection)

**Purpose:** locate the uploaded clip in soundscape space — estimate broad
context (season, diel bin), surface similar training recordings, and infer
plausible env ranges from the neighbours.

**Approach:** embed the clip with a pre-trained audio encoder, k-NN against
the cleaned `ambient_index.csv` segment pool (the same pool Layer A
retrieval uses in generation), and read the neighbours' env metadata.

- **Authoritative observation:** the clip's acoustic character + its nearest
  training clips (`similar_clips`).
- **Evidence toward latent context:** a **weak** season/diel prior. Because
  the bed at this site is not seasonally discriminative (spring ≈ autumn),
  E-A emits a *distribution* (neighbour votes), not a point label, plus a
  dispersion-based confidence. The aggregator treats this as a low-weight
  prior / tiebreaker — see [analysis_synthesis_policy.md](analysis_synthesis_policy.md) Rule 1.

**Output:** (distribution + dispersion confidence, **not** a point estimate)
```json
{
  "similar_clips": [{"segment_id": "seg_00417", "similarity": 0.71}],
  "context_evidence": {
    "season": {"spring": 0.30, "autumn": 0.30, "summer": 0.25, "winter": 0.15},
    "diel":   {"dawn": 0.40, "morning": 0.35, "night": 0.15, "afternoon": 0.10}
  },
  "confidence": 0.35,
  "note": "high neighbour dispersion — weak seasonal prior"
}
```

**Pre-trained model options:**

| Model | Type | What it gives | Notes / tradeoff |
|---|---|---|---|
| **LAION-CLAP** | audio+text contrastive | general embedding + free zero-shot text queries ("dry sclerophyll dawn chorus") | already loaded via AudioLDM2 — zero marginal cost |
| **Google Perch** | bioacoustic embedding | site/biome similarity; trained on iNaturalist + Xeno-canto | strongest for ecoacoustic similarity; extra dependency |
| **Project VAE** (`vae-site257-30epoch`) | site-specific latent | already on disk; matches site 257 distribution | trained on event-contaminated clips — weaker as a clean ambient embedding |

Optional supporting evidence: hand-crafted acoustic indices (ACI, entropy,
spectral centroid) — cheap to compute, useful for explainability.

**Code:** `layers/layer_e/attempts/<id>/ambient_similarity.py` [PLACEHOLDER]
**Data:** `data/ambient/ambient_index.csv` (shared with Layer A retrieval)

---

### E-B — Weather detector

**Purpose:** detect audible wind, rain, and thunder directly in the mixture.
These have stable spectral signatures (broadband low-freq for wind, dense
high-freq texture for rain, low-freq impulses for thunder) — no separation
needed. **Authoritative observation; makes no season/diel claim** (rain falls
in any season).

**Output — continuous magnitudes, two-tier.** Intensities are floats
`0.00–1.0`, anchored to the curated weather-asset library's range (0 =
silence floor, 1 = loudest labelled asset of that type), so a value is
reproducible and calibratable. Each channel carries:

- `intensity` — overall energy of the channel.
- `variability` — fluctuation over time (steady 0 ↔ gusty/showery 1).
- `coverage` — fraction of the clip the channel is audibly present.
- `label` — **derived** bucket (`none/light/moderate/strong|heavy`) computed
  from `intensity` via fixed thresholds. The float is primary; the bucket is a
  human-readable + generation-Layer-B view of the same number (single source
  of truth).

Wind/rain are continuous textures (reported as **spans**); thunder is discrete
(reported as **point events** — its `events` array *is* its timeline, so it
needs no `segments`).

**Tiered contract:** `summary` is **compulsory** — every E-B output has it,
and the report headline + generation Layer B depend only on it. `segments` /
`events` are **optional (advanced)**: present when the detector supports
timeline output (MVP E-B may ship summary-only), and consumers **must degrade
gracefully when absent**. When emitted, `segments` is a full array (quantized
~3–5 s, adjacent same-label spans merged, collapsing to a single whole-clip
span when uniform) — presence is driven by detector capability, not by whether
the clip happened to be steady.

```json
{
  "wind": {
    "summary":  { "intensity": 0.62, "variability": 0.40, "coverage": 0.95, "label": "moderate", "confidence": 0.83 },
    "segments": [
      { "t_start": 0.0,  "t_end": 8.0,  "intensity": 0.45, "label": "light" },
      { "t_start": 8.0,  "t_end": 14.0, "intensity": 0.78, "label": "moderate" },
      { "t_start": 14.0, "t_end": 30.0, "intensity": 0.55, "label": "moderate" }
    ]
  },
  "rain": {
    "summary":  { "intensity": 0.10, "variability": 0.70, "coverage": 0.20, "label": "light", "confidence": 0.55 },
    "segments": [ { "t_start": 6.0, "t_end": 11.0, "intensity": 0.30, "label": "light" } ]
  },
  "thunder": {
    "intensity": 0.00,
    "event_count": 0,
    "events": [],
    "mean_interval_s": null,
    "confidence": 0.90
  },
  "confidence": 0.80
}
```

`summary.variability` is the spread of segment intensities; `summary.coverage`
is the fraction of time above the silence floor — so the two tiers are
internally consistent (segments are the source, summary the rollup). For
thunder, `events` holds `{ "onset_s", "strength" }` and `mean_interval_s` is
the average gap between claps (`null` if fewer than 2 events).

**Pre-trained model options:**

| Model | Type | What it gives | Notes / tradeoff |
|---|---|---|---|
| **PANNs CNN14** | AudioSet tagger (527 classes) | direct `Wind`, `Rain`, `Raindrop`, `Thunderstorm` logits | strongest off-the-shelf weather baseline |
| **YAMNet** | AudioSet tagger (521 classes) | same label space, lighter | smaller, faster, slightly weaker than CNN14 |
| **LAION-CLAP zero-shot** | audio↔text | score against prompts like `"strong wind in trees"`, `"light rain"` | free if CLAP already loaded for E-A |
| **DSP features** (sub-200 Hz RMS, 2–8 kHz flatness) | hand-crafted | cheap explainability channel; sanity-check ML output | not a primary detector — calibration / XAI only |

**MVP path:** PANNs zero-shot baseline → fine-tune a small head on the
curated wind/rain assets in `data/weather/weather_assets/` once labels
accumulate.

**Code:** `layers/layer_e/attempts/<id>/weather_detector.py` [PLACEHOLDER]
**Shared data:** `data/weather/asset_index.csv` (weather intensity labels)

---

### E-C — Species and event detector

**Purpose:** identify biologically meaningful events on the raw mixture,
with onsets/offsets. Pre-trained bioacoustic detectors were *built* for
mixed-source audio — that's their entire reason for existing.

- **Authoritative observation:** species present + onsets/offsets.
- **Evidence toward latent context:** the **strongest** season/diel signal.
  Each detected species maps (via the phenology table) to an activity niche —
  a nocturnal owl pins diel to night; a cicada chorus pins warm-season
  daytime; a migratory species pins a narrow season. The aggregator weights
  this by `confidence × niche_specificity` (see
  [analysis_synthesis_policy.md](analysis_synthesis_policy.md)).

**Output:** each detection carries a `phenology` block joined from the species
phenology table (§ *Required artifact* below).
```json
[
  {
    "label": "Southern Boobook",
    "confidence": 0.91,
    "onset_s": 12.4,
    "offset_s": 13.1,
    "phenology": {
      "diel_window": "night",
      "season_window": "year-round",
      "niche_specificity": { "diel": 0.95, "season": 0.10 },
      "source": "site257_phenology_table"
    }
  }
]
```

**Pre-trained model options:**

| Model | Type | What it gives | Notes / tradeoff |
|---|---|---|---|
| **BirdNET-Analyzer** | bird classifier (~6k species) | direct species labels + onsets; covers most Australian birds | primary detector; already used as Layer A audit tool |
| **Google Perch** | bioacoustic embedding | embeddings for a small fine-tuned head where BirdNET is weak (frogs, insects, mammals) | extra training but covers BirdNET's blind spots |
| **A2O annotation index** | curated dataset labels | high-confidence overrides / cross-reference | not a detector — validation channel |
| **LAION-CLAP zero-shot** | audio↔text | open-vocabulary fallback ("Southern Boobook call", "helicopter") | catches anything not in BirdNET's label set |
| **AudioGen EnCodec tokens** | discrete audio features | features for a custom event classifier | optional; only worth it if BirdNET + Perch are insufficient |

**MVP path:** BirdNET as primary; A2O cross-reference for overrides;
CLAP zero-shot as the open-vocabulary fallback.

**Code:** `layers/layer_e/attempts/<id>/event_detector.py` [PLACEHOLDER]
**Shared data:** `data/events/annotation_event_index.csv`

---

### MVP-1 build order (analysis)

```
   Step 1 ─ CLAP embedding service   (load once, shared by E-A + zero-shot fallbacks)
   Step 2 ─ BirdNET subprocess       (E-C primary)
   Step 3 ─ PANNs wind/rain head     (E-B primary; summary-only first, segments later)
   Step 4 ─ Species phenology table  (prereq for season/diel fusion — see below)
   Step 5 ─ Aggregator               (deterministic fusion → Report JSON)
   Step 6 ─ LLM-OSS narration        (renders report; analytical / immersive)
```

No new training required to reach an end-to-end smoke. Fine-tuned heads
(weather classifier, Perch-based event head), E-B `segments`, and the unified
cross-head timeline come later. Fusion + report-writing policy:
[analysis_synthesis_policy.md](analysis_synthesis_policy.md).

---

### Required artifact: species phenology table

E-C's season/diel evidence is only quantifiable with a lookup table mapping
each species to its activity niche — **it does not exist yet** and is a
prerequisite for aggregator fusion:

```
species_id → { season_window, diel_window,
               niche_specificity: { season, diel },   # narrow niche → high
               source }                                # site-257 manual | A2O | Xeno-canto
```

Ideally site-257-specific and manually validated (e.g. confirm "cicada →
summer afternoon" against the real recordings before hard-coding it); a
general source is the fallback. A year-round generalist gets wide windows and
~0 specificity, contributing ~no season/diel evidence — correct behaviour.
Full spec: [analysis_synthesis_policy.md § 7](analysis_synthesis_policy.md).

---

### Analysis Report Fields

The aggregator separates **authoritative observations** from **fused latent
inference** and records head **disagreements**. Full schema + the trust
hierarchy that fills `inferred_context`:
[analysis_synthesis_policy.md § 3–4](analysis_synthesis_policy.md).

| Field | Source |
|---|---|
| `observations.weather` | E-B — wind/rain/thunder `summary` (+ optional `segments`/`events`) |
| `observations.events` | E-C — detected species + onsets |
| `observations.ambient` | E-A — `similar_clips` |
| `inferred_context.diel`, `inferred_context.season` | Aggregator — fused posterior + distribution + primary evidence |
| `disagreements` | Aggregator — heads that conflicted + resolution |
| `confidence` | Aggregator — calibrated overall confidence |
| `limitations` | Notes on model limitations and dataset coverage |
