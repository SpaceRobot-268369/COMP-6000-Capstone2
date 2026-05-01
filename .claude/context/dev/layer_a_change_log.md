# Layer A — Ambient Bed: Design Change Log

Date: 2026-05-01
Branch: `model/lucas/layer-a-ambient`

This document captures every change made to the Layer A design during the
2026-05-01 review session and the reasoning behind each change. The final
proposal at the bottom supersedes the Layer A sections of
`pipeline_design.md`, `architecture.md`, and `mvp_decision_log.md`.

---

## Why this revision happened

The Layer A design that came out of the original MVP decision log assumed
two things that turned out to be wrong:

1. That retrieving real clips by environmental similarity would give a
   usable ambient bed. The clips contain bird calls, vehicles, helicopters
   and other events — the very content that Layer C is supposed to own. A
   clip-level retrieval pool double-counts events with Layer C and breaks
   the layered design.
2. That the existing VAE checkpoint was a meaningful representation for
   Layer A. It was trained on uncleaned full clips, so its latent manifold
   mixes ambient with events. Sampling or decoding from it cannot give
   "ambient only".

Three further refinements emerged during review.

---

## Changes (in the order they were made)

### Change 1 — Move Layer A from clip-level to segment-level retrieval

**Old:** retrieve full clips by env-vector cosine similarity, use them
directly as the bed (or decode VAE latents).

**New:** build a *cleaned ambient-only segment pool* in a precompute step.
Retrieve from that pool at runtime.

**Why:** layered separation only works if Layer A actually contains *only*
ambience. Anything else makes Layer C double-count and makes the
explanation JSON dishonest.

---

### Change 2 — Drop temp/humidity/wind/rain from the retrieval key

**Old retrieval key:** 29-d env vector — included temp, humidity, wind
speed/direction, precipitation, days_since_rain, plus diel bin and season.

**New retrieval key:**
- **Hard filter:** `diel_bin` + `season`.
- **Soft rank:** cosine similarity over `[hour_sin, hour_cos, month_sin,
  month_cos]` only.

**Why:**
- Wind and rain are directly audible signals owned by Layer B. Including
  them in Layer A's key double-counts.
- Temperature and humidity affect species/insect behaviour — that effect
  flows through Layer C, not Layer A.
- The ambient bed's character is driven by *time of day* and *seasonal
  position*. Pulling weather variables in either double-counts (with B)
  or injects similarity that is irrelevant to ambient character (with C).
- Categorical features (diel_bin, season) belong in a hard filter, not in
  cosine — one-hot cosine is unstable and can rank summer-dusk above
  winter-dawn just because numeric temperatures happen to be close.

---

### Change 3 — Cleaning is audio-only; annotations are validation, not gating

**Old cleaning input set (proposed):** A2O annotations, BirdNET, and
energy-based detectors combined.

**New cleaning input set:** audio-only signals. Annotations are used only
for downstream validation.

**Why:**
- A2O annotations are sparse and asymmetric: presence ⇒ event, but
  absence ⇏ no event. Only ~22 of 287 recordings carry events at all.
  Using "no annotation" as a negative signal is unsound — it would let
  unannotated events into the pool.
- The only trustable signal is the audio itself.
- Annotations remain useful as a *precision check* over the retained
  segments (does our pool overlap their known events?), so they are kept
  in the validation harness, not the gate.

---

### Change 4 — Drop BirdNET from the gate; use stationarity (3·MAD)

**Original gate stack (after Change 3):** BirdNET frame scores + onset
peaks + spectral flux spikes + band-energy CV + RMS bounds + mel
stationarity.

**New gate stack:** per-clip anomaly detection on standard audio
features — mel, RMS, spectral centroid, spectral flatness, spectral flux,
zero-crossing rate. A frame is anomalous if any feature deviates more
than 3·MAD from a 30 s rolling-median baseline computed *on its own
clip*.

**Why:**
- Events are an open class. BirdNET only knows the species in its
  training set; it is deaf to helicopters, vehicles, frogs, branch snaps,
  distant voices and anything else. As a permissive gate (we want to
  *exclude* events) this is the wrong failure mode — anything BirdNET
  doesn't recognise gets accepted.
- Ambient is locally stationary. Every event — whatever its source —
  shares one property: it deviates from its own clip's running baseline.
  So the right detector is content-agnostic stationarity, not category
  detection.
- 3·MAD with a per-clip rolling baseline is **self-calibrating**: a clip
  recorded near a cicada chorus gets a higher noise floor automatically,
  and "deviation" remains meaningful in both quiet and dense recordings.
- BirdNET is demoted to a post-hoc audit alongside annotations:
  "out of the segments we kept, what did BirdNET detect?" If the fire
  rate is too high, tighten the MAD threshold and re-run.
- Side benefit: removes a slow neural inference pass over ~6,148 clips
  from the precompute. Whole pipeline becomes CPU + librosa, runs in
  minutes.

---

### Change 5 — Longer segments (20–60 s, target 30 s)

**Old segment length:** 5–10 s sliding windows.

**New segment length:** find contiguous unmasked spans ≥ 20 s, slice them
into segments of 20–60 s (target 30 s).

**Why:**
- 5–10 s segments require many crossfades for any usable target duration.
  Crossfade boundaries are audible if the segments differ in timbre or
  level — the bed will sound "broken".
- 30 s segments cover a 60 s request with one mid-fade. 60 s segments
  cover most requests with no crossfade at all.
- The longest-contiguous-span approach naturally produces this length
  range — it looks for the longest event-free stretches in each clip
  rather than carving the clip into arbitrary pieces.

---

## Final proposal — Layer A: Ambient Bed (post-revision)

### Stage 1 — offline cleaning (`precompute/build_ambient_index.py`)

```
downloaded_clips/*.webm
        │
        ▼
resample → 22050 Hz mono WAV (already cached as *.webm.wav)
        │
        ▼
per-frame features over the whole clip:
  mel-mean (mean log-power per frame), RMS, spectral centroid,
  spectral flatness, spectral flux, zero-crossing rate
        │
        ▼
per-clip rolling baseline (30 s window):
  rolling median + MAD per feature
        │
        ▼
anomaly mask: any feature deviates > 3·MAD from its rolling median
        │
        ▼
dilate ± 0.5 s → invert mask → contiguous "ambient" spans
        │
        ▼
keep spans ≥ 20 s
verify: RMS in [p20, p80] of clip + low frame-to-frame mel variance
        │
        ▼
slice each verified span into segments of 20–60 s (target 30 s)
        │
        ▼
data/ambient/ambient_segments/<segment_id>.wav
data/ambient/ambient_index.csv
  segment_id, source_clip, t_start, t_end,
  diel_bin, season, hour_sin, hour_cos, month_sin, month_cos
```

### Stage 2 — runtime retrieval (`modules/ambient/retrieval.py`)

```
user env request → derive (diel_bin, season, hour, month)
        │
        ▼
HARD FILTER: diel_bin == request AND season == request
        │
        ▼
SOFT RANK: cosine over [hour_sin, hour_cos, month_sin, month_cos]
           top-k = 5
        │
        ▼
BLEND: weights = softmax(sim / τ), τ = 0.1
       crossfade-mix the 5 segments (~1 s crossfade)
       RMS-match
       tile/loop to target_duration_s
        │
        ▼
LayerResult(
  audio, sample_rate=22050, gain_db=-3,
  metadata={ retrieved_clips, blend_weights, requested_env }
)

Fallback: if hard filter returns < 5,
          relax to neighbouring diel_bin and set
          metadata.low_confidence = True.
```

### Validation (does **not** gate the pool)

- Manual listening audit: 30 random segments → ≥ 27 sound like ambience.
- Annotation cross-check: < 1 % of retained-segment-seconds overlap any
  A2O annotated event in the ~22 annotated recordings.
- BirdNET fire-rate audit over retained segments — used to tune the MAD
  threshold if too lenient.
- Per-request scrutinization triplet (`debug/layer_a_bed.wav`,
  `debug/layer_a_spec.png`, JSON) per `layer_verification_formats.md`.
- Behavioural tests: dawn/winter vs afternoon/summer should retrieve
  disjoint sets and sound different. Same request 5× should be
  deterministic.

### Deliberately excluded

- VAE checkpoint on the Layer A runtime path (event-contaminated
  training data). Retained for transformation mode and Module E.
- Vocoder on the Layer A path (no decoder, so no need).
- Weather variables in retrieval key.
- Annotations and BirdNET in the cleaning gate.
