# Analysis Synthesis Policy — Aggregator Fusion + Report Writing

> Companion to [pipeline_design.md](pipeline_design.md) (Analysis Mode).
> That doc defines the **heads** (E-A / E-B / E-C) and what each produces.
> This doc defines what happens **after** the heads run: how the Layer E
> aggregator fuses their outputs into a single answer, and how the
> downstream LLM-OSS layer turns that answer into a report.

---

## Contents

1. [Use case: production vs. dev](#1-use-case-production-vs-dev)
2. [Core principle: observations vs. inferences](#2-core-principle-observations-vs-inferences)
3. [Aggregator fusion (deterministic)](#3-aggregator-fusion-deterministic)
4. [Fused report JSON](#4-fused-report-json)
5. [LLM-OSS writing policy (two registers)](#5-llm-oss-writing-policy-two-registers)
6. [Per-head pass standards](#6-per-head-pass-standards)
7. [Required artifact: species phenology table](#7-required-artifact-species-phenology-table)

---

## 1. Use case: production vs. dev

**Production is audio-only and speculative.** The uploaded clip is treated as
anonymous: season, diel (time of day), and weather are *inferred from the
waveform alone*. This is the hard, honest case — the ambient bed at this site
is not seasonally discriminative (spring and autumn sound nearly identical),
so the report must be willing to say "undetermined" rather than force a guess.

**Dev may use clip metadata as a validation aid only.** During development we
often *do* know a clip's true timestamp/GPS (A2O clips carry them). Reading
that metadata to check the model's inference is a **dev convenience**, never
part of the production analysis path. Ground-truth comparison validates the
pipeline; it does not feed it.

> Rule: nothing in the production inference path reads clip metadata. If a
> future product wants metadata-backed "model vs. truth" reporting, that is a
> separate, explicitly-designed mode.

---

## 2. Core principle: observations vs. inferences

Every analysis output is one of two kinds:

- **Observation** — something a head *directly detects* in the audio. Wind
  intensity, rain, a species call at 0:12, the acoustic character of the bed.
  The owning head is **authoritative**: no fusion, no second-guessing.
- **Inference (latent context)** — season, diel, plausible env ranges.
  *No head observes these directly.* They are latent variables estimated from
  observations, **owned by nobody**, and produced only by the aggregator's
  fusion step.

This dissolves a common design error: assigning "estimate the season" to a
single head (e.g. the ambient head) and then being disappointed it is
unreliable. Season was never a single head's observation — it is a fused
inference, and the *event* head usually carries the strongest evidence for it
(via species phenology), with the ambient head as a weak prior.

| Kind | Examples | Owner | Fusion? |
|---|---|---|---|
| Observation | wind/rain/thunder, species + onsets, ambient character, similar clips | the detecting head, authoritative | none |
| Inference | season, diel, plausible env ranges | aggregator | yes — evidence-weighted |

---

## 3. Aggregator fusion (deterministic)

**The aggregator computes the fused answer in deterministic code. The LLM does
NOT do the weighting.** LLMs re-weight silently run-to-run; deterministic
fusion is reproducible and auditable. The LLM only *narrates* the aggregator's
output (§5).

### Trust hierarchy

| Question | Trust first | Trust second | Ignore |
|---|---|---|---|
| Wind / rain / thunder | E-B | — | — |
| Which species | E-C | E-A neighbours (cross-ref) | E-B |
| **Diel (time of day)** | **E-C** (nocturnal / crepuscular / diurnal niche) | E-A | E-B |
| **Season** | **E-C** (phenology) | E-A | E-B (very weak) |
| Site / biome character | E-A | E-C | E-B |

### Fusion rules

- **Rule 0 — observations pass through untouched.** Wind/rain/thunder → E-B.
  Species presence → E-C (above its confidence threshold). Ambient character /
  similar clips → E-A. No fusion; the owning head wins by definition.

- **Rule 1 — season/diel fusion is evidence-weighted, gated by confidence ×
  specificity.** For each candidate `(season, diel)` value, weight each head's
  vote by:

  ```
  evidence_weight(head) = head_confidence × niche_specificity
  ```

  - **E-C dominates** when it detects a phenologically-specific species at
    high confidence (cicada chorus → warm-season daytime; nocturnal owl →
    night; a migratory species → narrow season). Narrow niche × high
    confidence → high weight.
  - **E-A is the fallback / tiebreaker** — used when there are no
    discriminative events, or to disambiguate between seasons E-C leaves open.
    Capped at a low weight unless its k-NN neighbours are tight (low
    dispersion) *and* its confidence is high.
  - **E-B contributes almost nothing** to season/diel; at most a weak soft
    prior (e.g. sustained heavy rain slightly upweights wet-season).

- **Rule 2 — conflict handling.** If high-confidence E-C contradicts E-A:
  **prefer E-C, lower the overall posterior confidence, and record the
  disagreement** (never hide it). If *all* heads are low-confidence: emit a
  **range or "undetermined"**, never a false-precise point estimate.

- **Rule 3 — distributions, not labels.** E-A votes across its k neighbours;
  E-C detections may point at different seasons. The aggregator emits
  `P(season)` and `P(diel)` plus the top contributing evidence, so the report
  can say "likely summer (0.65), possibly autumn (0.30)."

---

## 4. Fused report JSON

The aggregator emits one object separating **observations** (authoritative)
from **inferred_context** (fused) and recording **disagreements**. The LLM
narrates this; it is also the API contract for any UI.

```json
{
  "observations": {
    "weather": {
      "wind":    { "summary": { "intensity": 0.62, "variability": 0.40, "coverage": 0.95, "label": "moderate", "confidence": 0.83 } },
      "rain":    { "summary": { "intensity": 0.10, "variability": 0.70, "coverage": 0.20, "label": "light",    "confidence": 0.55 } },
      "thunder": { "intensity": 0.00, "event_count": 0, "events": [], "mean_interval_s": null, "confidence": 0.90 }
    },
    "events": [
      { "label": "Southern Boobook", "confidence": 0.91, "onset_s": 12.4, "offset_s": 13.1 }
    ],
    "ambient": { "similar_clips": [ { "segment_id": "seg_00417", "similarity": 0.71 } ] }
  },
  "inferred_context": {
    "diel":   { "estimate": "night",       "posterior": 0.88, "distribution": { "night": 0.88, "afternoon": 0.06, "morning": 0.04, "dawn": 0.02 }, "primary_evidence": "E-C: Southern Boobook (nocturnal)" },
    "season": { "estimate": "undetermined", "posterior": 0.40, "distribution": { "summer": 0.40, "autumn": 0.35, "spring": 0.15, "winter": 0.10 }, "primary_evidence": "E-A weak prior; no seasonally-specific species" }
  },
  "disagreements": [
    { "field": "season", "E-A": "autumn", "E-C": "inconclusive", "resolution": "low-confidence range reported" }
  ],
  "confidence": 0.60,
  "limitations": [
    "Season is under-determined by audio at this site — spring and autumn beds are near-identical.",
    "Time-of-day rests on the detected owl, not on the ambient texture."
  ]
}
```

---

## 5. LLM-OSS writing policy (two registers)

The LLM-OSS layer turns the fused JSON into prose. It **renders**, it does not
**decide** — all weighting already happened in §3. Two output registers share
one set of *content* rules and differ only in *phrasing*.

### Shared content rules (both registers)

1. **Observations are stated as fact** ("wind: moderate"; "a Southern Boobook
   calls at 0:12").
2. **Inferences are hedged to the posterior** ("likely a summer night, ~0.7").
   A low posterior is reported as "undetermined" — never invent precision.
3. **Disagreements are surfaced**, with the resolution reason ("the ambience
   resembles autumn, but the detected cicadas point to summer — we weight the
   species evidence more heavily").
4. **Always close with limitations.** This is a *speculative* project;
   embracing uncertainty is on-brand, not a weakness.

### Register selection

Selectable per analysis (default: **Analytical** for dev, **Immersive** for
the demo UI). Not shown together unless requested.

| | Style A — Analytical | Style B — Immersive |
|---|---|---|
| Voice | structured, sectioned (observations / inferences / limitations) | flowing scene description |
| Best for | researcher reading results, debugging | demo, public-facing "speculative soundscape" |
| Uncertainty | explicit numbers (≈0.65) | woven into language ("keeps its secret") |
| Timestamps | exact (`0:12`) | softened ("around the twelve-second mark") |

### Canonical samples

**Style A — Analytical** (night; species fixes the time, season undetermined):

> **What we can hear**
> - A **Southern Boobook** call at **0:12** (high confidence).
> - Quiet woodland ambience, **moderate wind**, no rain.
>
> **What this suggests**
> - **Time of day — almost certainly night.** The Southern Boobook is strictly
>   nocturnal, so its presence is a strong time signal *(≈0.88)*.
> - **Season — can't be determined from the audio.** The ambient texture fits
>   both summer and autumn, and no strongly seasonal species were detected.
>
> **Limitations**
> - Season is genuinely hard to read from this site's background bed — spring
>   and autumn recordings sound nearly identical. The time-of-day estimate
>   rests on the detected owl, not on the ambience.

**Style B — Immersive** (same fused JSON as above):

> Somewhere out in the dark, a **Southern Boobook** calls — a low two-note
> *boo-book* surfacing around the **twelve-second mark**, and unmistakably a
> creature of the night. The woodland around it is hushed: a steady wind
> working through the canopy, no rain. The hour is certain; the season keeps
> its secret. This background could belong to a summer evening as easily as an
> autumn one — here the two sound almost the same — so we name the time of day
> and leave the month open.

**Style B — Immersive** (cicada afternoon, warm-season hedge; exercises the
weather fields):

> The recording is thick with **cicadas**, a dense shimmering drone that fills
> every second — the sound of heat itself. A light wind stirs the leaves; no
> rain. This is almost certainly a **summer afternoon**, though the same chorus
> could just as well carry into an early-autumn one — the warmth is
> unmistakable even when the exact month isn't. What's not in doubt is the time
> of day: the cicadas are at their afternoon peak.

---

## 6. Per-head pass standards

Analysis heads are graded on **calibration + role-appropriate usefulness**,
**not accuracy**. The governing principle:

> A head passes if it is **right when confident and honest when not** — not if
> it is accurate every time. Uncertainty reported as uncertainty is a pass,
> not a fail.

| Head | ❌ Wrong bar | ✅ Role-appropriate pass standard |
|---|---|---|
| **E-A Ambient** | "predicts exact season/diel" | (1) **Retrieval relevance** — top-k neighbours are perceptually/contextually similar. (2) **Calibration** — high confidence ⇒ season/diel within ±1 bin; ambiguous bed ⇒ *reports low confidence* instead of a false-precise label. **Not expected to nail exact season.** |
| **E-B Weather** | "classifies intensity perfectly" | **Presence/absence first** (never hallucinate rain), then `summary` intensity within ~one bucket of the labelled assets. `summary` graded strictly; `segments` is **advanced/bonus, non-gating** for the smoke, graded on ordinal shape ("intensity rises where the audio gets windier"), not exact boundaries. |
| **E-C Events** | "detects every event" | **Precision-first at the operating threshold** — a *reported* species is usually correct; recall may be partial. Onsets within tolerance (≈±1 s). Missing a faint distant call is acceptable; confidently mislabelling one is not. |
| **Aggregator / report** | "season/diel always correct" | (1) **Fused-confidence calibration** — ~0.8 means right ~80% of the time. (2) **Always surfaces disagreement** on conflict. (3) **Never emits a false-precise season** on weak evidence — defaults to a range or "undetermined." |

---

## 7. Required artifact: species phenology table

E-C's "strong evidence" for season/diel is quantified with the shared lookup
table at `acoustic_ai/layers/layer_e/shared/species_phenology.csv`, mapping each
species to its activity niche. This table is the single source of truth for
E-C handler metadata and aggregator fallback enrichment.

```
species_id → {
  season_window:     e.g. "summer" | "warm-season" | "year-round"
  diel_window:       e.g. "night" | "crepuscular" | "diurnal" | "afternoon"
  niche_specificity: { "season": 0.0–1.0, "diel": 0.0–1.0 }   # narrow niche → high
  source:            provenance (site-257 manual | A2O histogram | Xeno-canto)
}
```

- **Scope:** currently coarse/general-source metadata. Ideally replace or refine
  entries with site-257-specific manual validation — e.g. validate
  the "cicada → summer afternoon" assumption against the actual recordings
  before increasing any specificity. A general source (Xeno-canto / A2O seasonal
  histograms) remains the fallback for species lacking site data.
- A year-round generalist resident gets wide windows and ~0 specificity, so it
  contributes ~no season/diel evidence — correct behaviour.
