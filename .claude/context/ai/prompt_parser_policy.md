# Prompt Parser Policy — Pre-fill, Validity Gate + Layer Decoding

> Companion to [pipeline_design.md](pipeline_design.md) (Generation Mode).
> That doc defines the **layers** (A / B / C) and what input contract each
> expects. This doc defines what happens **before** the layers run: how the
> raw user prompt is pre-processed, defaulted, validity-checked, and only
> then decoded into the three layer contracts.
>
> This is the generation-side mirror of
> [analysis_synthesis_policy.md](analysis_synthesis_policy.md): both are
> LLM-OSS layers governed by a written policy. Analysis turns detector
> outputs *into* a report; the parser turns a raw prompt *into* layer inputs.

---

## Contents

1. [What the parser is](#1-what-the-parser-is)
2. [Stage 1 — pre-process & default fill](#2-stage-1--pre-process--default-fill)
3. [Stage 2 — validity / coherence gate](#3-stage-2--validity--coherence-gate)
4. [Stage 3 — decode into layer contracts](#4-stage-3--decode-into-layer-contracts)
5. [Parse result schema](#5-parse-result-schema)

---

## 1. What the parser is

The **Prompt Parser** is the orchestrator at the front of the generation
workflow. It is **LLM-OSS powered** (same model family as the Layer E report
writer), driven by this policy — not a single regex pass. A lightweight
rule-based fast-path may short-circuit trivially unambiguous prompts, but the
LLM + policy is the source of truth.

It does **three things**, in order:

1. **Pre-process & default-fill** the prompt — normalise it and supply
   sensible defaults for anything the user did *not* specify.
2. **Validate coherence** — reject requests that don't make sense for our
   site/models, and *suggest a corrected prompt* instead of failing silently.
3. **Decode** the (now complete, validated) request into the three
   layer-specific input contracts (A cell, B weather JSON, C species
   checklist).

Why this lives in front of the layers: generation is split into independent,
modular layers, so a raw natural-language prompt can't be fed straight into
any one layer's model. Each layer owns a different input contract, and most
prompts under-specify (a user says "autumn dawn" and says nothing about
weather or fauna). The parser is what turns one partial sentence into three
complete, aligned, *plausible* inputs.

---

## 2. Stage 1 — pre-process & default fill

The parser normalises the prompt (casing, synonyms, units) and then fills any
layer the user left unspecified with an explicit default. **Silence is a
decision, not a gap.**

| Layer | If the user *does not* mention it | Rationale |
|---|---|---|
| A — Ambient bed | **Always on.** Resolve `(season, diel)` from the prompt; if absent, use the attempt's `default_cell`. | The site bed is the substrate of every soundscape — there is no "no ambient" option. |
| B — Weather | **Off.** No rain, wind, or rain+wind unless explicitly requested. Thunder/storm wording is corrected to `rain+wind` until reliable site thunder candidates are exposed. | A clear, calm day is the neutral baseline. Weather is an *additive* event the user opts into. |
| C — Events | **Empty checklist.** No species placed unless named or clearly implied. | Foreground calls are deliberate. An empty timeline yields a quiet, ambient-only scene rather than random fauna. |

So "a misty autumn dawn" fills to: Layer A `(autumn, dawn)`, Layer B *none*,
Layer C *empty* — a quiet dawn bed, no rain. The defaults are recorded
explicitly in the parse result so the UI can show the user what was assumed.

---

## 3. Stage 2 — validity / coherence gate

After defaults are filled, the parser checks that the request is something our
site and models can actually voice. **We synthesise a remote Australian dry
woodland (site_257 Bowra), not arbitrary audio.** Requests outside that domain
are caught here.

Gate checks (non-exhaustive; the policy is the live list):

- **Out-of-domain content** — dense city traffic, sirens, machinery, music,
  crowds. None of these belong in a remote dry-woodland soundscape and none
  are covered by our layers. Severity scales with saturation: **one or two**
  stray anthropic elements over an otherwise in-domain scene are **swapped**
  out (correct-and-continue, keep the rest); a prompt **saturated** with
  out-of-domain content (≥ `OUT_OF_DOMAIN_BLOCK_THRESHOLD` distinct concepts,
  nothing in-domain left to keep) is **blocked → rejected** with the nearest
  in-domain scene suggested. The deterministic gate owns this decision
  (`gate.py`); negated mentions ("no traffic") are not counted.
- **Phenologically implausible fauna** — a species that does not occur at the
  site, or not in the requested `(season, diel)`. Cross-check against the
  species phenology table in
  [analysis_synthesis_policy.md §7](analysis_synthesis_policy.md). → **soften**:
  drop or swap for a plausible caller and note the substitution.
- **Climatically implausible weather** — e.g. snow at an arid inland site. →
  **block or swap** for a plausible analogue (heavy rain, dust-laden wind).
- **Internally contradictory** — "silent, with a loud dawn chorus". →
  **ask / pick the dominant intent** and note it.

**Gate behaviour is correct-and-continue, not hard-fail.** Wherever possible
the parser returns `status: "corrected"` with a rewritten prompt and a plain
sentence explaining the change ("City traffic isn't part of this site — I
generated a remote dawn bed with a passing breeze instead"). Only genuinely
unrecoverable requests return `status: "rejected"` with a suggestion. This
gate is the **canonical home for the block/suggest rules** — keep it in sync
with the site's data reference and the phenology table.

---

## 4. Stage 3 — decode into layer contracts

Only a complete, validated request reaches decoding. The parser emits one
aligned input per layer:

1. **Layer A** — a valid `(season, diel)` cell tuple for cell-based banks, or
   a background-texture sub-prompt (foreground events stripped) for
   open-prompt models.
2. **Layer B** — structured JSON: `weather_type` (`rain` | `wind` | …),
   `intensity` (`light` | `medium` | `heavy`), `duration_s`. Omitted entirely
   when Stage 1 left weather off.
3. **Layer C** — a species checklist (common names) plus density/timeline
   params, mapped to per-species LoRA weights or the audited retrieval query.

Each downstream layer therefore receives exactly what its API expects, and
nothing it can't handle. The dev-generation contract still applies on top of
this: for locked bank attempts the FastAPI server owns the prompt/checkpoint
and the backend forwards only `{ seed }` (+ a validated `(season, diel)` cell)
— see [CLAUDE.md → Generation workflow prompt parsing & contracts](../../../CLAUDE.md).

---

## 5. Parse result schema

The parser returns a single structured object (also surfaced to the UI):

```json
{
  "status": "ok | corrected | rejected",
  "note": "human-readable explanation of any defaults filled or corrections made",
  "filled_defaults": ["weather:none", "events:empty"],
  "layer_a": { "season": "autumn", "diel": "dawn" },
  "layer_b": null,
  "layer_c": { "species": [], "density": "sparse" }
}
```

- `ok` — request was complete and coherent (after default fill).
- `corrected` — parser changed something (gate swap or default); `note` and
  `filled_defaults` explain what and why; generation proceeds.
- `rejected` — unrecoverable; `note` carries a suggested alternative prompt
  and no layer contracts are emitted.
