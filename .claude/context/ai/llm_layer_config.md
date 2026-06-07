# LLM-OSS Layer — Model Choice & Serving Config

> Decision record for the open-source LLM that powers the two **LLM-OSS**
> consumers in this project:
>
> - the generation-side **Prompt Parser**
>   ([prompt_parser_policy.md](prompt_parser_policy.md))
> - the analysis-side **Layer E report writer**
>   ([analysis_synthesis_policy.md §5](analysis_synthesis_policy.md))
>
> Both are *the same model instance* (the policies call them "the same model
> family"). This doc records **which model**, **how it is served**, and **why
> a small model is strong enough**. Implementation scaffold lives at
> [`acoustic_ai/llm/`](../../../acoustic_ai/llm/).

---

## Contents

1. [Decision summary](#1-decision-summary)
2. [Why a small model is strong enough](#2-why-a-small-model-is-strong-enough)
3. [Serving: in-process, no new port](#3-serving-in-process-no-new-port)
4. [VRAM budget (serverB, 16 GB)](#4-vram-budget-serverb-16-gb)
5. [Model choice + license note](#5-model-choice--license-note)
6. [Guardrails](#6-guardrails)
7. [Upgrade lever](#7-upgrade-lever)
8. [Open follow-ups](#8-open-follow-ups)

---

## 1. Decision summary

| Question | Decision |
|---|---|
| **Where does it run?** | In-process inside the existing FastAPI app on serverB `:8000`. **No second server, no extra port.** |
| **Serving stack** | `transformers` (already a dep) + `bitsandbytes` 4-bit NF4. Optional `lm-format-enforcer` / `outlines` for grammar-constrained JSON. |
| **Primary model** | `Qwen/Qwen2.5-3B-Instruct` (4-bit, ~2.5 GB resident). |
| **License-clean alt** | `microsoft/Phi-3.5-mini-instruct` (3.8B, **MIT**, ~3 GB). |
| **Consumers** | One model instance serves both the Prompt Parser and the Layer E report writer. |
| **Upgrade lever** | `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) via selective pre-warm — only if the 3B's immersive prose underperforms on demos. |

---

## 2. Why a small model is strong enough

The architecture **deliberately keeps the LLM out of the hard reasoning.** The
unreliable parts — season/diel inference, evidence weighting, conflict
resolution, phenology plausibility — are **deterministic code**, not LLM calls:

- *"The aggregator computes the fused answer in deterministic code. The LLM
  does NOT do the weighting … The LLM only narrates."*
  ([analysis_synthesis_policy.md §3](analysis_synthesis_policy.md))
- The parser's validity gate cross-checks against a **phenology lookup table**
  ([analysis_synthesis_policy.md §7](analysis_synthesis_policy.md)) — "is this
  species plausible in autumn dawn?" is a table read, not model judgment.

What is left for the LLM is a small, well-bounded job:

| Task | What the LLM actually does | Load |
|---|---|---|
| Parser — extract | "a misty autumn dawn" → `{season, diel}`, spot named species/weather | classic NL extraction — easy |
| Parser — default-fill | trivial once extraction is done (mostly deterministic) | trivial |
| Parser — gate phrasing | *decision* is a table lookup; LLM writes the "swapped X for Y" note | easy |
| Parser — emit contract | JSON to the parse-result schema, **guaranteed by constrained decoding** | solved by tooling, not model size |
| Report — render | turn a *fully-decided* fused JSON into prose, hedged to the given posterior, two registers | moderate — only place prose quality scales with size |

This profile (extraction + constrained JSON + rendering) is exactly what small
instruct models are good at. The design was built so you **do not** need a
heavyweight reasoner.

---

## 3. Serving: in-process, no new port

serverB exposes **no public ingress**; the FastAPI app binds `127.0.0.1:8000`
and is reached from Server A through the `ai-tunnel` Compose sidecar
([on_demand_ai_worker.md](../setup/server/on_demand_ai_worker.md)). The two LLM
uses are **internal function calls**, not HTTP calls to another service, so:

- The LLM is a **shared module** (`acoustic_ai/llm/`), loaded once and called
  by the generation orchestrator (parser) and the Layer E aggregator (report
  writer). It is **not** a registry layer/attempt — it generates no
  seed-based audio and has no `handler.py`.
- It rides the same `127.0.0.1` bind and the same pre-warm machinery as the
  audio layer defaults. **No Ollama/vLLM daemon, no `:11434`/`:8001`, no extra
  health-check or firewall surface.**

Running Ollama/vLLM on a localhost-only port would also work (still no public
port), but it adds a second process to babysit for no benefit at MVP scale, and
vLLM's default ~90% VRAM pre-allocation fights the audio models for the single
GPU. In-process is the consistent, lighter choice here.

---

## 4. VRAM budget (serverB, 16 GB)

Pre-warm loads every layer default on boot
([on_demand_ai_worker.md](../setup/server/on_demand_ai_worker.md)), so the
resident audio stack is roughly:

| Resident on boot | ~VRAM |
|---|---|
| AudioLDM2 base + 16 LoRA adapters (Layer A) | ~7 GB |
| AudioGen-medium (Layer C) | ~3 GB |
| CLAP `laion/clap-htsat-unfused` (Layer E) | ~0.6 GB |
| **Audio subtotal** | **~10.5 GB** |

That leaves ~5.5 GB, and the Layer A diffusion step needs activation headroom
on top of weights. So:

- **4-bit 3B (~2.5 GB)** → ~13 GB resident, ~3 GB headroom for diffusion. **Safe.**
- 4-bit 7B (~5 GB) → ~15.5 GB resident; diffusion activations would OOM. **Not resident-safe** without the upgrade lever (§7).

The LLM never runs *simultaneously* with diffusion (parser runs *before* Layer
A/C, report writer runs *after* the detectors), so the only cost is its
resident weights — which 2.5 GB covers.

---

## 5. Model choice + license note

**Primary — `Qwen/Qwen2.5-3B-Instruct` (4-bit).** Best capability-per-GB at
this size; strong JSON + instruction-following. License caveat: Qwen2.5's
**3B and 72B** ship under the Qwen *research* license (the 1.5B / 7B / 14B are
Apache-2.0). Fine for a university research prototype; flag it if the team
needs a permissive license.

**License-clean alt — `microsoft/Phi-3.5-mini-instruct` (3.8B, MIT).** Fully
permissive, strong instruction-following/reasoning for its size, also
resident-safe at 16 GB. Use this if license cleanliness matters more than the
last few points of JSON quality.

Default to **Qwen2.5-3B** for raw JSON quality, or **Phi-3.5-mini** for MIT.
Either is genuinely strong enough for the bounded job in §2.

---

## 6. Guardrails

Small models have two soft spots here; both are handled without a bigger model:

1. **Faithfulness (report writer).** Small models occasionally embellish
   (invent a species, overstate confidence). The LLM renders from a *closed*
   fused JSON, so **post-validate** that every species/number in the prose
   appears in the input JSON; reject + retry on a miss. This matters more than
   model size.
2. **Valid JSON (parser).** Use **constrained / grammar decoding** against the
   parse-result schema ([prompt_parser_policy.md §5](prompt_parser_policy.md))
   — do not rely on prompt-only "please return JSON."
3. **Prose richness (immersive register).** Few-shot the canonical samples
   already in [analysis_synthesis_policy.md §5](analysis_synthesis_policy.md);
   that pulls a 3B's prose up substantially. If demos still feel flat, that is
   the cue for §7.

Determinism: run the **parser at temperature 0** (greedy) for reproducible
contracts; the immersive report register may use a small temperature for warmth.

---

## 7. Upgrade lever

If the 3B's immersive prose underperforms on real demos, move to
`Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) **without buying a bigger GPU**:

- **Selective pre-warm** — `AI_PREWARM=layer_a`, keep AudioGen lazy → frees
  ~3 GB for a resident 7B; or
- **Load/unload** the LLM around the diffusion step (free before `generate`,
  reload for the report).

Ship 3B first, measure the immersive register on demos, escalate only if needed.

---

## 8. Open follow-ups

- [ ] Add `bitsandbytes` (and optional `lm-format-enforcer`) to
  `acoustic_ai/requirements.txt` when the service is wired in — deferred so a
  scaffold commit doesn't pull a new heavy dep into CI / the serverB sync.
- [ ] Build the **phenology lookup table**
  ([analysis_synthesis_policy.md §7](analysis_synthesis_policy.md)) — a hard
  prerequisite for the parser validity gate and aggregator fusion, and
  deliberately *not* the LLM's job.
- [ ] Wire `acoustic_ai/llm.warm()` into the server lifespan pre-warm (opt-in
  via `AI_LLM_PREWARM`, default off) once the model id is confirmed on serverB.
- [ ] Decide constrained-decoding backend (`lm-format-enforcer` vs `outlines`)
  against the chosen model's tokenizer.
