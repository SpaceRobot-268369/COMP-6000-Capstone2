# LLM-OSS Layer — Implementation Plan

> **Scope:** stand up the in-process LLM-OSS service on serverB and expose the
> APIs the generation/analysis workflows need. This is the **how/sequence**;
> the **what/why** (model choice, VRAM budget, rationale) lives in
> [llm_layer_config.md](llm_layer_config.md). Policies that govern behavior:
> [prompt_parser_policy.md](prompt_parser_policy.md) (parser) and
> [analysis_synthesis_policy.md](analysis_synthesis_policy.md) (report writer).
>
> **Status at start of plan:** the module scaffold
> [`acoustic_ai/llm/`](../../../acoustic_ai/llm/) exists (config, service,
> prompt stubs, README) but is **not wired into any endpoint** and its deps are
> **not installed**. Nothing calls it yet.
>
> **Implementation status (updated):** code for Phases 1–7 is **landed and
> tested model-free** — skills loader, constrained-JSON service, parser + gate,
> report writer + faithfulness guard, the `/generation/parse` and
> `/analysis/narrative` endpoints, opt-in `/analyze` narrative + LLM pre-warm,
> backend proxies, the scene-page tone toggle, and 16 unit tests. **Remaining is
> serverB-only** (Phase 8 + the model-dependent parts of Phases 1/5/6): install
> deps in `.venv`, pre-download the HF model, validate 16 GB VRAM, flip
> `AI_LLM_PREWARM=on`, deploy. **Deferred:** authoring the real skill files
> (owner: Lucas) and the species phenology table (§10).

---

## Contents

1. [Guiding constraints](#1-guiding-constraints)
2. [Target architecture](#2-target-architecture)
3. [API surface](#3-api-surface)
4. [Phased implementation](#4-phased-implementation)
5. [Model provisioning on serverB](#5-model-provisioning-on-serverb)
6. [VRAM validation](#6-vram-validation)
7. [Testing](#7-testing)
8. [Deploy](#8-deploy)
9. [Risks & rollback](#9-risks--rollback)
10. [Dependencies & sequencing](#10-dependencies--sequencing)

---

## 1. Guiding constraints

| Constraint | Source | Consequence |
|---|---|---|
| In-process, no new port | [llm_layer_config.md §3](llm_layer_config.md) | LLM is a Python module call, not an HTTP service. APIs ride the existing `:8000` app. |
| One model serves both consumers | parser + report policies | Single `LLMService` singleton; parser and report writer share it. |
| 16 GB GPU, shared with audio | [llm_layer_config.md §4](llm_layer_config.md) | 4-bit 3B resident (~2.5 GB). Validate headroom before enabling pre-warm. |
| Base models load from HF Hub, not DVC | confirmed in `layer_a .../handler.py` (`from_pretrained("cvssp/audioldm2")`) | The Qwen/Phi weights are pulled from HF Hub and cached — **not** DVC-tracked. Only trained checkpoints are DVC. |
| LLM does no reasoning | [analysis_synthesis_policy.md §3](analysis_synthesis_policy.md) | Aggregator fusion + phenology checks stay deterministic. LLM only extracts/encodes/renders. |
| Parse must precede (and gate) generation | [prompt_parser_policy.md §3,§5](prompt_parser_policy.md) | A standalone parse step is required so reject/correct happens before the GPU runs. |

---

## 2. Target architecture

```
Frontend (:5173)
   │  prompt
   ▼
Express backend (:4000)  ──── POST /api/generation/parse ───┐
   │                                                        │
   │  resolved contracts {seed, season, diel, weather, …}   │
   ▼                                                        ▼
FastAPI AI server (serverB 127.0.0.1:8000, via ai-tunnel)
   ├── POST /generation/parse   ── LLMService.complete_json() ──► parse-result JSON   (NEW)
   ├── POST /generation/render  ── A → B → C → D  (contracts in)                      (unchanged contract)
   └── POST /layers/layer_e/.../analyze ── detectors → aggregator → LLMService.complete() ► report  (MODIFIED)

   acoustic_ai/llm/  (in-process)
     get_service() ─ singleton LLMService (Qwen2.5-3B-Instruct, 4-bit)
       ├── complete_json()  → parser
       └── complete()       → report writer
```

Two LLM call-sites, one model. The parser is a **separate cheap endpoint** so
the UI can show assumptions / handle `corrected` / `rejected` before paying for
generation. `render` stays contract-driven (NL → contracts already happened in
`parse`).

### 2.1 Skill integration — separate files (decided)

Skills (the per-job instruction sets) are authored as **separate files**, not
hard-coded Python strings. The current `acoustic_ai/llm/prompts/*.py` stubs are
**renamed/replaced by `acoustic_ai/llm/skills/`** (a runtime artifact —
**not** `.claude/skills/`, which is the dev-agent's). A thin Python loader reads
+ caches each skill file and supplies it as the **system message**; the
job-specific data is the **user message**. Division of labor:

| In a **skill file** (static) | In **Python** (the service) |
|---|---|
| Instruction body ("how to do this job") | Load + cache the skill file |
| Few-shot examples (e.g. canonical report samples) | Inject job data as the user message |
| Human-readable schema *description* | The **enforced** output schema (constrained decoding) |
| Register variants (analytical / immersive) | Faithfulness validation + retries |

Call flow:

```
load skill file (cached)            ──► system message
job data (prompt / fused JSON /     ──► user message
          deterministic gate findings)
        │
        ▼
LLMService.complete()  /  complete_json(schema=…)
```

Key rule: the **skill (system message) is static**; the **job payload (user
message) is dynamic**. So the schema is enforced in code (the decoder), while
its prose description lives in the file. Format: markdown; add `jinja2` slots
only if a skill needs variable injection beyond the user message (most won't).
Skills are versionable as files (`skills/parser/v1.md`, …) for A/B iteration.

---

## 3. API surface

### 3.1 `POST /generation/parse` — NEW (FastAPI `:8000`)

Runs the three-stage parser ([prompt_parser_policy.md](prompt_parser_policy.md)).
**No audio generated.**

```jsonc
// request
{ "prompt": "a misty autumn dawn with light rain" }

// 200 — parse-result schema (policy §5)
{
  "status": "ok | corrected | rejected",
  "note": "human-readable explanation of defaults/corrections",
  "filled_defaults": ["events:empty"],
  "layer_a": { "season": "autumn", "diel": "dawn" },
  "layer_b": { "weather_type": "rain", "intensity": "light", "duration_s": 10.0 },
  "layer_c": { "species": [], "density": "sparse" }
}
```

- `rejected` → `layer_*` null, `note` carries a suggested alternative prompt.
- Validity gate uses **deterministic** site/phenology checks (passed into the
  user message); the LLM encodes/narrates them, never overrides them.

### 3.2 `POST /generation/render` — contract unchanged

Already exists (`OrchestratedGenerationRequest` → `registry.orchestrate_generation`).
Consumes resolved contract fields (`seed, season, diel, weather_type, intensity, …`).
**Optional sugar (decide in Phase 3):** also accept a raw `prompt` to parse
internally for a one-shot path — but the standalone `/parse` is the required
one.

### 3.3 `POST /layers/layer_e/attempts/{id}/analyze` — MODIFIED

Today returns the aggregator/head report JSON. Add a **render step**: after the
aggregator produces the fused JSON, call the report writer to attach prose.

```jsonc
// request: multipart file=<audio>   (+ optional ?register=analytical|immersive)
// 200 (added field)
{ "ok": true, "report": { /* fused JSON, unchanged */ },
  "narrative": { "register": "analytical", "text": "…" },   // NEW
  "attempt": { … } }
```

- Faithfulness guard: validate the narrative introduces no species/number
  absent from `report`; reject + retry on a miss ([llm_layer_config.md §6](llm_layer_config.md)).
- `register` is driven by the **scene-page tone toggle** (§3.5). Default
  analytical (dev) / immersive (demo) per policy.

### 3.4 `POST /analysis/narrative` — NEW (backs the tone toggle)

Re-render prose from an **already-computed** fused report in a different
register, **without re-running the detectors** (the fused JSON is identical; only
the wording changes). This is what makes the top-of-page tone toggle (§3.5)
instant and cheap.

```jsonc
// request
{ "report": { /* fused report JSON from a prior analyze */ },
  "register": "analytical | immersive" }

// 200
{ "ok": true, "narrative": { "register": "immersive", "text": "…" } }
```

- LLM-only call; same faithfulness guard as `analyze`.
- Alternative (MVP shortcut): have `analyze` return **both** registers' text in
  one response so the toggle is pure client-side. Trade-off: renders the unused
  register every time. Prefer this lazy `/analysis/narrative` endpoint unless
  the double-render cost is negligible.

### 3.5 Scene-page tone toggle (frontend)

On the **immersive scene page**, a **register toggle at the top-center of the
page** lets the user switch the report tone between **Analytical** and
**Immersive** at any time. Toggling re-renders the narrative only (via
`/analysis/narrative`, §3.4) against the cached fused report — it must **not**
re-upload audio or re-run detectors. Default selection = immersive on the demo
scene page.

### 3.6 Express backend (`:4000`)

| Route | Action |
|---|---|
| `POST /api/generation/parse` (NEW) | Thin proxy → FastAPI `/generation/parse`. Returns parse-result to frontend. |
| `POST /api/generation` (exists) | Unchanged orchestration; now called with contracts the frontend got from `/api/generation/parse`. |
| `POST /api/analysis/narrative` (NEW) | Thin proxy → FastAPI `/analysis/narrative` (§3.4). Backs the scene-page tone toggle. |
| Analysis route (when added) | Pass through `register` to FastAPI `analyze`. |

---

## 4. Phased implementation

### Phase 1 — Dependencies (serverB + requirements)
- [ ] Add to `acoustic_ai/requirements.txt`: `bitsandbytes` (4-bit), and either
      `lm-format-enforcer` or `outlines` (constrained JSON). Keep `transformers`/
      `accelerate` (already present).
- [ ] Pin minimum versions; verify they resolve in `acoustic_ai/.venv` on serverB
      (CUDA build of `bitsandbytes`).

### Phase 2 — Complete the service module
- [ ] Implement constrained JSON in `LLMService.complete_json()` (replace the
      tolerant-extraction fallback) keyed on the parse-result schema.
- [ ] Add a faithfulness validator helper (narrative ⊆ fused-JSON facts) for the
      report writer.
- [ ] Build the **skill loader**: rename `acoustic_ai/llm/prompts/` →
      `acoustic_ai/llm/skills/`; replace the string-builder stubs with a thin
      loader that reads + caches a skill file and returns it as the system
      message (pattern decided in §2.1). Keep the enforced output schema in code.
- [ ] **TODO (deferred — owner: Lucas): author the skill files.** Each LLM job
      is sent with a pre-written skill (instructions + few-shot) as a **separate
      file** under `acoustic_ai/llm/skills/` (§2.1). The real skills — parser
      skill (3-stage contract) and report skill (two registers, few-shotting the
      canonical samples from
      [analysis_synthesis_policy.md §5](analysis_synthesis_policy.md)) — are
      written later. Wiring (Phases 3–4) can proceed against placeholder skill
      files and swap in the authored ones when ready.

### Phase 3 — Parser front-end logic + `/generation/parse`
- [ ] Build the parser orchestration (Stage 1 default-fill → Stage 2 gate →
      Stage 3 decode), calling `svc.complete_json()` and merging deterministic
      gate results.
- [ ] Add `POST /generation/parse` route to `acoustic_ai/server/server.py`
      (+ Pydantic request/response models).
- [ ] (Decide) optional raw-`prompt` path on `/generation/render`.

### Phase 4 — Wire the report writer into analysis
- [ ] After aggregator fusion, call `svc.complete()` with the fused JSON +
      `report_system_prompt(register)`.
- [ ] Add `register` query param to `analyze`; default analytical (dev) /
      immersive (demo) per policy.
- [ ] Add `POST /analysis/narrative` (§3.4) — re-render prose from a cached
      fused report in a chosen register, no detector re-run.
- [ ] Apply the faithfulness guard (both `analyze` and `/analysis/narrative`).

### Phase 5 — Pre-warm + boot integration
- [ ] Add opt-in `llm.warm()` to the server lifespan, gated by `AI_LLM_PREWARM`
      (default **off**), so current boots are unchanged until validated.
- [ ] Confirm `/health` stays green while the LLM warms in the background thread.

### Phase 6 — Backend proxy + frontend loop
- [ ] Express `POST /api/generation/parse` and `POST /api/analysis/narrative`
      proxies (§3.6).
- [ ] Frontend (generation): prompt box → parse → show
      `note`/`filled_defaults`/contracts (and `corrected`/`rejected` UX) →
      confirm → `render`.
- [ ] Frontend (analysis): **scene-page tone toggle at top-center** (§3.5) —
      switch Analytical ⇄ Immersive; re-renders narrative via
      `/api/analysis/narrative` against the cached fused report (no re-upload,
      no detector re-run). Default immersive on the demo scene page.

### Phase 7 — Tests (see §7) and Phase 8 — Deploy (see §8).

---

## 5. Model provisioning on serverB

The LLM weights come from **HF Hub**, same as AudioLDM2/AudioGen — **not DVC**.

- [ ] Confirm `Qwen/Qwen2.5-3B-Instruct` (ungated) downloads on serverB; cache
      in the standard HF cache (`~/.cache/huggingface`). Outbound HF works
      (the audio base models already pull this way).
- [ ] Pre-download once so first boot isn't slow:
      `./acoustic_ai/.venv/bin/python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"`
- [ ] `AI_LLM_MODEL` env switches to `microsoft/Phi-3.5-mini-instruct` (MIT) if
      license cleanliness is required.
- [ ] No `model/` checkpoint, no `registry.yaml` entry — the LLM is a service,
      not a registry attempt.

---

## 6. VRAM validation

Before enabling `AI_LLM_PREWARM` on serverB:

- [ ] Boot with `AI_PREWARM=all` (audio defaults) **+** load the LLM; capture
      `nvidia-smi` peak during a Layer A generate.
- [ ] Confirm total resident + diffusion activations stay under 16 GB with
      headroom (target ≤ ~14 GB resident; expect ~13 GB per
      [llm_layer_config.md §4](llm_layer_config.md)).
- [ ] If tight: drop to `microsoft/Phi-3.5-mini` 4-bit, or apply the upgrade
      lever pattern in reverse (selective `AI_PREWARM`).

---

## 7. Testing

| Level | What | Where |
|---|---|---|
| Unit (import-safe) | prompt builders, JSON extraction/validation, schema conformance — no model load | local venv (mock `complete`) |
| Unit (parser logic) | default-fill, gate swap/reject branches with a stubbed `complete_json` | local |
| Integration (model) | real `complete_json` returns schema-valid JSON; report faithfulness guard catches a planted hallucination | serverB |
| Endpoint | `POST /generation/parse` ok/corrected/rejected; `analyze` returns narrative | serverB via tunnel |
| VRAM smoke | §6 | serverB |

Keep model-dependent tests serverB-only (local Mac is MPS, not the deploy target).

---

## 8. Deploy

Standard serverB flow ([on_demand_ai_worker.md](../setup/server/on_demand_ai_worker.md)):

- [ ] Merge to `main` → CICD `sync-server-b-models` runs `pip install -r
      acoustic_ai/requirements.txt` (picks up `bitsandbytes` etc.), `dvc pull`,
      and restarts uvicorn.
- [ ] First post-deploy boot: LLM downloads from HF (slow once) unless
      pre-downloaded in §5.
- [ ] Flip `AI_LLM_PREWARM=on` only after §6 passes.

---

## 9. Risks & rollback

| Risk | Mitigation / rollback |
|---|---|
| VRAM OOM with LLM resident | Keep `AI_LLM_PREWARM` off by default; lazy-load. Fall back to Phi-3.5-mini. Worst case: load/unload around diffusion. |
| `bitsandbytes` CUDA build mismatch on serverB | Validate in venv before merge; fall back to fp16 (no 4-bit) if a 3B still fits. |
| Invalid JSON from parser | Constrained decoding (Phase 2) makes invalid JSON impossible; extraction fallback as backstop. |
| Report hallucination | Faithfulness guard rejects + retries. |
| Validity gate needs phenology data that doesn't exist | **Blocker** — see §10. Ship parser with site/biome + season-window stubs; full gate after the table lands. |
| Deploy restart interrupts in-flight jobs | Accepted trade-off (documented); deploy when serverB is idle. |

Rollback = revert the wiring commit; the scaffold module is inert when nothing
calls it, and `AI_LLM_PREWARM` defaults off.

---

## 10. Dependencies & sequencing

- **Hard prerequisite for the full validity gate:** the **species phenology
  table** ([analysis_synthesis_policy.md §7](analysis_synthesis_policy.md)) does
  not exist yet. Parser Stages 1 + 3 and the report writer do **not** need it;
  Stage 2's fauna-plausibility check does. Sequence: ship parse with
  domain/biome + coarse season checks, then tighten the gate once the table
  lands. Track as its own task.
- **Deferred (owner: Lucas) — LLM job skills.** The per-job instruction sets
  ("skills") are authored later as **separate files** under
  `acoustic_ai/llm/skills/` (integration pattern in §2.1). Phases 3–4 wire
  against placeholder skill files and swap in the real ones when ready — not a
  blocker for wiring.
- **Independent of:** Layer A/B/C/D internals — the contracts already exist.
- **Order:** Phase 1 → 2 → (3 ∥ 4) → 5 → 6 → 7 → 8. Phase 4 (report writer) can
  proceed in parallel with Phase 3 (parser) since they share only the service.
```
