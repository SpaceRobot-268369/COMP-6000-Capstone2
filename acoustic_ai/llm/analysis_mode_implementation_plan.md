# Analysis Mode — End-to-End Implementation Plan

> **Scope:** turn the (already backend-complete) Layer E analysis stack into a
> **fully functioning product mode** — uploaded clip → three detector heads →
> deterministic aggregator → LLM-OSS narration → immersive scene. This is the
> **how/sequence**; the **what/why** lives in
> [pipeline_design.md (Analysis Mode)](../../.claude/context/ai/pipeline_design.md),
> [analysis_synthesis_policy.md](../../.claude/context/ai/analysis_synthesis_policy.md)
> (aggregator fusion + report-writing policy), and
> [llm_layer_config.md](../../.claude/context/ai/llm_layer_config.md) (model choice).
> The LLM service that powers narration is stood up by the companion
> [llm_layer_implementation_plan.md](./llm_layer_implementation_plan.md) — this
> plan consumes it; it does not re-do it.

---

## Status at start of plan

The analysis **backend is substantially complete**; the gaps are *integration*,
not new models.

| Piece | State |
|---|---|
| LLM-OSS service (Qwen2.5-3B 4-bit) | **Up + baseline VRAM-validated on serverB**; co-residency with generation still needs the §5 check (llm plan §6) |
| Report writer + faithfulness guard + skills | **Landed, tested model-free**, but must be tightened for aggregator `common_name` fields (`acoustic_ai/llm/report.py`, `faithfulness.py`) |
| Aggregator — deterministic fusion | **Done** (`songke__smoke_3__analysis_aggregator`) |
| E-A ambient head | Real model (`lucas__mvp_2__clap_knn_probe_enlarged`, candidate) |
| E-B weather head | Real model (`murphy__mvp_1__weather_direct_detection`) |
| E-C events head | Real model (`songke__prod_1__e_c_species_event_detector`) |
| FastAPI `/analysis/run`, `/analysis/narrative`, per-head `/analyze` | Exist |
| Backend proxies (`/api/analysis`, `/api/analysis/narrative`, `/api/layers/:/analyze`) | Exist |
| ImmersivePage + ToneToggle (re-render via `/analysis/narrative`) | Wired |

### The five gaps this plan closes

1. **No narrative on the orchestrated path.** `/analysis/run`
   ([`acoustic_ai/server/server.py`](../server/server.py)) returns the fused
   report + `head_reports` but **never calls `write_report`** — narration is only
   opt-in on the per-head `/analyze`. The product entry produces no prose.
2. **The product page is a demo mock.** `/analysis` →
   [`HomePage.jsx`](../../frontend/src/pages/HomePage.jsx): **both** the 4 preset
   cards **and** the upload zone are fake — upload guesses the environment from
   `file.name` via `resolvePrompt`, runs canned `composeNarration`, and a fixed
   `setTimeout` "analysis" animation, then navigates with `fromDemo: true`. The
   real Layer E stack is only reachable from the *dev* page.
3. **serverB prewarm/runtime check still pending.** `AI_LLM_PREWARM` defaults
   off; one co-residency `nvidia-smi` check is still outstanding (llm plan §6).
   `/analysis/run` will narrate force-on and will not depend on the per-head
   `AI_LLM_NARRATIVE` flag.
4. **Phenology table deferred.** Aggregator season/diel fusion leans on the E-C
   handler's baked-in phenology blocks, not the site-257 table
   ([analysis_synthesis_policy.md §7](../../.claude/context/ai/analysis_synthesis_policy.md)).
   *Quality follow-up, non-blocking — see §6.*
5. **Narration contract not pinned.** The aggregator returns `decision`,
   `llm_input`, and deterministic `narration`; the report writer accepts any
   dict; the faithfulness guard currently reads `observations.events[].label`.
   This plan pins which payload the LLM sees and which event names the guard
   treats as allowed.

---

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Narrative on `/analysis/run` | **Inline render** (call `write_report` after fusion; default register `immersive`). | The scene needs prose immediately; ToneToggle only fires on a *register switch*. Avoids the extra round-trip / empty-narrative flash of the lazy path. |
| Report-writer payload | API call-sites pass the full fused `report`, but `write_report` renders the compact `report["decision"]` payload. | `decision` is the narration-ready contract. `llm_input.task` hard-codes tone language, while `write_report(register)` owns the register-specific system prompt. |
| Narrative failure fallback | `/analysis/run` returns deterministic `report["narration"].summary` if the LLM is down or the faithfulness guard rejects the final retry. | Product mode should still reach the scene with honest prose; return `narrative_error` / `narrative_violations` for debugging instead of `null`. |
| Faithfulness species source | The guard allows species from both snake-case IDs and display names: `observations.events[]`, `decision.detected_calls[]`, and `llm_input.decision.detected_calls[]`. | Aggregator events use `label: "ninox_boobook"` plus `common_name: "Southern Boobook"`. Guarding only `label` falsely rejects faithful prose. |
| HomePage entries | **Make everything real** — upload **and** all 4 preset cards run the real pipeline. | Authentic end-to-end; presets exercise the same `/api/analysis` path as uploads. |
| Preset audio source | Frontend fetches each preset's real `audioUrl` (already backend-served from the `expected/` bank) → blob → same `analyseAudio()` path. | No new backend endpoint; presets and uploads share one code path. |
| Express proxy | **No Express register parsing.** Keep `/api/analysis` as raw multipart stream passthrough. | `body: req, duplex: "half"` already forwards every frontend `FormData` field, including `register`; parsing would buffer the upload stream and regress large-file handling. |
| Phenology table | **Quality follow-up**, two-phase (coarse source first, manual site-257 later). | Pipeline runs end-to-end with the current baked-in phenology; the table grounds (does not unblock) fusion. Needs domain input + a sync with songke. |

---

## Target architecture (analysis path)

```
Frontend /analysis (HomePage)
   │  uploaded clip  OR  preset clip (fetched → blob)
   ▼
Express backend :4000  POST /api/analysis                      (unchanged raw multipart passthrough)
   ▼
FastAPI :8000  POST /analysis/run  (multipart file, register)  (MODIFIED: attach narrative)
   ├── E-A ambient  ┐
   ├── E-B weather  ├─ analyze() per head
   ├── E-C events   ┘
   ├── aggregator   → deterministic fused report JSON
   └── llm.write_report(report, register)
       ├─ LLM prompt payload = report.decision
       └─ fallback payload = report.narration.summary
   ▼
   { ok, report, head_reports, attempts, narrative }
   ▼
Frontend  navigate("/immersive", { report, narrative, register })
   ├── scene params from report.decision (fallback: inferred_context/observations)
   └── ToneToggle  POST /api/analysis/narrative  (register switch only, no re-run)
```

One model, two call-sites (parser elsewhere; report writer here). The
`/analysis/narrative` endpoint already exists and stays the lazy re-render path
for the tone toggle.

---

## Handoff contracts to implement exactly

### Narrative payload and guard

`/analysis/run` and `/analysis/narrative` should keep accepting the full fused
`analysis_aggregator.v1` report from API callers. Inside `write_report`, derive
the prompt payload like this:

```python
source_report = report or {}
if source_report.get("schema_version") == "analysis_aggregator.v1":
    prompt_payload = source_report.get("decision") or source_report
elif source_report.get("schema_version") == "analysis_llm_input.v1":
    prompt_payload = source_report.get("decision") or source_report
else:
    prompt_payload = source_report
```

Do **not** pass the full `llm_input` wrapper as the prompt payload. Its `task`
string currently says "immersive, third-person perspective narration with an
analytical tone", which conflicts with `register`. The register-specific skill
loaded by `write_report(register)` is the source of truth for tone.

The faithfulness guard validates against `source_report`, not just
`prompt_payload`. Allowed species/event names must be collected from all of:

- `report.observations.events[].label`
- `report.observations.events[].common_name`
- `report.decision.detected_calls[].label`
- `report.decision.detected_calls[].common_name`
- `report.llm_input.decision.detected_calls[].label`
- `report.llm_input.decision.detected_calls[].common_name`

Normalize by lowercasing, replacing `_` / `-` with spaces, and collapsing
whitespace before comparison. That makes `ninox_boobook`, `Ninox boobook`, and
`Southern Boobook` distinct allowed aliases for the same observed call. The UI
must display `common_name` when present; `label` remains the machine ID.

Fallback object for `/analysis/run`:

```jsonc
{
  "register": "immersive",
  "text": "The recording is best described as night with none weather...",
  "source": "deterministic_fallback",
  "faithful": true,
  "violations": []
}
```

Use this fallback when `write_report` raises or returns `faithful: false` after
its retry budget. Preserve `narrative_error` and/or `narrative_violations` in
the response so the failure is visible during review.

### Scene mapping table

The immersive engine wants:

```js
{ season, time, rain, rainAmount, wind, thunder, narration }
```

Map the fused report into that shape before navigating:

| Engine key | Primary source | Fallback / rule |
|---|---|---|
| `season` | `report.decision.season.value` | Else `report.inferred_context.season.estimate`; if missing or `undetermined`, use engine default `autumn`. |
| `time` | `report.decision.time_of_day.value` | Else `report.inferred_context.diel.estimate`; if missing or `undetermined`, use engine default `dawn`. |
| `rain` | `report.decision.weather.rain.label` | `true` when label is present and not `none`, or `intensity > 0.05`; otherwise `false`. |
| `rainAmount` | `report.decision.weather.rain.intensity` | Clamp to `[0, 1]`; if `rain` is true but intensity is absent, use `0.6`. |
| `wind` | `report.decision.weather.wind.intensity` | Clamp to `[0, 1]`; default `0`. The current world stores wind intensity even though the visual effect is minimal. |
| `thunder` | `report.decision.weather.thunder` | `true` when label is not `none`, `intensity > 0.05`, or `events.length > 0`. |
| `narration` | `response.narrative.text` | Else deterministic `report.narration.summary`; never use the old `composeNarration` output for real analysis. |

If a legacy `resolved.events` array is still needed for display-only code,
derive it conservatively: any E-C known bird species maps to `birdsong`;
labels/common names containing `cicada` map to `insects`; labels/common names
containing `cricket` map to `crickets`; de-duplicate. Do not use that coarse
array as the narration source — the LLM/deterministic narration should name the
actual `common_name` values.

### Field-name footguns

| Do this | Not this | Why |
|---|---|---|
| `overall_confidence` | `confidence` | Aggregator v1 intentionally has no top-level `confidence`; old UI fallbacks may use `overall_confidence ?? confidence` for legacy heads only. |
| `decision.time_of_day.value` → engine `time` | `inferred_context.diel` directly everywhere | `decision` is the compact downstream contract; `inferred_context` is the fallback/audit source. |
| `common_name` for display and guard aliases | Raw `label` in prose | E-C labels are snake-case IDs such as `ninox_boobook`. |
| `decision.weather.{rain,wind,thunder}.{label,intensity}` | `observations.weather` first | `decision.weather` has the scene-ready component shape. Use observations only as fallback/debug. |
| `report.narration` for deterministic fallback | `response.narrative` | `report.narration` is generated by the aggregator; `response.narrative` is the new LLM/fallback API field. |

### Preset blob path

Preset cards use the real sample catalog URL only as the source audio. To run
the same upload path as a user file:

```js
const presetRes = await fetch(preset.audioUrl, { credentials: "include" });
if (!presetRes.ok) throw new Error(`Could not load preset audio (${presetRes.status})`);
const blob = await presetRes.blob();
const analysisFile = new File([blob], "preset.wav", {
  type: blob.type || "audio/wav",
});
const data = await analyseAudio(analysisFile, {}, "immersive");
```

This is same-origin (`/api/layers/.../samples/.../audio.wav`), so no CORS work
is needed. Keep `preset.audioUrl` in page state for playback and source caption.

### Loading and error UX

There is no streaming analysis response. Keep `STAGES` as a cosmetic cycler
while the single `analyseAudio()` promise is pending; it must not determine
navigation timing. On error, stay on `/analysis`, clear the analyzing state, and
render the backend error message. Do not navigate to an immersive scene unless
the real `/api/analysis` call returns `ok: true`.

The backend request timeout is `AI_REQUEST_TIMEOUT_MS` (default `300000`, dev
and server compose default `540000`). If the frontend adds an `AbortController`,
set the client timeout to the same value plus a small buffer (for example
`545000` in dev) via a Vite env var. Do not introduce a shorter hard-coded
client timeout.

### Codex / local testing boundary

Codex runs on the local Mac path with MPS-capable Python, not on serverB. It can
edit the wiring and run import-safe, model-free tests with mocked LLM output,
following the existing fake-service unittest pattern in `test_llm_layer.py`.
It cannot validate real Qwen prose quality, VRAM co-residency, or live serverB
latency unless a human exposes that runtime.

Local tests to extend/run:

```bash
./acoustic_ai/.venv/bin/python -m unittest \
  acoustic_ai.tests.test_llm_layer \
  acoustic_ai.tests.test_analysis_orchestrator \
  acoustic_ai.tests.test_analysis_aggregator_fusion \
  acoustic_ai.tests.test_analysis_aggregator_adapters
```

Add focused cases for:

- `faithfulness.py`: `label: "ninox_boobook"` + `common_name: "Southern Boobook"`
  allows prose that says "Southern Boobook"; unobserved `galah` still fails.
- `report.py`: fake service captures the user message and proves the serialized
  prompt payload is `analysis_decision.v1`, not the whole aggregator report and
  not the `llm_input.task` wrapper.
- `/analysis/run`: mocked `registry.orchestrate_analysis` + mocked
  `write_report`; assert `register` is read as a FastAPI `Form` field,
  `narrative` is attached, and deterministic fallback is used on LLM failure.

Human/serverB checks remain: real prose, real faithfulness under the model,
request latency, and `nvidia-smi` co-residency.

---

## Phased implementation

### Phase 1 — Narrative inline on the orchestrated endpoint *(backend)*

- [ ] `registry.orchestrate_analysis(...)`
      ([`acoustic_ai/server/registry.py:631`](../server/registry.py)) — keep
      pure fusion; **do not** call the LLM here (keep registry import-light).
- [ ] `server.py` `orchestrated_analysis`
      ([`acoustic_ai/server/server.py:305`](../server/server.py)) — accept a
      `register` form field (default `immersive`); after `orchestrate_analysis`
      returns, call `from llm import write_report` and attach
      `narrative = write_report(result["report"], register)`. This endpoint is
      force-on for product mode; `AI_LLM_NARRATIVE` remains a per-head
      `/analyze` opt-in and must not gate `/analysis/run`.
- [ ] If `write_report` raises or returns `faithful: false`, attach the
      deterministic fallback from `result["report"]["narration"]["summary"]`
      and include `narrative_error` / `narrative_violations`.
- [ ] `report.py` — normalize full aggregator reports to the compact
      `decision` prompt payload internally; keep accepting older/full dicts so
      `/analysis/narrative` and tests do not need separate call contracts.
- [ ] `faithfulness.py` — collect allowed species from both `label` and
      `common_name` fields across `observations.events`, `decision.detected_calls`,
      and `llm_input.decision.detected_calls`.
- [ ] Response contract gains
      `narrative: { register, text, source, faithful, violations }`.

### Phase 2 — Backend proxy passthrough *(backend, Express)*

- [ ] `POST /api/analysis` ([`backend/src/index.js:1011`](../../backend/src/index.js))
      — **no code change required for register**. It already pipes the raw
      multipart stream through with `body: req, duplex: "half"`, so any frontend
      `FormData` field rides through.
- [ ] Do not add Express-side multipart parsing or buffering for this route.
- [ ] No change needed to `/api/analysis/narrative` (already proxies).

### Phase 3 — HomePage: make everything real *(frontend)*

- [ ] `analyseAudio(file, attempts, register)` in
      [`frontend/src/lib/api.js:104`](../../frontend/src/lib/api.js) — add an
      optional `register = "immersive"` arg appended to the FormData.
- [ ] [`HomePage.jsx`](../../frontend/src/pages/HomePage.jsx):
  - [ ] **Upload** — replace `handleAnalyzeUploadedFile` (filename guessing) with
        a real `analyseAudio(file, {}, "immersive")` call.
  - [ ] **Presets** — `handleSelectPreset` fetches the preset's `audioUrl` →
        `Blob`/`File` → same `analyseAudio()` path (no hardcoded resolved state).
  - [ ] Drop client-side `resolvePrompt` / `composeNarration` for the real path.
  - [ ] `STAGES` animation becomes a **real loading state** driven by the request
        lifecycle (cycle while awaiting the call, then navigate) rather than a
        fixed timer.
  - [ ] On success, `navigate("/immersive", { state: { report, narrative,
        register, audioUrl, sourceCaption, fromAnalysis: true,
        backPath: "/analysis" } })`.
  - [ ] Surface real errors (serverB down, 4xx/5xx) instead of always succeeding.

### Phase 4 — ImmersivePage consumes the real report *(frontend)*

- [ ] [`ImmersivePage.jsx`](../../frontend/src/pages/ImmersivePage.jsx) — read the
      fused `report` from page state; map to scene params:
  - `(season, time)` ← `report.decision.season.value` and
        `report.decision.time_of_day.value` (fallback to inferred context;
        `undetermined` → `autumn` / `dawn`)
  - weather ← `report.decision.weather.{rain,wind,thunder}` component labels
        and intensities (fallback to observation summaries only if decision is
        absent)
  - events ← display names from `report.decision.detected_calls[].common_name`
        when legacy event tags are needed
- [ ] Feed `report` + initial `narrative.text` + `register` to
      [`ToneToggle`](../../frontend/src/components/ToneToggle.jsx) (already wired
      to `/api/analysis/narrative` for switches).
- [ ] Keep a graceful fallback when `inferred_context` is `undetermined`
      (pick a neutral default cell; don't crash the scene).

### Phase 5 — serverB enablement *(ops)*

- [ ] During a Layer A generate with the LLM resident, capture `nvidia-smi` peak
      (the outstanding co-residency check, llm plan §6).
- [ ] Set `AI_LLM_PREWARM=on` so the first analysis doesn't pay a cold LLM load.
- [ ] Confirm `/health` stays green while the LLM warms in the background thread.

### Phase 6 — Phenology table *(follow-up, grounds fusion)*

- [ ] **6a** — author a coarse table (A2O / Xeno-canto seasonal+diel windows +
      `niche_specificity`) per the schema in
      [analysis_synthesis_policy.md §7](../../.claude/context/ai/analysis_synthesis_policy.md);
      make it the **single source of truth**.
- [ ] Point the aggregator fusion **and** songke's E-C handler at the table
      (the handler currently bakes phenology blocks in — coordinate so we don't
      fork the data).
- [ ] **6b** — replace entries with manually-validated site-257 windows
      (e.g. confirm "cicada → summer afternoon" against real recordings).

---

## API / contract changes

### `POST /analysis/run` — MODIFIED

```jsonc
// request: multipart  file=<audio>   (+ optional register=analytical|immersive, default immersive)
// 200 (added field)
{
  "ok": true,
  "report":       { /* fused JSON, unchanged */ },
  "head_reports": { "ambient": {…}, "weather": {…}, "events": {…} },
  "attempts":     { "ambient": …, "weather": …, "events": …, "aggregator": … },
  "narrative":    { "register": "immersive", "text": "…",      // NEW
                    "source": "llm", "faithful": true, "violations": [] }
}
```

- `narrative` should still be non-null on LLM failure by using
  `report.narration.summary` with `source: "deterministic_fallback"`.
- `narrative_error` and/or `narrative_violations` may be present when fallback
  was used.
- Faithfulness guard (narrative ⊆ report facts) is enforced by `write_report`
  using `label` and `common_name` aliases.

### `POST /api/analysis` (Express) — UNCHANGED
Raw stream passthrough. The frontend appends `register` to `FormData`; Express
does not parse or re-append it.

### Unchanged
`/analysis/narrative` (lazy re-render), per-head `/analyze`, all generation
endpoints.

---

## Files touched

| File | Change |
|---|---|
| [`acoustic_ai/server/server.py`](../server/server.py) | `register` field + inline `write_report` on `/analysis/run` |
| [`acoustic_ai/server/registry.py`](../server/registry.py) | none required (fusion stays LLM-free) |
| [`acoustic_ai/llm/report.py`](./report.py) | normalize full reports to `decision` prompt payload; expose deterministic fallback contract |
| [`acoustic_ai/llm/faithfulness.py`](./faithfulness.py) | allow `label` + `common_name` aliases from aggregator observations/decision |
| [`backend/src/index.js`](../../backend/src/index.js) | no code change required; keep raw multipart passthrough |
| [`frontend/src/lib/api.js`](../../frontend/src/lib/api.js) | `analyseAudio(file, attempts, register)` |
| [`frontend/src/pages/HomePage.jsx`](../../frontend/src/pages/HomePage.jsx) | real upload + preset analysis; real loading state |
| [`frontend/src/pages/ImmersivePage.jsx`](../../frontend/src/pages/ImmersivePage.jsx) | map fused report → scene; feed ToneToggle |
| [`acoustic_ai/tests/test_llm_layer.py`](../tests/test_llm_layer.py) | model-free tests for decision payload + common-name faithfulness |
| New/updated endpoint test | mocked `/analysis/run` narrative + fallback behavior |
| Phase 6 only | new phenology table + aggregator/E-C handler wiring |

---

## Testing

| Level | What | Where |
|---|---|---|
| Endpoint (no LLM) | `/analysis/run` returns report + `head_reports` + LLM or deterministic fallback narrative | local venv, mocked `write_report` |
| Integration (LLM) | `/analysis/run` returns a **faithful** narrative; planted hallucination is caught | serverB via tunnel |
| Frontend | upload → immersive renders real report; **preset** → same path; serverB-down shows an error, not a fake scene | local frontend → backend → tunnel |
| Tone toggle | switching register re-renders via `/api/analysis/narrative` with **no** re-upload / re-run | frontend |
| VRAM | co-residency `nvidia-smi` peak (Phase 5) | serverB |

LLM-dependent tests are **serverB-only** (local Mac is MPS, not the deploy
target). Codex/local can still run the import-safe unittest command in
["Codex / local testing boundary"](#codex--local-testing-boundary).

---

## Risks & rollback

| Risk | Mitigation / rollback |
|---|---|
| LLM down on the request path | try/except → deterministic `report.narration.summary` fallback + `narrative_error`; report still returns. |
| Faithfulness guard rejects final retry | Use deterministic fallback; surface `narrative_violations` for review. |
| Cold LLM load blows the backend timeout | Flip `AI_LLM_PREWARM=on` (Phase 5) before relying on inline narration. |
| `inferred_context` undetermined → scene can't pick a cell | Neutral default cell fallback (Phase 4). |
| Preset blob fetch / CORS | Presets are served by the same backend the app already fetches from; reuse existing `audioUrl`s. |
| Real path slower than the canned demo | Accepted (decision: everything real). Real loading state covers it; presets are still one click. |
| Fusion quality limited without phenology table | Non-blocking; Phase 6 grounds it. Fusion already defaults to "undetermined" on weak evidence. |

Rollback = revert the wiring commits. Phase 1 is inert behind the try/except;
HomePage can revert to the demo path if the product flow regresses.

---

## Dependencies & sequencing

- **Depends on:** the LLM-OSS service being live (llm plan — done) and the
  aggregator + 3 heads (done). No new models for Phases 1–5.
- **Order:** Phase 1 → 2 → 3 → 4 → 5. Phase 6 is independent and can run in
  parallel (own task; needs domain input + songke sync).
- **Coordination:** Phase 6 touches songke's E-C handler — agree the phenology
  table as the single source of truth before migrating.
