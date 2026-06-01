# Implementation Plan — E-A Ambient Analysis · **mvp_1: k-NN retrieval + season probe**

| | |
|---|---|
| **Attempt ID** | `lucas__mvp_1__clap_knn_probe` |
| **Layer / role** | `layer_e` — Analysis · E-A ambient context |
| **Stage** | `mvp_1` (first mvp; graduates the smoke bake-off winner) |
| **Backbone** | **Frozen LAION-CLAP** (`laion/clap-htsat-unfused` via `transformers.ClapModel`) |
| **Scope** | Full path: trained probe + **server / Express / frontend wiring** |
| **Author / date** | lucas · 2026-06-01 |
| **Branch** | `model/lucas/layer-e-mvp-1` (to be created; trained-checkpoint stage) |

> Graduates from the three-attempt smoke bake-off. Verdict
> ([smoke_4 README](../lucas__smoke_4__clap_cell_plus_knn/README.md#bake-off-verdict)):
> **k-NN retrieval (smoke_3) is the winner.** Cell-match (smoke_2) is
> dominated; 50/50 fusion (smoke_4) regressed on the headline metric. This
> mvp keeps the k-NN engine, attacks its one weak axis (season) with a
> trained probe, and carries smoke_4's head-agreement as a cheap confidence
> / OOD signal.

---

## 1. Purpose

Ship the **Analysis-mode E-A path** end-to-end: an uploaded ambient clip →
estimated `(season, diel, hour, month)` + similar real recordings +
confidence, served to the frontend. The smokes proved the engine offline;
this mvp wires it into the running system and lifts the season ceiling.

**What carries over (proven in smokes):**
- frozen-CLAP 48 kHz / 10 s-window / mean-pool / L2-norm embedding contract
- k-NN over a source-clip-disjoint index for diel / hour / month + neighbours
- the `pipeline_design.md` E-A report schema, matched 1:1 by smoke_3

**What this mvp adds:**
1. a **trained linear probe** for *season* (the 0.52 weak axis)
2. the **agreement / OOD gate** (from smoke_4) as the confidence signal
3. **long-upload windowing** (real uploads are not clean 10 s segments)
4. **server + Express + frontend wiring** (all deferred by the smokes)

---

## 2. Smoke results this builds on (shared 391-segment source-disjoint val)

| metric | baseline | smoke_2 | **smoke_3 (k-NN)** | smoke_4 fused |
|---|---|---|---|---|
| season acc | 0.25 | 0.455 | **0.524** | 0.529 |
| diel acc | 0.25 | 0.632 | **0.683** | 0.639 |
| cell top-1 | 0.063 | 0.338 | **0.435** | 0.422 |
| hour MAE (h) | 4.93 | — | **2.25** | (= k-NN) |
| month MAE (mo) | 2.82 | — | **1.96** | (= k-NN) |
| agree gate | — | — | — | acc 0.565 vs 0.336 |

Season is the lone sub-bar miss for every method — a frozen-CLAP embedding
ceiling, not a data-quantity artefact. The probe (PLAN §5) is the lever.

---

## 3. Architecture — one embedding, three heads

```
upload.wav
  └─ window into 10 s chunks → CLAP embed each → mean-pool → q (512-d, L2)
       ├─ k-NN over index   → diel (vote), hour, month (circular blend), similar_clips
       ├─ season probe(q)   → season (4-way softmax) + calibrated confidence
       └─ cell-match anchors→ head-A cell  ──┐
       k-NN cell vote ───────────────────────┴─ agreement → confidence / OOD flag
```

- **diel / hour / month / neighbours:** smoke_3 k-NN, unchanged. k=5, τ=0.1.
- **season:** the probe replaces the k-NN season vote (k-NN keeps everything
  else). This is the minimal, targeted change — we only override the axis
  k-NN is weakest on.
- **confidence / OOD:** smoke_4 agreement (cell-match head vs k-NN cell vote)
  + probe softmax margin. Low-confidence / disagreement / off-distribution
  uploads are flagged, not silently guessed.

Report shape (superset of `pipeline_design.md` E-A):
```json
{
  "estimated_conditions": {"season": "...", "diel_bin": "...", "hour": 0.0, "month": 0.0},
  "season_source": "probe",
  "similar_clips": [{"segment_id": "...", "source_clip": "...", "similarity": 0.0}],
  "confidence": 0.0,
  "head_agreement": true,
  "ood_flag": false
}
```

---

## 4. Data plan — current-data v0, enlarged-set drop-in

**Decision (2026-06-01):** the enlarged dataset is the goal but is **not yet
built**. mvp_1 therefore ships a **v0 baseline on the current 1,982 Bowra
segments**, with the architecture deliberately decoupled from the data so the
larger set is a **drop-in rebuild** (re-embed → rebuild index → retrain probe;
no analyzer/serving code change).

| Stage | Data | Action |
|---|---|---|
| v0 (now) | 1,982 segments (`lucas__smoke_4__vae_baseline/data/ambient/`) | build index + train probe; establish mvp baseline numbers |
| v1 (later) | enlarged set (more Bowra recordings, then more sites) | re-run the same scripts; DVC re-version index + probe; re-eval |

**Prerequisite tracked separately:** the data-enlargement effort (download +
manifest + segment prep). It gates production-quality season numbers but
**does not block** wiring the pipeline on v0.

Index + probe inputs are **source-clip-disjoint** split (seed 42), as in the
smokes, so val numbers stay honest and comparable.

---

## 5. Season probe

Frozen CLAP; train a tiny head on **cached embeddings** (cheap; local MPS or
serverB T4; seed 42).

1. **Linear probe** — softmax(4) on the 512-d CLAP vector, cross-entropy on
   the train split. First choice (cheap, low overfit risk on 1,982).
2. **MLP probe** — one hidden layer, only if linear underfits.
3. **Class imbalance** — weight the loss by inverse cell frequency
   (source-thin cells: summer/autumn/winter morning).

**Gate:** the probe must beat the k-NN season baseline (0.524) on the held-out
val split by a clear margin; otherwise keep the k-NN season vote and record
the negative result. Probe checkpoint → `model/candidates/lucas/mvp_1__clap_knn_probe/`
(DVC binary + git `params.yaml` / `metrics.json` / README).

---

## 6. Analyzer + serving wiring

**`code/handler.py` `analyze(audio_path) -> report`** — the E-A entry point:
load index + probe + anchors once; window the upload; emit the §3 report.

- **Windowing:** uploads are arbitrary length and contain events / weather.
  Slide 10 s windows, embed, mean-pool to one `q` for the ambient estimate;
  the OOD flag handles event-heavy / non-Bowra input.
- **`registry.yaml`** entry so `GET /layers` exposes it to the frontend.
- **FastAPI** upload endpoint on the AI server (multipart wav → report JSON).
- **Express** `POST /api/analysis` forwarding to the AI server (mirrors the
  generation contract; server owns model paths / k / τ / thresholds).
- **Frontend** Analysis-mode UI: upload → render estimated conditions,
  confidence, OOD flag, and the similar-clip list.

Per the [Layer A dev-generation contract](../../../../../CLAUDE.md) precedent,
the server owns all model parameters; the client sends only the audio.

---

## 7. Offline evaluation + mvp bar

Same seed-42 source-clip-disjoint val split as the smokes (direct
comparability). Report **probe vs k-NN-season** head-to-head plus the full
env metrics.

| metric | mvp bar |
|---|---|
| season acc | **> 0.60** (clear lift over k-NN 0.524; stretch 0.70) |
| diel acc | ≥ 0.683 (no regression vs smoke_3) |
| hour MAE | < 2.5 h |
| month MAE | < 2.0 mo |
| OOD gate | agreement separates correct vs incorrect (acc gap > 0.15) |

Plus an **end-to-end smoke**: a real long upload through the live FastAPI →
Express → frontend path returns a well-formed report.

---

## 8. Promotion criteria (candidate → production)

Per [conventions §5.4](../../../../../.claude/context/conventions.md). Promote
to `model/production/layer_e_ambient/` only after: the §7 bar is met on v0,
the end-to-end path works, and a short listening / sanity audit of the
neighbour evidence on held-out uploads. Production card carries the audit
section. **v0 may ship as the served candidate without promotion** — a
production slot is created only on an explicit promotion decision.

---

## 9. Risks / open questions

- **Season ceiling persists on v0.** Probe may not clear 0.60 on single-site
  1,982 — season cues at one dry-woodland site may be genuinely weak in CLAP.
  Mitigation: this is exactly what the enlarged set (§4 v1) is for; record the
  v0 number honestly and treat the probe as the mechanism, data as the fuel.
- **Index ≠ upload domain.** Index is cleaned ambient-only; uploads carry
  events/weather. The OOD flag is the honesty valve; consider a secondary eval
  on raw 300 s windows.
- **Probe overfit on 1,982.** Keep it linear; inverse-frequency weighting;
  watch train/val gap. Escalate to MLP only with evidence.
- **Serving cold-start.** CLAP + index + probe load once at server start;
  confirm memory headroom alongside the Layer A/C models on serverB.

---

## 10. Dependencies

**No new model dependency** — frozen CLAP is already in the stack (shared with
AudioLDM2 generation). Probe uses `torch` (present). New work is wiring
(FastAPI endpoint, Express route, frontend view), not new libraries. Update
`requirements.txt` in-change only if anything is added.

---

## 11. Definition of done

- Probe trained, `metrics.json` shows probe-vs-k-NN season on the held-out
  split; checkpoint under `model/candidates/lucas/mvp_1__clap_knn_probe/`.
- `handler.py analyze()` returns the §3 report on a long upload; registry +
  FastAPI + Express + frontend Analysis mode wired and demoable end-to-end.
- README "Results" states the §7 bar verdict on v0 and the enlarged-set
  rebuild procedure.
- Index + probe artefacts DVC-tracked; metadata git-tracked.
