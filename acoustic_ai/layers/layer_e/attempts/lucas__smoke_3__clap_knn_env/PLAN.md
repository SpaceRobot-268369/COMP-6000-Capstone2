# Implementation Plan — E-A Ambient Analysis · **Approach 2: CLAP k-NN → env**

| | |
|---|---|
| **Attempt ID** | `lucas__smoke_3__clap_knn_env` |
| **Layer / role** | `layer_e` — Analysis · E-A ambient context |
| **Stage** | `smoke_3` (lucas/layer_e) |
| **Backbone** | **Frozen LAION-CLAP** (`laion/clap-htsat-unfused` via `transformers.ClapModel`) |
| **Scope** | Python module + **offline eval only** — no server / Express / frontend wiring |
| **Author / date** | lucas · 2026-06-01 |
| **Branch** | `feat/lucas/layer-e-ambient-analysis` |

> One of a **three-attempt bake-off** for ambient analysis. Siblings:
> [`smoke_2__clap_cell_match`](../lucas__smoke_2__clap_cell_match/PLAN.md)
> (closed-set cell label) and
> [`smoke_4__clap_cell_plus_knn`](../lucas__smoke_4__clap_cell_plus_knn/PLAN.md)
> (both heads + agreement). The shared frozen-CLAP foundation (§2) is identical
> across all three; this one differs by retrieving **real neighbour clips** and
> reporting **continuous env estimates**.

---

## 1. Purpose / hypothesis

**The "reverse prompt" question, retrieval form.** Instead of forcing the input
into one of 16 discrete cells, locate it in soundscape space: **embed the clip,
find its nearest real ambient segments, and read the answer off the neighbours'
metadata** — estimated season, diel, and *continuous* hour-of-day and
month-of-year, plus the list of similar recordings. This is the design
`pipeline_design.md § E-A` specifies almost verbatim.

**Hypothesis:** clips close in CLAP space share acoustic context, so neighbours'
env metadata averages into a useful estimate — and crucially gives a
**continuous** read (hour ≈ 6.3, month ≈ 10) that the discrete cell-match
(Approach 1) cannot, plus *explainable* evidence ("here are the 5 real Bowra
recordings it sounds most like").

**"Reverse prompt" deliverable:** the estimated `(season, diel, hour, month)`
can be templated back into a generation-style caption (optional, §3), but the
primary output is the **env estimate + neighbour evidence**, not a single label.

---

## 2. Shared foundation (common to all three attempts)

### 2.1 Embedding backbone
- **Model:** `laion/clap-htsat-unfused` via `transformers.ClapModel` +
  `ClapProcessor` — the **same CLAP inside `cvssp/audioldm2`**, so **no new
  dependency** (`transformers >= 4.40` already in `requirements.txt`).
- **Interpreter:** `acoustic_ai/.venv` (MPS) only.
- **Audio contract:** 48 kHz mono; load via `librosa`/`torchaudio`, resample,
  window into 10 s chunks, embed, **mean-pool**, L2-normalise.
- **Frozen:** `eval()` + `no_grad()` in Phase 1.

### 2.2 Data inputs (all verified present)
| Use | Path | Notes |
|---|---|---|
| **Reference index** | `…/lucas__smoke_4__vae_baseline/data/ambient/ambient_index.csv` + `ambient_segments/*.wav` | **1,982** cleaned ambient segments **on disk** (DVC-materialised); cols `segment_id, source_clip, t_start, t_end, diel_bin, season, hour_sin/cos, month_sin/cos` |
| Eval ground truth | same index (held-out split) and/or `site_257_training_manifest.csv` | hour/month/season/diel labels |

> The cleaned ambient pool the pipeline doc marks "to be built" **already
> exists** (built by the smoke_4 VAE attempt). This attempt consumes it directly.

### 2.3 Determinism
Seed **42** everywhere. Embedding matrix + index → `.npy` **DVC-tracked**;
csv/json/md → git.

---

## 3. Approach mechanics (this attempt)

**Offline — build the index once:**
- Embed all 1,982 ambient segments → `E` (N×D, L2-normalised), aligned with a
  metadata table carrying `segment_id, source_clip, diel_bin, season,
  hour_sin/cos, month_sin/cos`. Cache `E` as `.npy` (DVC).

**At query time:**
1. `q = CLAP_audio(clip)`.
2. Cosine similarity `q·Eᵀ`; take **top-k = 5** neighbours.
3. **Blend** by `w = softmax(sim/τ)`, τ ≈ 0.1:
   - `season`, `diel` → weighted majority vote of neighbours.
   - `hour` → `atan2(Σ w·hour_sin, Σ w·hour_cos)` → decode to 0–24 h.
   - `month` → `atan2(Σ w·month_sin, Σ w·month_cos)` → decode to 1–12.
   - `confidence` → mean top-k similarity (and/or neighbour label agreement).
4. Emit the E-A report (matches `pipeline_design.md` schema):
```json
{
  "estimated_conditions": {"season": "...", "diel_bin": "...", "hour": 0.0, "month": 0.0},
  "similar_clips": [{"segment_id": "...", "source_clip": "...", "similarity": 0.0}],
  "confidence": 0.0
}
```

**Optional "reverse prompt" add-on:** template
`estimated_conditions` → a generation-style caption string
(`"{diel} {season} ambient soundscape, Bowra dry woodland, Australia …"`) so the
output can seed Layer A generation. Clearly marked optional; the primary metric
is env accuracy, not caption quality.

---

## 4. Implementation workflow

```
code/
  clap_backbone.py        # SHARED design: embed_audio(paths)->E
  build_ambient_index.py  # embed 1,982 segments -> index_embeddings.npy + index_meta.csv
  ambient_similarity.py   # query(clip)->report JSON (the E-A entry point; matches pipeline_design name)
  eval.py                 # leave-out retrieval metrics -> metrics.json + report.md
  train_head.py           # Phase 2 only (see §6)
data/
  index_embeddings.npy    # DVC  (N×D segment matrix)
  index_meta.csv          # git  (aligned metadata)
  splits/{index,query}.csv# git  (seed-42 stratified by cell)
README.md  metrics.json
```

Steps:
1. **Scaffold** attempt dirs + README (conventions §6) + artifact-tier
   `.gitkeep`s.
2. **`clap_backbone.py`** — shared embedding contract (§2.1).
3. **`build_ambient_index.py`** — embed all segments, write
   `index_embeddings.npy` + `index_meta.csv`; DVC-add the `.npy`.
4. **Leave-out split** — partition the 1,982 segments into `index` / `query`
   (seed 42, stratified by cell). Query segments are removed from the index so
   no segment retrieves itself.
5. **`ambient_similarity.py`** — `query(path) -> report`; the future E-A handler
   entry point. Implements the top-k blend (§3).
6. **`eval.py`** — score the held-out query set (§5).
7. **Audit** — inspect whether high-similarity neighbours are genuinely the same
   context; tune `k` and `τ`. Record in README.
8. **Decide** Phase 2 (§6) only if the bar is missed.

**Not in this plan (deferred):** registry entry, `analyze()` handler, FastAPI
upload endpoint, Express route, frontend.

---

## 5. Offline evaluation design

**Split:** seed-42 stratified `index` / `query` over the 1,982 segments; query
items excluded from their own index.

| Metric | Definition | Baseline |
|---|---|---|
| **Hour circular MAE** | min angular dist between est. and true hour (h) | global-mean predictor; random (~6 h) |
| **Month circular MAE** | same, months | global-mean; random (~3 mo) |
| Season acc | k-NN majority vs. true | majority-class |
| Diel acc | k-NN majority vs. true | majority-class |
| **Precision@k** | fraction of top-k neighbours sharing the true cell | k/16 random |
| Confidence calibration | error vs. confidence bins | — |

**Sweeps:** `k ∈ {1,3,5,10}`, `τ ∈ {0.05,0.1,0.2}` — report the curve, pick the
knee.

**Smoke success bar:**
- Season acc **≥ 70%**, diel acc **≥ 55%**, **and**
- Hour circular MAE **beats the global-mean baseline by a clear margin** (target
  **< 3 h**), month MAE **< 2 months**, **and**
- Precision@5 visibly above the random `5/16` floor.

Targets are tunable after the first run; the gate is "neighbours carry real,
better-than-trivial env signal."

---

## 6. Phase 2 — optional fine-tuning (only if the bar is missed)

Backbone frozen; learn on cached embeddings (local MPS, seed 42):
1. **Supervised regression head** — small MLP on the CLAP vector predicting
   `(hour_sin, hour_cos, month_sin, month_cos)` (+ optional season/diel logits),
   then k-NN/regress in the *learned* space. Sharpens the env read.
2. **Metric learning** — supervised-contrastive / triplet objective pulling
   same-cell segments together, improving retrieval precision before the blend.
3. **CLAP audio-encoder LoRA** — last resort, gated; changes the embedding space
   and breaks the "shared with generation" property. Flag before doing it.

Trigger only on a missed §5 bar. The user's "new attempt with fine-tuning"
choice is met by the regression/metric head; LoRA is reserved.

---

## 7. Risks / open questions
- **Index ≠ query domain.** The index is *cleaned ambient-only* segments; real
  analysis uploads contain events/weather. Eval on clean segments is optimistic;
  note this and, if time allows, a secondary eval on raw 300 s manifest windows.
- **Self-retrieval / leakage.** Segments from the same `source_clip` are near-
  duplicates — split by `source_clip`, not just by segment, to avoid trivially
  easy neighbours inflating metrics. **(Important — bake into the split.)**
- **Circular decoding bugs.** Hour/month live as sin/cos; verify the
  `atan2` decode round-trips on known rows before trusting MAE.
- **Coverage gaps.** Source-thin cells give sparse neighbourhoods → low
  confidence there; surface per-cell metrics.

---

## 8. Dependencies
**None new for Phase 1.** `transformers`, `librosa`, `torchaudio`, `numpy`,
`scipy` already present. Update `requirements.txt` in-change if Phase 2 adds
anything.

---

## 9. Definition of done (this attempt)
- `metrics.json` + `report.md` with the k/τ sweep, env MAEs, retrieval
  precision, scored on a **source-clip-disjoint** held-out split.
- README "Results" states bar met/not + chosen `k`, `τ`.
- One-paragraph **bake-off verdict** vs. siblings (does continuous env + neighbour
  evidence beat a discrete cell label for the analysis UX?) — feeds the
  graduate-to-mvp decision.
