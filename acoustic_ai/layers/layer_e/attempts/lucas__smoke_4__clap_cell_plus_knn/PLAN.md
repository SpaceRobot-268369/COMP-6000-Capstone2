# Implementation Plan — E-A Ambient Analysis · **Approach 4: cell-match + k-NN (fused)**

| | |
|---|---|
| **Attempt ID** | `lucas__smoke_4__clap_cell_plus_knn` |
| **Layer / role** | `layer_e` — Analysis · E-A ambient context |
| **Stage** | `smoke_4` (lucas/layer_e) |
| **Backbone** | **Frozen LAION-CLAP** (`laion/clap-htsat-unfused` via `transformers.ClapModel`) |
| **Scope** | Python module + **offline eval only** — no server / Express / frontend wiring |
| **Author / date** | lucas · 2026-06-01 |
| **Branch** | `feat/lucas/layer-e-ambient-analysis` |

> Third of a **three-attempt bake-off**. It **composes** the other two:
> [`smoke_2__clap_cell_match`](../lucas__smoke_2__clap_cell_match/PLAN.md)
> (closed-set cell label) +
> [`smoke_3__clap_knn_env`](../lucas__smoke_3__clap_knn_env/PLAN.md) (k-NN env
> estimate), and asks **whether combining them beats either alone** and whether
> their **agreement** is a useful confidence / out-of-distribution signal. Shared
> foundation (§2) is identical to the siblings.

---

## 1. Purpose / hypothesis

**The "reverse prompt" question, full-report form.** Produce both outputs from
one CLAP embedding and fuse them:
- a **discrete cell label + caption** (the inverse-of-generation prompt), and
- a **continuous env estimate + similar real clips** (the retrieval evidence),

plus a **cross-check**: does the k-NN-derived cell agree with the nearest-anchor
cell? Agreement is a cheap, free confidence and out-of-distribution detector — if
the two independent heads disagree, the input is likely off-distribution (an
event-heavy or non-Bowra clip) and the report should say so.

**Hypotheses:**
1. **Fusion ≥ best single head** on cell accuracy (averaging the cell-match
   posterior with a k-NN cell vote reduces variance).
2. **Head agreement correlates with correctness** — high-agreement queries are
   reliably right; disagreement flags low confidence / OOD.

**Cost framing (the real question this attempt answers):** is the richer report
worth running two heads, or does one head already dominate? That verdict decides
which design graduates to `mvp`.

---

## 2. Shared foundation (common to all three attempts)

### 2.1 Embedding backbone
- **Model:** `laion/clap-htsat-unfused` via `transformers.ClapModel` +
  `ClapProcessor` — same CLAP as inside `cvssp/audioldm2`; **no new dependency**
  (`transformers >= 4.40` already in `requirements.txt`).
- **Interpreter:** `acoustic_ai/.venv` (MPS) only. **One CLAP load** feeds both
  heads — the embedding is computed once per query.
- **Audio contract:** 48 kHz mono, 10 s windows, mean-pool, L2-norm.
- **Frozen** in Phase 1.

### 2.2 Data inputs (all verified present)
| Use | Path | Notes |
|---|---|---|
| Cell anchors | `site_257_training_manifest.csv` (audio prototypes) + `registry.yaml` cells (text) | 16 anchors — from Approach 1 |
| k-NN index | `…/lucas__smoke_4__vae_baseline/data/ambient/ambient_index.csv` + `ambient_segments/*.wav` | 1,982 segments on disk (DVC) — from Approach 2 |
| Ground truth | training manifest + ambient index (held-out) | season/diel/hour/month |

### 2.3 Determinism
Seed **42**. `.npy` → DVC; csv/json/md → git.

---

## 3. Approach mechanics (this attempt)

Reuse the two heads (built per Approaches 1 & 2) over a **single shared query
embedding** `q`:

- **Head A — cell anchors:** `sim_A = softmax(q·Aᵀ / τ_A)` over 16 cells →
  posterior `p_A` (cell-match).
- **Head B — k-NN over the index:** top-k neighbours → per-cell vote → posterior
  `p_B`, plus the continuous `(hour, month)` blend and `similar_clips`.

**Fusion + cross-check:**
```python
p_fused   = normalize(w_A * p_A + w_B * p_B)        # w_A,w_B tuned on val (start 0.5/0.5)
cell_hat  = argmax(p_fused)
agreement = (argmax(p_A) == argmax(p_B))            # independent-head consensus
confidence = blend(max(p_fused), mean_topk_sim, agreement)
```

Report (superset of both siblings):
```json
{
  "predicted_cell": "summer_afternoon",
  "season": "summer", "diel": "afternoon",
  "caption": "<cell's locked generation prompt>",
  "estimated_conditions": {"season": "...", "diel_bin": "...", "hour": 0.0, "month": 0.0},
  "similar_clips": [{"segment_id": "...", "similarity": 0.0}],
  "head_agreement": true,
  "confidence": 0.0,
  "ood_flag": false        // raised when heads disagree AND both confidences low
}
```

---

## 4. Implementation workflow

```
code/
  clap_backbone.py        # SHARED: one embedding per query, fed to both heads
  build_anchors.py        # 16 cell anchors (from Approach 1)
  build_ambient_index.py  # 1,982-segment index (from Approach 2)
  ambient_fused.py        # run both heads on shared q; fuse; agreement/OOD; report JSON  (E-A entry point)
  eval.py                 # per-head + fused metrics + agreement analysis -> metrics.json + report.md
  train_combiner.py       # Phase 2 only (see §6)
data/
  anchors.npy  index_embeddings.npy  index_meta.csv   # .npy DVC, csv git
  splits/…                                            # git, seed-42
README.md  metrics.json
```

Steps:
1. **Scaffold** + README. Reuse the sibling modules' logic (attempts are
   self-contained, so copy rather than import across attempt folders — per the
   "no shared `common/`" convention).
2. **`clap_backbone.py`** — shared embedding (computed **once** per query).
3. **`build_anchors.py`** + **`build_ambient_index.py`** — produce both
   reference structures.
4. **`ambient_fused.py`** — both heads on shared `q`, fuse, agreement/OOD,
   emit report.
5. **`eval.py`** — score **Head A alone, Head B alone, and Fused** on the same
   held-out split (apples-to-apples); compute agreement↔correctness correlation.
6. **OOD probe** — feed deliberately off-distribution inputs (event-heavy Layer C
   clips, a generated Layer A bed from a different cell) and confirm
   `head_agreement` drops / `ood_flag` fires.
7. **Audit + decide** Phase 2.

**Not in this plan (deferred):** registry entry, `analyze()` handler, FastAPI
upload endpoint, Express route, frontend.

---

## 5. Offline evaluation design

Same seed-42, **source-clip-disjoint** held-out split as the siblings (so the
three are directly comparable).

| Metric | Purpose |
|---|---|
| Cell top-1/top-3: **A vs B vs Fused** | does fusion beat the best single head? (Hypothesis 1) |
| Season / diel acc: A vs B vs Fused | per-axis fusion benefit |
| Hour / month circular MAE (Head B / fused) | continuous env quality |
| **Agreement rate** + **acc\|agree vs acc\|disagree** | is consensus a real confidence signal? (Hypothesis 2) |
| **OOD detection** AUROC | agreement/confidence separating in-dist vs. event-heavy inputs |

**Baselines:** the two single heads are the baselines fusion must beat — the
whole point of this attempt.

**Smoke success bar:**
- Fused cell top-1 **≥ max(Head A, Head B)** (no regression; ideally +2 pts), **and**
- `acc | agreement` materially **>** `acc | disagreement` (agreement is
  informative), **and**
- OOD probe: disagreement/ood_flag fires on clearly off-distribution inputs.

If fusion does **not** beat the best single head and agreement carries no signal,
that is a **valid negative result** → recommend the simpler winning sibling and
do **not** graduate this attempt.

---

## 6. Phase 2 — optional fine-tuning (only if useful)

Frozen backbone; learn the **combiner** on cached embeddings (seed 42, MPS):
1. **Learned fusion** — logistic regression / small MLP over
   `[p_A, p_B, mean_topk_sim, agreement]` → cell posterior, instead of fixed
   `w_A/w_B`. Calibrates the blend and the confidence.
2. Optionally inherit Approach 1's probe and Approach 2's regression head as the
   two head front-ends, then fuse.
3. **CLAP LoRA** — last resort, gated (breaks shared-embedding-space property).

Trigger only if fixed-weight fusion underperforms or confidence is miscalibrated.

---

## 7. Risks / open questions
- **Correlated heads.** A and B share the *same* CLAP embedding, so their errors
  may correlate → fusion gains and "independent" agreement are weaker than they
  look. Measure error correlation explicitly; temper the OOD claim accordingly.
- **Double the moving parts** for possibly marginal gain — this attempt must
  justify its own complexity or be dropped.
- **Inherits both siblings' risks** — CLAP domain gap (Approach 1 §7) and
  index/query domain mismatch + self-retrieval leakage (Approach 2 §7). Apply the
  same source-clip-disjoint split and clean/raw eval caveats.

---

## 8. Dependencies
**None new for Phase 1.** Same stack as the siblings. Update `requirements.txt`
in-change if Phase 2 adds anything.

---

## 9. Definition of done (this attempt)
- `metrics.json` + `report.md` scoring **A vs B vs Fused** on one shared
  held-out split, plus agreement↔correctness and OOD results.
- README "Results" answers both hypotheses with numbers.
- **Bake-off verdict:** explicitly recommend one of the three designs
  (`cell_match` / `knn_env` / `fused`) to graduate to `mvp` and be wired to the
  server — including "fusion not worth it, pick the simpler one" if that's what
  the data says.
