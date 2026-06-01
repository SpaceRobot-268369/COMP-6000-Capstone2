# lucas__smoke_4__clap_cell_plus_knn

Composes smoke_2 (cell-anchor) and smoke_3 (k-NN env) on a **single
shared query embedding**, fuses their per-cell posteriors, and uses
**head agreement** as a cheap confidence / out-of-distribution signal.

Design and bake-off context live in [PLAN.md](PLAN.md).

## Approach in one paragraph

One CLAP audio embedding `q` per query feeds two heads. Head A: per-cell
softmax over cosine similarity to 16 cell anchors (text or audio
prototypes from smoke_2). Head B: top-k retrieval into the smoke_3
ambient index, blended into a per-cell vote. Posteriors are fused
`p_fused = norm(w_a · p_A + w_b · p_B)`. Disagreement between the two
argmaxes, combined with low fused confidence, raises an OOD flag —
useful because the heads index into the same embedding space from
different angles, so consistent answers are stronger evidence than
either alone.

## Data deviation from PLAN.md

Same as the sibling attempts: uses the `ambient_segments` pool (1,982
segments) rather than `downloaded_clips/*.webm`, so all three smokes
score on one source-clip-disjoint split.

## Layout

```
code/
├── paths.py                # shared
├── clap_backbone.py        # shared backbone (copy per "no shared common/")
├── embed_segments.py       # one-time embedding cache for all 1,982 segments
├── build_anchors.py        # 16 cell anchors (copied from smoke_2)
├── build_ambient_index.py  # k-NN index (copied from smoke_3)
├── ambient_fused.py        # AmbientFusedAnalyzer.analyze() — E-A entry point
└── eval.py                 # A vs B vs Fused + agreement / error-correlation
data/
├── embeddings_cache.npy    # DVC
├── splits/{train,val}.csv  # git — seed-42 source-clip-disjoint (shared with head A)
├── anchors_audio.npy       # DVC
├── anchors_text.npy        # DVC
├── index_embeddings.npy    # DVC
├── index_meta.csv          # git
└── confusion_{a,b,fused}.png  # DVC after eval
```

## Reproduce

```bash
cd acoustic_ai/layers/layer_e/attempts/lucas__smoke_4__clap_cell_plus_knn

../../../../../acoustic_ai/.venv/bin/python code/embed_segments.py
../../../../../acoustic_ai/.venv/bin/python code/build_anchors.py
../../../../../acoustic_ai/.venv/bin/python code/build_ambient_index.py
../../../../../acoustic_ai/.venv/bin/python code/eval.py
```

Single-clip analysis:

```bash
../../../../../acoustic_ai/.venv/bin/python code/ambient_fused.py \
  /path/to/ambient_clip.wav --variant audio -k 5
```

## Smoke success bar (from PLAN.md §5)

- Fused cell top-1 **≥ max(head A, head B)** (no regression; ideally +2 pts).
- `acc | agreement` materially **>** `acc | disagreement`.

A clean negative result — "fusion doesn't help, pick the simpler sibling"
— is a valid outcome and feeds the bake-off verdict.

## Results

Scored on the held-out val split (n_val = 391, source-clip-disjoint).
params: head_a=audio prototypes, k=5, tau=0.1, w_a=w_b=0.5.

| metric | head A (cell) | head B (knn) | fused |
|---|---|---|---|
| cell top-1 | 0.338 | **0.435** | 0.422 |
| cell top-3 | 0.619 | 0.688 | **0.706** |
| season acc | 0.455 | 0.524 | **0.529** |
| diel acc | 0.632 | **0.683** | 0.639 |
| mean confidence | 0.163 | 0.551 | 0.348 |

**Hypothesis 1 (fusion >= best single head): FAILS on the headline metric.**
Fused cell top-1 0.422 < head B 0.435, and fusion *hurts* diel (0.639 vs
0.683). It edges ahead only on cell top-3 (0.706) and season (ties, 0.529).
The 50/50 blend drags the stronger k-NN head toward the weaker cell-match
head. The smoke bar (fused top-1 >= max(A,B)) is not met.

**Hypothesis 2 (agreement signals correctness): PASSES.**
- agreement rate 0.376 (the two heads pick the same cell 37.6% of the time)
- acc | agree    = 0.565 (n=147)
- acc | disagree = 0.336 (n=244)
- error-correlation A vs B = 0.279 (only moderate — the heads are not
  redundant despite sharing one CLAP embedding)

Agreement is a real, cheap confidence / OOD signal: when both independent
heads concur, accuracy nearly doubles relative to disagreement.

Confusion matrices: `data/confusion_{a,b,fused}.png`.

## Bake-off verdict

**Winner: smoke_3 (`clap_knn_env`) — graduate it to mvp. Do NOT graduate
this fused attempt.**

Single-head k-NN (head B) is the strongest engine across the board — cell
top-1 0.435, season 0.524, diel 0.683, and well-calibrated confidence
(0.551) — while also delivering the continuous hour/month estimate
(hour MAE 2.25 h, month 1.96 mo, both crushing trivial baselines) and the
explainable neighbour evidence that cell-match cannot produce. Cell-match
(smoke_2) is strictly dominated. Fusion (smoke_4) does not beat k-NN on the
primary metric and regresses on diel, so the second head is not worth its
cost as a *classifier*.

**Carry forward (cheap, optional):** keep the cell-match head only as an
agreement-based confidence / OOD flag on top of the k-NN engine — run both,
surface `head_agreement` and gate low-confidence / off-distribution inputs
(acc jumps 0.336 -> 0.565 on agreement). This is a lightweight add-on to
the smoke_3 design, not a reason to ship the 50/50 fusion.

**Known ceiling:** season is the hard axis for all three methods
(best ~0.52) — a frozen-CLAP embedding limitation, not a method artefact.
The PLAN §6 linear probe on the cached embeddings is the obvious next
lever and is the recommended first mvp experiment.
