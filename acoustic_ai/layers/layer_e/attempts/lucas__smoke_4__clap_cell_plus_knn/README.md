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

_Empty until eval has been run. Filled in once `metrics.json` exists with
the A vs B vs Fused numbers and the agreement analysis._

## Bake-off verdict

_Recommendation across the three smokes lives here once all three have
been scored on the same source-clip-disjoint val split._
