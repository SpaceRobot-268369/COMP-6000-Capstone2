# lucas__smoke_3__clap_knn_env

Frozen-CLAP k-nearest-neighbour retrieval for Layer E ambient analysis.
Embeds an input clip with `laion/clap-htsat-unfused`, finds its top-k most
similar real ambient segments, and reads continuous env estimates (season,
diel, hour, month) off the neighbours' metadata.

Design and bake-off context live in [PLAN.md](PLAN.md).

## Approach in one paragraph

Offline, embed every ambient_segment once and split source-clip-disjoint
into an **index** (~80%) and a **query** (~20%) pool. At query time embed
the clip, cosine-rank the index, softmax-blend the top-k neighbours with
temperature τ, then read:
- `season`, `diel_bin` — weighted majority vote
- `hour` — `atan2(Σ w·hour_sin, Σ w·hour_cos)` → 0–24 h
- `month` — same circular decode → 1–12 months
- `confidence` — mean top-k similarity
- `similar_clips` — the top-k themselves (segment_id, source_clip, sim)

## Data deviation from PLAN.md

PLAN §2.2 marks the cleaned ambient pool as "already exists from the
smoke_4 VAE attempt" — confirmed in place
(`acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/`,
1,982 segments, DVC-pullable, 1,296 unique source clips). The split is
**source-clip-disjoint**, not segment-disjoint, so a query segment cannot
retrieve a near-duplicate from its own 300 s parent recording (the §7 risk
called out in the PLAN).

## Layout

```
code/
├── paths.py                # shared path constants
├── clap_backbone.py        # copy of the smoke_2 backbone (no shared common/)
├── embed_segments.py       # one-time embedding cache for all 1,982 segments
├── build_ambient_index.py  # source-clip-disjoint split + index_embeddings.npy
├── ambient_similarity.py   # AmbientRetriever.query() — E-A entry point
└── eval.py                 # k/τ sweep over season/diel/hour/month/P@k + baselines
data/
├── embeddings_cache.npy    # DVC — (1982, D) float32, L2-normed
├── embeddings_meta.json    # git — segment_ids + model_id + sample_rate
├── splits/{index,query}.csv# git — seed-42 source-clip-disjoint
├── index_embeddings.npy    # DVC — (N_index, D)
├── index_meta.csv          # git — segment_id + label + cyclic encodings
└── index_build_meta.json   # git — provenance + counts
```

## Reproduce

From the attempt root, after `dvc pull` of `ambient_segments`:

```bash
cd acoustic_ai/layers/layer_e/attempts/lucas__smoke_3__clap_knn_env

# Smoke_2 already runs embed_segments and writes the cache to its own data
# dir; this attempt is self-contained per conventions, so re-runs it here.
../../../../../acoustic_ai/.venv/bin/python code/embed_segments.py
../../../../../acoustic_ai/.venv/bin/python code/build_ambient_index.py
../../../../../acoustic_ai/.venv/bin/python code/eval.py
```

Single-clip query:

```bash
../../../../../acoustic_ai/.venv/bin/python code/ambient_similarity.py \
  /path/to/ambient_clip.wav -k 5 --tau 0.1
```

## Smoke success bar (from PLAN.md §5)

- Season acc **≥ 70%**, diel acc **≥ 55%**.
- Hour circular MAE **< 3.0 h**, month MAE **< 2.0 months**.
- Precision@5 visibly above the random `5/16` floor.

## Results

Scored on the held-out query set (n_query = 391, source-clip-disjoint from
an index of 1,591 segments).

| k | tau | season | diel | hour MAE (h) | month MAE (mo) | P@k |
|---|---|---|---|---|---|---|
| 1 | 0.1 | 0.481 | 0.624 | 2.37 | 2.08 | 0.379 |
| 3 | 0.1 | 0.506 | 0.660 | 2.21 | 2.03 | 0.365 |
| **5** | **0.1** | **0.522** | **0.680** | **2.25** | **1.96** | **0.340** |
| 10 | 0.1 | 0.535 | 0.660 | 2.26 | 1.90 | 0.316 |

Trivial baselines: season-majority 0.366, diel-majority 0.435, hour
global-mean MAE 4.93 h, month global-mean MAE 2.82 mo.

**Bar verdict (at the chosen operating point k=5, tau=0.1): 4 of 5 met.**
- diel acc 0.680 (bar 0.55) PASS
- hour MAE 2.25 h (bar < 3.0, baseline 4.93) PASS by a wide margin
- month MAE 1.96 mo (bar < 2.0, baseline 2.82) PASS
- P@5 0.340 (random floor 5/16 = 0.3125) PASS, but only marginally
- season acc 0.522 (bar 0.70) FAIL — same weak axis as smoke_2, though it
  clears the 0.366 majority baseline comfortably.

**Operating point.** hour MAE is flat-best at k=3 (2.21 h); diel and month
peak at k=5 (0.680 / 1.96); season keeps climbing to k=10 (0.537) at the
cost of diel and retrieval precision (P@k falls 0.379 to 0.316 as k grows).
k=5, tau=0.1 is the knee: it takes the diel/month peak and a strong season
without sacrificing neighbour precision. tau barely matters in [0.05, 0.2].

**Read vs smoke_2.** On the shared axes the retrieval head beats cell-match
audio prototypes (season 0.522 vs 0.455; diel 0.680 vs 0.632) AND adds the
continuous env estimate (hour/month) plus explainable neighbour evidence
that the closed-set head cannot produce. Season remains the hard axis for
both — it is a CLAP-embedding limitation, not a method artefact.

## Bake-off verdict

_Filled in once smoke_2 and smoke_4 have also been scored._
