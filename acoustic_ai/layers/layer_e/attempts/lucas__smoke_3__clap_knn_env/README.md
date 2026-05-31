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

_Empty until eval has been run. Fill in once `metrics.json` exists with
the k/τ sweep, baselines, and the knee point._

## Bake-off verdict

_Filled in once smoke_2 and smoke_4 have also been scored._
