# lucas__smoke_2__clap_cell_match

Frozen-CLAP closed-set cell match for Layer E ambient analysis. Embeds an
input ambient clip with `laion/clap-htsat-unfused` and returns the nearest
of the 16 `(season, diel)` cells the Layer A bank conditions on, plus that
cell's locked generation prompt as the "reverse prompt".

Design and bake-off context live in [PLAN.md](PLAN.md).

## Approach in one paragraph

Build 16 cell anchors and classify each query by nearest-anchor cosine.
Two anchor variants are evaluated head-to-head: **text anchors** (pure
zero-shot — `CLAP_text(cell_caption)` for each of the 16 prompts in
`registry.yaml → layer_a.lucas__prod_1__per_cell_loras.params.cells`) and
**audio prototypes** (mean of `CLAP_audio(seg)` over the train split's
segments per cell). Both anchor sets are L2-normalised; query embedding is
the same CLAP audio vector, so the two heads share a single forward pass.

## Data deviation from PLAN.md

PLAN §2.2 lists the `site_257_training_manifest` (6,148 clips) as the
audio-prototype source. Those entries point to `downloaded_clips/*.webm`
files that are neither DVC-tracked nor on disk in any lucano clone today.
This attempt therefore uses the **`ambient_segments` pool**
(`acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/`,
1,982 segments, DVC-pullable) for both prototype construction and eval.
This aligns smoke_2 with smoke_3 / smoke_4 on a single data source, which
also satisfies the source-clip-disjoint split requirement.

## Layout

```
code/
├── paths.py               # shared path constants + CELL_ORDER
├── clap_backbone.py       # 48 kHz / 10 s / mean-pool / L2-norm CLAP wrapper
├── embed_segments.py      # one-time embedding cache for all 1,982 segments
├── build_anchors.py       # seed-42 source-clip-disjoint split + 2 anchor sets
├── ambient_cell_match.py  # CellMatcher.classify() — E-A entry point
└── eval.py                # text vs audio head metrics, confusion matrices
data/
├── embeddings_cache.npy   # DVC after first run — (1982, D) float32, L2-normed
├── embeddings_meta.json   # segment_ids in row order + model_id, sample_rate
├── splits/{train,val}.csv # git — seed-42 stratified, source-clip-disjoint
├── anchors_text.npy       # DVC after first run — (16, D)
├── anchors_audio.npy      # DVC after first run — (16, D)
├── anchors_meta.json      # git — provenance + prompts + counts
└── confusion_{text,audio}.png  # DVC after eval
```

## Reproduce

From the attempt root, on a machine where `acoustic_ai/.venv` is set up and
`ambient_segments` is `dvc pull`-ed:

```bash
cd acoustic_ai/layers/layer_e/attempts/lucas__smoke_2__clap_cell_match

../../../../../acoustic_ai/.venv/bin/python code/embed_segments.py
../../../../../acoustic_ai/.venv/bin/python code/build_anchors.py
../../../../../acoustic_ai/.venv/bin/python code/eval.py
```

`embed_segments.py` is idempotent — re-run with `--force` to recompute.

Single-clip classification:

```bash
../../../../../acoustic_ai/.venv/bin/python code/ambient_cell_match.py \
  /path/to/ambient_clip.wav --variant audio
```

## Smoke success bar (from PLAN.md §5)

- Season acc **≥ 70%**, diel acc **≥ 55%** for at least one variant.
- Cell top-3 **≥ 50%**.
- Confusion matrix shows structured errors (adjacent season/diel), not noise.

## Results

_Empty until eval has been run. Fill in once `metrics.json` exists with
text-vs-audio numbers, the bar verdict, and a short read of the confusion
structure._

## Bake-off verdict

_Filled in once smoke_3 and smoke_4 have also been scored on the same
split._
