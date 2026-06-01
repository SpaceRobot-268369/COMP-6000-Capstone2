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

Scored on the held-out val split (n_val = 391, source-clip-disjoint from
train), softmax tau = 0.1.

| metric | text anchors | audio prototypes | bar |
|---|---|---|---|
| cell top-1 | 0.077 | **0.338** | — (chance 0.063) |
| cell top-3 | 0.340 | **0.619** | >= 0.50 PASS |
| season acc | 0.233 | **0.455** | >= 0.70 FAIL |
| diel acc | 0.547 | **0.632** | >= 0.55 PASS |
| mean confidence | 0.132 | 0.163 | — |

**Bar verdict: 2 of 3 met (audio prototypes).** cell top-3 and diel acc
clear their bars; season acc (0.455) falls short of 0.70 though it is well
above the 0.25 chance floor. Audio prototypes beat text anchors on every
metric — zero-shot text anchors are near chance on season (0.233) and cell
top-1 (0.077), confirming the locked captions carry non-acoustic tokens
(dates, temperatures) that CLAP cannot hear.

**Weak axis is season, not diel** — the reverse of the PLAN §7 prediction.
Diel (time-of-day) is the more separable axis in CLAP space here; season is
the confuser. Per-cell (audio): night/afternoon cells with many segments do
best (spring_night 0.69, autumn_afternoon 0.53, spring_morning 0.56,
summer_night 0.51), while source-thin or acoustically ambiguous cells
collapse to 0 (spring_afternoon, summer_dawn, autumn_morning). Errors are
structured (mass on same-diel / adjacent-season cells), not uniform noise —
see `data/confusion_audio.png`.

Phase-2 (linear probe on the frozen embeddings, PLAN §6) is the natural
lever for the season gap, but that decision is deferred to the bake-off
verdict below once smoke_3/smoke_4 are scored on the same split.

## Bake-off verdict

**Winner: smoke_3 (`clap_knn_env`).** Single-head k-NN retrieval is the
strongest engine (cell top-1 0.435, season 0.524, diel 0.683) and uniquely
adds continuous hour/month estimates + neighbour evidence. Cell-match
(smoke_2) is dominated; fusion (smoke_4) regresses on diel and does not beat
k-NN on cell top-1 — its only keeper is the head-agreement confidence/OOD
signal (acc 0.336 -> 0.565 on agreement), worth carrying as a cheap optional
flag on top of the k-NN engine. Season (~0.52 best) is a frozen-CLAP ceiling
for all three; the PLAN §6 linear probe is the recommended first mvp lever.
Full analysis: [smoke_4 README](../lucas__smoke_4__clap_cell_plus_knn/README.md#bake-off-verdict).
