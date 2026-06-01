# lucas__mvp_1__clap_knn_probe

First mvp for Layer E ambient analysis. Graduates the smoke bake-off winner
(smoke_3 k-NN retrieval) and adds a trained season probe + agreement/OOD gate.
Full design + decisions: [PLAN.md](PLAN.md).

## Engine (one frozen-CLAP embedding, three heads)

- **k-NN** over a source-clip-disjoint index → diel (vote), hour & month
  (circular blend), `similar_clips` evidence.
- **season probe** — a tiny trained head (linear by default) on the 512-d CLAP
  vector → 4-way season. Replaces *only* the k-NN season vote (season is the
  axis k-NN is weakest on, ~0.52 in the smokes).
- **agreement / OOD gate** — cell-match anchor head vs k-NN cell vote; low
  confidence + disagreement raises `ood_flag` (smoke_4's keeper).

## Data

v0 baseline on the current 1,982 Bowra ambient segments. ONE seed-42
source-clip-disjoint split: **train** → k-NN index + probe training; **val** →
held-out eval. The enlarged dataset (PLAN §4) is a drop-in rebuild — re-run the
three scripts below, no code change.

## Layout

```
code/
├── paths.py            # shared constants (paths, SEASON_ORDER, CELL_ORDER)
├── clap_backbone.py    # 48 kHz / 10 s / mean-pool / L2-norm CLAP wrapper (copied, proven)
├── embed_segments.py   # one-time embedding cache for all 1,982 segments
├── build_split.py      # single split -> index_embeddings + index_meta + anchors_audio
├── train_probe.py      # train season probe on cached embeddings -> model/candidates/...
├── handler.py          # AmbientAnalyzer.analyze() — the E-A entry point
└── eval.py             # probe-vs-kNN season + diel/hour/month + agreement gate
data/
├── embeddings_cache.npy   # DVC
├── index_embeddings.npy   # DVC
├── anchors_audio.npy      # DVC
├── index_meta.csv         # git
└── splits/{train,val}.csv # git
```

## Reproduce

From the attempt root, with `acoustic_ai/.venv` set up and `ambient_segments`
`dvc pull`-ed:

```bash
cd acoustic_ai/layers/layer_e/attempts/lucas__mvp_1__clap_knn_probe
../../../../../acoustic_ai/.venv/bin/python code/embed_segments.py
../../../../../acoustic_ai/.venv/bin/python code/build_split.py
../../../../../acoustic_ai/.venv/bin/python code/train_probe.py
../../../../../acoustic_ai/.venv/bin/python code/eval.py
```

Single-clip analysis:

```bash
../../../../../acoustic_ai/.venv/bin/python code/handler.py /path/to/upload.wav
```

## mvp bar (from PLAN §7)

- season acc **> 0.60** (clear lift over k-NN 0.524; stretch 0.70)
- diel acc ≥ 0.683 (no regression vs smoke_3)
- hour MAE < 2.5 h, month MAE < 2.0 mo
- agreement separates correct vs incorrect (acc gap > 0.15)

## Results

_Empty until eval has been run on serverB. Fill in with probe-vs-kNN season,
the env metrics, the agreement-gate gap, and the bar verdict._

## Serving wiring

_Deferred until the offline bar is met (PLAN §6): registry entry, FastAPI
upload endpoint, Express `/api/analysis`, frontend Analysis mode._
