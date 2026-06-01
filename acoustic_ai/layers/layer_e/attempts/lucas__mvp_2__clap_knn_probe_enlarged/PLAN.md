# Implementation Plan - E-A Ambient Analysis mvp_2

| | |
|---|---|
| **Attempt ID** | `lucas__mvp_2__clap_knn_probe_enlarged` |
| **Layer / role** | `layer_e` - Analysis ambient context |
| **Stage** | `mvp_2` |
| **Backbone** | Frozen LAION-CLAP (`laion/clap-htsat-unfused`) |
| **Dataset** | `resources/site_257_bowra-dry-a/ambient_pool_v2` |
| **Candidate** | `model/candidates/lucas/mvp_2__clap_knn_probe_enlarged` |

## Purpose

Retry the Layer E ambient-analysis MVP on the enlarged Bowra ambient pool
instead of the original 1,982-segment Layer A pool. The method stays the same
as `mvp_1`: frozen CLAP embeddings, source-clip-disjoint k-NN for diel/hour/
month/similar-clips, a trained season probe, and the cell-agreement OOD gate.

## Run

From this attempt root:

```bash
../../../../../acoustic_ai/.venv/bin/python code/embed_segments.py --force
../../../../../acoustic_ai/.venv/bin/python code/build_split.py
../../../../../acoustic_ai/.venv/bin/python code/train_probe.py
../../../../../acoustic_ai/.venv/bin/python code/eval.py
```

## Bar

- Season probe accuracy should beat the k-NN season baseline and clear `0.60`.
- Diel accuracy should stay at or above the prior smoke_3 value (`0.683`).
- Hour MAE should stay below `2.5 h`.
- Month MAE should stay below `2.0 mo`.

If the probe still fails to beat k-NN, keep the result as a negative candidate
and serve the simpler k-NN vote rather than the trained head.
