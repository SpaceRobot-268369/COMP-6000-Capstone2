# mvp_1 clap_knn_probe - eval report

n_val = 391 (held-out, source-clip-disjoint from the train index).

| metric | result | baseline / bar |
|---|---|---|
| probe season acc | 0.491 | k-NN 0.522; bar > 0.60 |
| diel acc | 0.680 | smoke_3 0.683 |
| hour MAE (h) | 2.25 | bar < 2.5 |
| month MAE (mo) | 1.96 | bar < 2.0 |
| agreement rate | 0.376 | - |
| season acc \| agree | 0.5782312925170068 | vs disagree 0.4385245901639344 |

Per-season probe accuracy:

| season | n | acc |
|---|---|---|
| spring | 143 | 0.497 |
| summer | 88 | 0.557 |
| autumn | 99 | 0.455 |
| winter | 61 | 0.443 |

**Probe vs k-NN season: -0.031.** Probe does NOT beat k-NN — keep k-NN season vote.
