# mvp_2 clap_knn_probe_enlarged - eval report

n_val = 1239 (held-out, source-clip-disjoint from the train index).

| metric | result | baseline / bar |
|---|---|---|
| probe season acc | 0.553 | k-NN 0.512; bar > 0.60 |
| diel acc | 0.706 | smoke_3 0.683 |
| hour MAE (h) | 1.97 | bar < 2.5 |
| month MAE (mo) | 1.80 | bar < 2.0 |
| agreement rate | 0.368 | - |
| season acc \| agree | 0.6425438596491229 | vs disagree 0.5006385696040868 |

Per-season probe accuracy:

| season | n | acc |
|---|---|---|
| spring | 396 | 0.490 |
| summer | 305 | 0.675 |
| autumn | 262 | 0.435 |
| winter | 276 | 0.620 |

**Probe vs k-NN season: +0.041.** Probe improves season.
