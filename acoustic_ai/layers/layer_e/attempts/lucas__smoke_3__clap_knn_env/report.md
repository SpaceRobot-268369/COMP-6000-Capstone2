# smoke_3 clap_knn_env — eval report

n_query = 391 segments (source-clip-disjoint from index of size 1591)

## Baselines
- season majority `spring` → acc 0.366
- diel majority `night` → acc 0.435
- hour global-mean → MAE 4.93 h
- month global-mean → MAE 2.82 months

## Smoke bar
- season_acc >= 0.70
- diel_acc >= 0.55
- hour MAE < 3.0
- month MAE < 2.0

## Sweep results (k × tau)

| k | tau | season | diel | hour MAE (h) | month MAE (mo) | P@k |
|---|---|---|---|---|---|---|
| 1 | 0.05 | 0.481 | 0.624 | 2.37 | 2.08 | 0.379 |
| 1 | 0.1 | 0.481 | 0.624 | 2.37 | 2.08 | 0.379 |
| 1 | 0.2 | 0.481 | 0.624 | 2.37 | 2.08 | 0.379 |
| 3 | 0.05 | 0.506 | 0.660 | 2.21 | 2.03 | 0.365 |
| 3 | 0.1 | 0.506 | 0.660 | 2.21 | 2.03 | 0.365 |
| 3 | 0.2 | 0.506 | 0.660 | 2.22 | 2.03 | 0.365 |
| 5 | 0.05 | 0.522 | 0.678 | 2.24 | 1.96 | 0.340 |
| 5 | 0.1 | 0.522 | 0.680 | 2.25 | 1.96 | 0.340 |
| 5 | 0.2 | 0.522 | 0.680 | 2.26 | 1.96 | 0.340 |
| 10 | 0.05 | 0.537 | 0.665 | 2.24 | 1.89 | 0.316 |
| 10 | 0.1 | 0.535 | 0.660 | 2.26 | 1.90 | 0.316 |
| 10 | 0.2 | 0.532 | 0.662 | 2.27 | 1.91 | 0.316 |

**Best (by hour MAE, ties broken by season acc):** k=3, tau=0.05
