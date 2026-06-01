# smoke_3 knn_env - example predictions

5 held-out query segments (excluded from the k-NN index), k=5.
season/diel: Y = match. hour/month columns show true/est (abs error).

| segment | true->est season | s | true->est diel | d | hour t/e (err) | month t/e (err) | conf |
|---|---|:--:|---|:--:|---|---|---|
| `1536463_clip009_s001` | autumn->autumn | Y | afternoon->afternoon | Y | 14/14 (0.0) | 5/6 (0.5) | 0.95 |
| `215190_clip007_s000` | summer->winter | N | night->night | Y | 22/23 (1.0) | 1/7 (5.5) | 0.98 |
| `215467_clip002_s000` | summer->spring | N | dawn->dawn | Y | 6/8 (2.0) | 2/1 (0.7) | 0.94 |
| `216086_clip015_s000` | autumn->spring | N | dawn->night | N | 6/4 (2.2) | 4/12 (4.4) | 0.98 |
| `216470_clip010_s000` | autumn->summer | N | afternoon->night | N | 14/19 (5.0) | 5/4 (1.2) | 0.94 |

**Tally:** season 1/5, diel 3/5. (hour/month errors in the columns above.)
