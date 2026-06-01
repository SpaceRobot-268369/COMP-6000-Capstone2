# smoke_4 fused - example predictions

5 held-out val segments, fused (w_a=w_b=0.5, k=5).
cell: Y = fused cell exactly matches ground truth. agree = the two heads picked the same cell.

| segment | true cell | head A (cell) | head B (knn) | fused | cell | heads | conf | ood |
|---|---|---|---|---|:--:|:--:|---|:--:|
| `1536463_clip009_s001` | autumn_afternoon | winter_afternoon | autumn_afternoon | autumn_afternoon | Y | DISAGREE | 0.46 | - |
| `215190_clip007_s000` | summer_night | autumn_night | autumn_night | autumn_night | N | agree | 0.28 | - |
| `215467_clip002_s000` | summer_dawn | spring_morning | summer_dawn | summer_dawn | Y | DISAGREE | 0.23 | OOD |
| `216086_clip015_s000` | autumn_dawn | spring_dawn | spring_night | spring_night | N | DISAGREE | 0.22 | OOD |
| `216470_clip010_s000` | autumn_afternoon | autumn_morning | summer_night | summer_night | N | DISAGREE | 0.31 | - |

**Tally:** exact cell 2/5; heads agreed on 1/5.
When the two independent heads agree, accuracy is far higher (eval: 0.565 vs 0.336) - agreement is the usable confidence/OOD signal.
