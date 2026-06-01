# smoke_2 cell_match - example predictions

5 held-out val segments (never seen in anchor construction), audio-prototype variant.
season/diel/cell columns: Y = matches ground truth, N = miss.

| segment | true cell | predicted cell | conf | season | diel | cell | top-3 |
|---|---|---|---|:--:|:--:|:--:|---|
| `1536463_clip009_s001` | autumn_afternoon | winter_afternoon | 0.13 | N | Y | N | winter_afternoon (0.13), summer_dawn (0.11), spring_afternoon (0.11) |
| `215190_clip007_s000` | summer_night | autumn_night | 0.17 | N | Y | N | autumn_night (0.17), spring_night (0.16), winter_dawn (0.15) |
| `215467_clip002_s000` | summer_dawn | spring_morning | 0.11 | N | N | N | spring_morning (0.11), spring_dawn (0.10), summer_afternoon (0.10) |
| `216086_clip015_s000` | autumn_dawn | spring_dawn | 0.10 | N | Y | N | spring_dawn (0.10), spring_morning (0.09), summer_afternoon (0.09) |
| `216470_clip010_s000` | autumn_afternoon | autumn_morning | 0.10 | Y | N | N | autumn_morning (0.10), summer_morning (0.10), spring_afternoon (0.09) |

**Tally:** season 1/5, diel 3/5, exact cell 0/5.
