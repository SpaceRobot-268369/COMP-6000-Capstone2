# smoke_2 cell_match — eval report

n_val = 391 segments (source-clip-disjoint from train)
softmax tau = 0.1

## Smoke bar
- season_acc >= 0.70
- diel_acc >= 0.55
- cell_top3 >= 0.50

## Results

| metric | text anchors | audio prototypes |
|---|---|---|
| cell top-1 | 0.077 | 0.338 |
| cell top-3 | 0.340 | 0.619 |
| season acc | 0.233 | 0.455 |
| diel acc | 0.547 | 0.632 |
| mean confidence | 0.132 | 0.163 |

Per-cell accuracy (audio prototypes):

| cell | n | acc |
|---|---|---|
| spring_dawn | 24 | 0.208 |
| spring_morning | 25 | 0.560 |
| spring_afternoon | 23 | 0.000 |
| spring_night | 71 | 0.690 |
| summer_dawn | 24 | 0.000 |
| summer_morning | 15 | 0.200 |
| summer_afternoon | 14 | 0.071 |
| summer_night | 35 | 0.514 |
| autumn_dawn | 12 | 0.083 |
| autumn_morning | 11 | 0.000 |
| autumn_afternoon | 32 | 0.531 |
| autumn_night | 44 | 0.227 |
| winter_dawn | 17 | 0.176 |
| winter_morning | 5 | 0.400 |
| winter_afternoon | 19 | 0.421 |
| winter_night | 20 | 0.050 |

Confusion matrices: `data/confusion_text.png`, `data/confusion_audio.png`.
