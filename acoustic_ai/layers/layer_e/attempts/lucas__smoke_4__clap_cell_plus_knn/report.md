# smoke_4 cell+knn fused — eval report

n_val = 391 segments (source-clip-disjoint).
params: head_a_variant=audio, k=5, tau_a=0.1, tau_b=0.1, w_a=0.5, w_b=0.5.

## Hypothesis 1 — fusion ≥ best single head

| metric | head A (cell) | head B (knn) | fused |
|---|---|---|---|
| cell top-1 | 0.338 | 0.435 | 0.422 |
| cell top-3 | 0.619 | 0.688 | 0.706 |
| season acc | 0.455 | 0.524 | 0.529 |
| diel acc | 0.632 | 0.683 | 0.639 |
| mean confidence | 0.163 | 0.551 | 0.348 |

## Hypothesis 2 — head agreement signals correctness

- agreement rate: **0.376**
- accuracy | agree:    0.565 (n=147)
- accuracy | disagree: 0.336 (n=244)
- error-correlation between A and B: 0.279

Confusion matrices: `data/confusion_{a,b,fused}.png`.
