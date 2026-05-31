# mvp_2__per_cell_loras (per-cell LoRA bank)

## Summary

- Owner: Lucas
- Layer / role: Layer A — ambient bed
- Status: candidate (passed listening audit; **not promoted**)
- Base model: cvssp/audioldm2 (frozen)
- Trained at: 2026-05-30 (shinypokemon, Tesla T4)
- Attempt: `acoustic_ai/layers/layer_a/attempts/lucas__mvp_2__per_cell_loras/`

## What this is

A **bank of 16 LoRA adapters**, one per (season, diel_bin) cell, each trained
with the proven `mvp_1_1` recipe (r=8, α=32, 5 epochs). Layout:

```
mvp_2__per_cell_loras/
├── <season>_<diel>/adapter_model.safetensors   (×16, DVC-tracked)
├── <season>_<diel>/adapter_config.json         (×16, git)
├── params.yaml                                  (frozen recipe + per-cell tiers)
├── metrics.json                                 (per-cell losses + known issues)
└── README.md                                    (this file)
```

Inference selects the adapter matching the requested (season, diel) and merges
it onto the frozen base — see the Phase 3 router.

## Why per-cell (decision context)

`mvp_1` (shared r=8) and `mvp_1_2` (shared r=64, 8 epochs) both blurred when
one LoRA had to cover all 16 cells. `mvp_1_1` proved a single cell trains to
smoke quality at r=8. Conclusion: the bottleneck was **gradient interference
between cells**, not capacity or data. Per-cell LoRAs remove it by
construction. Full evidence chain in the attempt DEVLOG.

## Results / metrics

See [metrics.json](./metrics.json). Most cells land at val loss ~0.057–0.10
(reference: `mvp_1_1` = 0.0565). Two cells flagged `watch` (low train / high
val): `summer_afternoon` (0.147) and `autumn_dawn` (0.144) — Phase 2.5
augmentation candidates. Listening audit: **passed overall.**

## Results analysis / audit

_Developer listening audit (2026-05-30): passed. Per-cell quality holds across
the bank; cells are audibly distinct by season/time-of-day. Residual minor
quality variance on the two `watch` cells. Long-form generation not yet
available._

## Known limitations

- **Not servable standalone** — requires the Phase 3 inference router to map
  (season, diel) → adapter. Until then this bank cannot generate through the API.
- No long-form generation (10 s tiles only); tiled+crossfade generator planned.
- 3 thin morning cells (autumn 38 / summer 33 / winter 22 clips) and 2 `watch`
  cells are the augmentation backlog for Phase 2.5.
- Per-cell `README.md` files written by the training script were not retained
  here (training-script overwrite bug, tracked separately).

## Follow-up actions

- Phase 3: build the inference router (`handler.py` keyed on season/diel).
- Phase 2.5: augment + retrain `summer_afternoon`, `autumn_dawn`, and the 3
  thin morning cells if they underperform end-to-end.
- Promotion to `model/production/ambient/` only after the router works
  end-to-end and a validation sign-off — **not done.**
