# layer_a_ambient (production)

## Summary

- Layer / role: Layer A — ambient bed
- Status: **production**
- Base model: cvssp/audioldm2 (frozen)
- Architecture: per-cell LoRA bank (16 season×diel adapters)
- Source candidate: `model/candidates/lucas/mvp_2__per_cell_loras`
  (attempt `lucas__mvp_2__per_cell_loras`)
- Promoted at: 2026-05-31
- Promoted by: Lucas
- Served via attempt: `lucas__prod_1__per_cell_loras` (registry default for layer_a)

## What this is

The promoted Layer A ambient-bed model: 16 independent LoRA adapters, one per
(season, diel_bin) cell, merged onto the frozen AudioLDM2 base at inference
time by the per-cell router (`set_adapter((season, diel))`). Layout mirrors
the source candidate:

```
layer_a_ambient/
├── <season>_<diel>/adapter_model.safetensors      (×16, DVC-tracked)
├── <season>_<diel>/adapter_config.json            (×16, git)
├── params.yaml      (frozen recipe + per-cell tiers)
├── metrics.json     (per-cell losses + known issues)
└── README.md        (this card)
```

The `.dvc` pointers reference the same cache blobs as the source candidate
(promotion is a pointer copy — no binary duplication in the remote).

## Why promoted

`mvp_1` (shared r=8) and `mvp_1_2` (shared r=64) both blurred across the 16
cells; `mvp_1_1` proved a single cell trains to smoke quality at r=8. The
per-cell bank removes the shared-weight gradient interference and recovers
per-cell quality. Full evidence chain:
`acoustic_ai/layers/layer_a/attempts/lucas__mvp_2__per_cell_loras/DEVLOG.md`.

## Results / metrics

See [metrics.json](./metrics.json). Most cells land at val loss ~0.057–0.10
(reference `mvp_1_1` = 0.0565).

## Results analysis / audit

**Sign-off:** Developer listening audit **passed** (Lucas, 2026-05-31).
Per-cell quality holds across the bank; cells are audibly distinct by season
and time-of-day. Promoted **with documented caveats** (below).

### Known limitations (promoted with caveats)

- **No long-form generation** — adapters produce ~10 s coherent beds only.
  A tiled + crossfade long-form generator is a planned follow-up.
- **Two `watch` cells** — `summer_afternoon` (val 0.147, 53 clips) and
  `autumn_dawn` (val 0.144, 53 clips) show a mild overfit signature and are
  the primary Phase 2.5 augmentation targets.
- **Three thin cells** — `winter_morning` (22), `summer_morning` (30),
  `autumn_morning` (34 clips) are source-scarce; quality is acceptable but
  least robust. Augmentation / second-site data is the long-term fix.
- **Browser end-to-end** verified at the router/dispatch level on GPU; full
  click-through in the running stack is recommended before public demo.

## Promotion context

- Recipe (per cell): r=8, α=32, 5 epochs, batch 4, lr 1e-5, fp16, RMS-norm
  off. Identical to the proven `mvp_1_1` recipe. See [params.yaml](./params.yaml).
- Dataset: `resources/site_257_bowra-dry-a/mvp2_per_cell_dataset` (cap 200,
  wind ≤5.5, caption v2).
- Hardware: AWS EC2 Tesla T4 (`shinypokemon`).

## Follow-up actions

- Phase 2.5: augment + retrain the 2 `watch` cells and 3 thin cells.
- Long-form tiled bed generator.
- Full browser verification in the running stack before any public demo.
