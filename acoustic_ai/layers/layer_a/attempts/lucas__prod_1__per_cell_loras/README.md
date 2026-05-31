# lucas__prod_1__per_cell_loras

## Summary

- Owner: Lucas
- Layer / role: Layer A — ambient bed
- Status: **production** (registry default for layer_a)
- Base model: cvssp/audioldm2 (frozen)
- Promoted from: `lucas__mvp_2__per_cell_loras`
- Checkpoint: `model/production/layer_a_ambient/` (16-cell bank)
- Promoted at: 2026-05-31

## What this is

The production serving attempt for the Layer A per-cell ambient bank. Code is
the frozen copy of `lucas__mvp_2__per_cell_loras` — same PEFT multi-adapter
router `handler.py` — pointed at the promoted production checkpoint slot
instead of the candidate slot.

The router loads the AudioLDM2 base once, registers all 16 (season, diel) LoRA
adapters by name, and at request time switches with `set_adapter((season,
diel))` using that cell's locked prompt. Dev-generation contract (seed +
(season, diel) selector) is documented in CLAUDE.md.

## Dataset / inputs

Inherited from the source candidate — `resources/site_257_bowra-dry-a/mvp2_per_cell_dataset`
(cap 200, wind ≤5.5, caption v2). No retraining at promotion; the binaries are
the candidate's, promoted by pointer copy.

## Promotion context

- Promotion type: `mvp_2` → `prod_1` (conventions §5.4).
- Registry: this attempt is registered in `acoustic_ai/registry.yaml` and is
  the layer_a `default:`. Checkpoint → `model/production/layer_a_ambient`.
- Sign-off + caveats: see the production card
  `model/production/layer_a_ambient/README.md` (audit section).

## Results / metrics

See `model/production/layer_a_ambient/metrics.json` and the source attempt's
DEVLOG for per-cell losses and the listening-audit verdict.

## Known limitations

Promoted with documented caveats: no long-form generation; two `watch` cells
(`summer_afternoon`, `autumn_dawn`) and three thin morning cells are Phase 2.5
targets; browser end-to-end recommended before public demo. Full list in the
production card.

## Follow-up actions

- Phase 2.5 augmentation for the weak/thin cells, then re-promote.
- Long-form tiled bed generator.
- Browser verification in the running stack.
