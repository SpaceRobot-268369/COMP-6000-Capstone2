# Layer A — `lucas__mvp_1_1__spring_night_replica`

**Stage:** mvp-1.1 (Phase 1A diagnostic).
**Owner:** Lucas.
**Parent attempt:** `lucas__mvp_1__audioldm2_all_conditioned`.

## Hypothesis

Training a smoke-style single-scene LoRA on the **spring-night subset of the
MVP-1 dataset** (90 train + 10 val clips) reaches `lucas__smoke_1` quality on
the spring-night prompt.

If yes → the MVP-1 data filter pipeline (rules 1–7 + §6.1 content filters) is
not the cause of MVP-1's per-cell quality blur. Proceed to per-cell LoRAs
(MVP-2).

If no → the data pipeline degraded the source audio somewhere (over-strict
rule 5, over-aggressive content filter, segment extraction bug). Halt; fix
data; rerun this attempt before any architecture work.

## Scope

In scope:
- `(season=spring, diel_bin=night)` rows only from the MVP-1 manifest.
- Single LoRA, smoke-style hyperparams (r=8, alpha=32, 5 epochs).
- Caption v2 unchanged — date token INCLUDED. The goal here is to compare
  apples-to-apples against MVP-1's data pipeline, not to confound with a
  caption-schema change. Caption v3 (no date) lands at Phase 1B and MVP-2.

Out of scope:
- Per-cell LoRA architecture (Phase 2).
- Caption schema change (Phase 1B + Phase 2).
- Other (season, diel) cells.

## Comparison target

| reference | when | source |
|---|---|---|
| `lucas__smoke_1__audioldm2_spring_night` showcase | by ear | existing showcase artifacts |
| `lucas__mvp_1` seed 42 baseline (spring night) | by ear | `acoustic_ai/layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/dev-artifacts-self-testing/seed_42_baseline/` |

Outcome lands in this attempt's `DEVLOG.md` retrospective section.
