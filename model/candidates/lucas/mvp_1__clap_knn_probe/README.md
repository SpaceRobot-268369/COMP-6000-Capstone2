# mvp_1__clap_knn_probe — season probe (NEGATIVE result, parked)

Trained season head for Layer E `lucas__mvp_1__clap_knn_probe`. A tiny probe
on frozen CLAP embeddings → 4-way season.

## Summary

- **Does not beat the k-NN season baseline (0.522) on v0 single-site data.**
  - linear: val season acc **0.445** (underfits; not linearly separable)
  - mlp (1×256): val season acc **0.491** (overfits; train 0.755 / val 0.49)
- Checkpoint saved is the **mlp** variant (more balanced per-season).
- Kept as a documented negative result + reproducibility baseline for the
  enlarged-dataset retry (attempt PLAN §4 v1). **Not served**; serving uses
  the k-NN season vote.

## Files
- `season_probe.pt` — DVC; torch checkpoint (state_dict + arch/season_order).
- `probe_meta.json` — git; arch, best val acc, hyperparams, seed.

## Provenance
- Attempt: `acoustic_ai/layers/layer_e/attempts/lucas__mvp_1__clap_knn_probe`
- Trained on serverB (T4), seed 42, frozen `laion/clap-htsat-unfused` embeddings.
- params: see attempt `params.yaml` (training:).

## Audit
_Empty — pending evaluation notes / review._
