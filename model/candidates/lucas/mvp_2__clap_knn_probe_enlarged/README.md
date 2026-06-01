# mvp_2__clap_knn_probe_enlarged

Season-probe checkpoint for Layer E
`lucas__mvp_2__clap_knn_probe_enlarged`.

## Summary

Retrained on `resources/site_257_bowra-dry-a/ambient_pool_v2` with a
1-hidden-layer MLP probe.

- Probe season acc: **0.553**
- k-NN season acc: **0.512**
- Delta: **+0.041**
- Diel acc: **0.706**
- Hour MAE: **1.97 h**
- Month MAE: **1.80 mo**

This is an improvement over the k-NN season baseline, but it does not clear
the original `0.60` season bar.

## Files

- `season_probe.pt` - DVC; torch checkpoint.
- `probe_meta.json` - git; training hyperparameters and best validation
  season accuracy.

## Provenance

- Attempt: `acoustic_ai/layers/layer_e/attempts/lucas__mvp_2__clap_knn_probe_enlarged`
- Dataset: `resources/site_257_bowra-dry-a/ambient_pool_v2`
- Backbone: frozen `laion/clap-htsat-unfused` CLAP embeddings.
- Training: seed 42, 500 epochs, MLP hidden size 256.
