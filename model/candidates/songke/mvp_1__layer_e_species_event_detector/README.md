# layer_e_species_event_detector_v1

## Summary

- Layer: E-C events analysis
- Status: candidate
- Author: songke
- Backbone: `laion/clap-htsat-unfused`
- Head: MLP probe over frozen CLAP audio embeddings
- Checkpoint: `best_probe.pt`
- Training manifest: `local_data/ec_species/manifests/ec_species_13class_no_magpie_manifest.csv`
- Training clips: 2842 local 5 s positive clips

## Known Species

- `ninox_boobook`
- `laughing_kookaburra`
- `rhipidura_leucophrys`
- `psophodes_cristatus`
- `cincloramphus_mathewsi`
- `podargus_strigoides`
- `red_capped_robin`
- `anas_superciliosa`
- `australian_raven`
- `peaceful_dove`
- `galah`
- `crested_bellbird`
- `rainbow_bee_eater`

## Metrics

- Test accuracy: 0.817
- Test macro-F1: 0.811
- Best epoch: 460

See `metrics.json` for per-class precision, recall, F1, and confusion details.

## Intended Use

This checkpoint supports the Layer E events head. It detects whether uploaded audio contains one or more trained known species, then returns event time ranges, confidence values, and per-event species match percentages.

The detector should be treated as a known-species classifier. It does not yet include an explicit unknown/background rejection model, so out-of-scope animals may still be assigned to the nearest known label.

## Artifact Tracking

`best_probe.pt` is a model binary and must be tracked through DVC, not git. Metadata files such as this README, `params.yaml`, `metrics.json`, and the `.dvc` pointer belong in git.
