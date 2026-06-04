# layer_e_c_species_event_detector (production)

## Summary

- Layer / role: Layer E-C known-species event analysis
- Status: **production**
- Backbone: `laion/clap-htsat-unfused` (frozen)
- Architecture: MLP probe over frozen CLAP audio embeddings
- Source candidate: `model/candidates/songke/mvp_1__layer_e_species_event_detector`
- Source attempt: `songke__smoke_2__known_species_clap_probe`
- Promoted at: 2026-06-04
- Promoted by: Songke
- Intended production attempt: `songke__prod_1__e_c_species_event_detector`

## What This Is

The promoted model artifact for the Layer E-C known-species event detector.
It classifies overlapping five-second audio windows, then the E-C handler
merges repeated detections into event time ranges and enriches them with
species phenology metadata.

The checkpoint supports 13 known Australian species. It is intended for the
online Analysis Mode demo and for future E-A/E-B/E-C report aggregation.

## Why Promoted

The source candidate completes the required demo path:

- frontend upload and E-C analysis
- event onset/offset ranges
- event confidence and species match percentages
- phenology metadata
- aggregator-ready `analysis_report`

Local evaluation produced:

- test accuracy: `0.817`
- test macro-F1: `0.811`

The model was promoted after local frontend and API testing. Promotion also
places the DVC checkpoint under `model/production/`, which allows the Server B
production sync workflow to materialise it automatically.

## Artifacts

- Checkpoint binary: `best_probe.pt` (DVC-tracked)
- DVC pointer: `best_probe.pt.dvc`
- Frozen production parameters: `params.yaml`
- Evaluation metrics: `metrics.json`

## Results Analysis / Audit

**Sign-off:** accepted for the project demo by the team on 2026-06-03.

The model is suitable for demonstrating known-species event detection. It is
not an open-world animal classifier.

### Known Limitations

- Only the configured 13 species can be identified.
- There is no explicit unknown/background rejection model.
- Unseen species or non-animal sounds may be assigned to the nearest known label.
- Test accuracy is approximately 82%, with weaker recall for
  `ninox_boobook` and `rainbow_bee_eater`.
- Confidence values are classifier scores, not ecological certainty.
- E-C does not yet fuse disagreements with E-A or E-B.

## Follow-Up Actions

- Create and validate the production E-C attempt and registry entry.
- Verify `available: true` through local and deployed `GET /layers`.
- Verify the deployed webpage with a known-species recording.
- Add unknown/background rejection in a later model version.
