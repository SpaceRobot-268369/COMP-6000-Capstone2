# Layer C Retrieval v2 Library

Owner: burger
Stage: mvp_2
Method: audited real-snippet retrieval

This attempt exposes the Layer C retrieval MVP through the registry handler.
It does not load an AudioGen checkpoint. Instead, it selects Pass-reviewed
bird-call snippets from the v2 event index, schedules them into a 60-second
Layer C event timeline, and renders a frontend-ready WAV plus metadata.

## Structure

```text
code/
  handler.py
data/
  media_asset_bank/
    layer_c_retrieval_v2_event_index.csv
    species_band_config_final_human_v1.csv
model/candidates/burger/mvp_2__retrieval_v2_library/
  index.json
  media_asset_bank/
    review_package_full_v2_final_human_bandpass.dvc
```

The attempt-local CSV files are build/audit provenance. The runnable retrieval
artifact lives under `model/candidates/burger/mvp_2__retrieval_v2_library/`
as `index.json` plus a DVC-tracked `media_asset_bank/`. The runtime route uses
the final human-audited bandpass package and returns Layer C only output, not
an A-layer mix.

## Data Summary

- Species: 63
- Pass-reviewed snippets: 4,329
- Default demo species: Splendid Fairywren
- Default context: summer morning
- Runtime package: final human-audited bandpass snippets
- Pacific Swift has 9 snippets because the v2 candidate pool only had 9 valid
  candidates under the current screening rules.

## Runtime

The registered attempt is `burger__mvp_2__retrieval_v2_library`.
Its `asset_bank` is declared in `acoustic_ai/registry.yaml` and points to
`model/candidates/burger/mvp_2__retrieval_v2_library`.

To materialize the audited retrieval package for local generation testing:

```bash
dvc pull model/candidates/burger/mvp_2__retrieval_v2_library/media_asset_bank/review_package_full_v2_final_human_bandpass.dvc
```
