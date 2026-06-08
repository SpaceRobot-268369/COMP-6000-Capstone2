# mvp_2__retrieval_v2_library

## Summary

- Owner: burger
- Layer / role: Layer C events
- Status: candidate
- Retrieval method: audited real-snippet retrieval over species, season, and diel metadata
- Built at: 2026-06

## Purpose / hypothesis

Provide a reliable Layer C retrieval baseline for the MVP demo. The bank stores human-audited, species-specific bird-call snippets from site_257 Bowra dry woodland and lets the Layer C handler select and schedule real calls into a layer-C-only event stem.

## Dataset / inputs

- Dataset: site_257 Bowra dry woodland bird annotation events.
- Source clips / manifests: `acoustic_ai/layers/layer_c/attempts/burger__mvp_2__retrieval_v2_library/data/media_asset_bank/layer_c_retrieval_v2_event_index.csv`.
- Filtering or preprocessing: Pass-reviewed snippets, final human bandpass bounds, one folder per species, `crop_bandpass.wav` used for runtime retrieval.
- Known data caveats: Pacific Swift has 9 snippets because the v2 candidate pool only had 9 valid candidates under the current screening rules.

## Training or promotion context

- Index build command: derived from the v2 event CSV into `index.json`.
- Code branch / commit: `feat/burger/layer-c-retrieval-v4`.
- Important settings: 63 species, canonical seed 42, 60-second Layer C timeline, layer-C-only output.

## Artifacts

- Index: `index.json`.
- Media asset bank: `media_asset_bank/review_package_full_v2_final_human_bandpass.dvc`.
- Sample outputs:
  - Expected: `acoustic_ai/layers/layer_c/attempts/burger__mvp_2__retrieval_v2_library/expected.dvc`.
  - Showcase: not populated beyond the live deterministic dev route.

## Bank stats

- Assets: 4,329
- Bank size: 3.3 GiB
- Retrieval key(s): species common name, event type, season, diel bin, score, quality score
- Attributes vocabulary: event_type, species_common_name, species_scientific_name, audio_event_id, score, quality_score, diel_bin, season, duration_s, recording_id, event_start_seconds, event_end_seconds, source_manifest, verdict, notes, source
- Audio sources: site_257
- Index: index.json (schema_version 1)

## Results / metrics

Local smoke checks passed for species-selectable Layer C generation with 60-second 22,050 Hz output and frontend expected/generated comparison panels.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

- Retrieval uses real audited snippets, not from-scratch generated bird calls.
- Runtime quality depends on the audited bank coverage for the selected species and context.

## Follow-up actions

- Add a canonical DVC-tracked `showcase/seed_42_baseline/` case if the team wants the attempt to satisfy the full artifact-tier checklist.
