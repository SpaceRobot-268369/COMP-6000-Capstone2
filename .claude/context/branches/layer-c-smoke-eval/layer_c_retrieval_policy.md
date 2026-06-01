# Layer C Retrieval Baseline Policy

## Goal

Build a stable Layer C retrieval baseline that uses real audited bird-call
snippets instead of LoRA-generated audio.

Pipeline:

```text
annotation events
-> candidate event manifest
-> downloaded/prepared snippets
-> retrieval event index
-> species/time/season selector
-> timeline scheduler
-> Layer C event WAV + timeline metadata
```

## First Scope

Only two bird vocal event types are in scope:

1. Horsfield's Bronze-cuckoo (`Chrysococcyx basalis`)
2. Splendid Fairywren (`Malurus splendens`)

Build and validate Horsfield's Bronze-cuckoo first. Add Splendid Fairywren only
after the cuckoo retrieval loop produces a working 60 second debug bundle.

Out of scope for this retrieval baseline:

- LoRA or Stable Audio training
- synthetic generated snippets as source material
- frontend/server endpoint work
- multi-species scheduling in one timeline
- full mixer integration

## Source Rules

Only real dataset-derived snippets may enter the retrieval pool.

Allowed sources:

- BirdNET / annotation events from site 257 annotation CSVs
- existing manually audited Pass snippets from prior Layer C work
- prepared `.wav` snippets derived from real `.webm` event segments

Disallowed sources:

- AudioGen generated samples
- Stable Audio generated samples
- manually edited/fake motif snippets
- any clip with unclear provenance

## Candidate Selection Policy

Common rules for both species:

- bird vocal events only
- BirdNET score >= 0.9 for newly selected candidates
- raw annotation event duration between 1.0 and 10.0 seconds
- extract with +/-3 seconds buffer
- spread candidates across recordings before reusing a recording
- avoid repeated adjacent snippets from the same recording in scheduled output
- exclude obvious multi-species overlaps, strong wind/rain, vehicles, human
  voices, clipping, and target calls that are too weak or too background-like

Species-specific preference:

| Event type | Common name | Scientific name | Preferred diel bins |
|---|---|---|---|
| `horsfields_bronze_cuckoo` | Horsfield's Bronze-cuckoo | `Chrysococcyx basalis` | dawn, morning |
| `splendid_fairywren` | Splendid Fairywren | `Malurus splendens` | dawn, morning, afternoon, dusk |

The existing smoke manifest script does not have a cuckoo preset, so cuckoo is
passed as a custom event type:

```bash
--event-type "Horsfield's Bronze-cuckoo|Chrysococcyx basalis|horsfields_bronze_cuckoo"
```

## Existing Cuckoo Assets

Prefer these real audited assets before downloading new clips:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/manual_audit_horsfields_bronze_cuckoo_pass24_trainset.csv
```

This file contains 24 real Horsfield's Bronze-cuckoo snippets with human Pass
labels. Use their cropped audio paths for retrieval.

The original six-snippet core remains useful as a stricter reference set:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/stable_audio_core_reference_bank_pass6.csv
```

Fresh 80-candidate expansion manifest and status tables:

```text
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/cuckoo/manifest.csv
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/cuckoo/retrieval_pool_v2/candidate_manifest_80_status.csv
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/cuckoo/retrieval_pool_v2/download_needed_horsfields_bronze_cuckoo_top11.csv
```

## Existing Fairywren Assets

Prefer these existing audited sources for the second species:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/natural_core_v1/retrieval_pool_v1/manual_audit_splendid_fairywren_retrieval_top30.csv
```

Start with the explicit `Pass` rows from the retrieval top-30 audit. If the
scheduled timeline needs more variety, audit the expansion sheet:

```text
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/fairywren/retrieval_pool_v2/manual_audit_splendid_fairywren_retrieval_expansion_top50.csv
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/fairywren/retrieval_pool_v2/manual_audit_splendid_fairywren_retrieval_expansion_top50_absolute.m3u
```

Fresh 80-candidate manifest and status table:

```text
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/fairywren/manifest.csv
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/fairywren/retrieval_pool_v2/candidate_manifest_80_status.csv
```

## Retrieval Index Contract

The event index should be written to:

```text
acoustic_ai/data/events/retrieval/layer_c_event_index.csv
```

Required columns:

```text
snippet_id,event_type,species_common_name,species_scientific_name,audio_event_id,
audio_path,score,diel_bin,season,duration_s,recording_id,event_start_seconds,
event_end_seconds,source_manifest,verdict
```

The CSV index is git-trackable metadata. Large audio snippets should remain in
their existing resource locations or be DVC-tracked if copied into a durable
retrieval snippet directory.

## Selector Policy

Selector input:

```json
{
  "species": "Horsfield's Bronze-cuckoo",
  "diel_bin": "morning",
  "season": "summer",
  "target_duration_s": 60,
  "seed": 42
}
```

Selection rules:

- hard filter by species/common event type
- prefer exact `diel_bin`, with nearby fallback when needed
- soft score by season match, BirdNET score, quality score if available, and
  duration fit
- avoid selecting the same recording consecutively
- use seed-controlled randomization for reproducible variation

## Scheduler Policy

Scheduler output:

- target duration: default 60 seconds
- no overlapping event snippets
- natural event spacing, initially 8-20 seconds
- event gain should avoid clipping and leave headroom for Module D
- final Layer C event layer sample rate: 22050 Hz

Debug bundle path:

```text
debug/layer_c/retrieval/<species_slug>_seed<seed>/
```

Required files:

```text
layer_c_events.wav
layer_c_timeline.json
layer_c_timeline.png
```

Each scheduled event in JSON must record:

- snippet id
- audio path
- species/event type
- source recording
- BirdNET score
- diel and season match notes
- onset and offset seconds
- gain
- selection reason

## First Completion Standard

The cuckoo retrieval baseline is complete when:

- the index can load real Horsfield's Bronze-cuckoo snippets
- same seed produces the same selected snippets and timeline
- `layer_c_events.wav` is 60 seconds long
- events do not overlap or clip
- timeline JSON explains every selected event
- human listening confirms no obvious wrong-species or unusable snippets
