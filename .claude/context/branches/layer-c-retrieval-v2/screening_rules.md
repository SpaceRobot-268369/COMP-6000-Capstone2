# Layer C Retrieval v2 Screening Rules

Branch context for rebuilding the Layer C retrieval sample library from
`resources/site_257_bowra-dry-a/bird_audio_annotation_inventory_other_tags.csv`.

## Scope

- Use only the first 63 species rows in
  `bird_audio_annotation_inventory_other_tags.csv` after the header.
- All first-63 species are eligible, including the eight species previously
  used by Layer C retrieval work.
- Previously human-approved samples may be reused, but they must be repackaged
  under the v2 directory layout and rechecked against the v2 crop and bandpass
  requirements.
- If local samples are not enough to fill the target count for a species, fetch
  additional source recordings or annotation audio from S3 and continue
  selection.

## Species Targets

Target sample count is derived from `local_annotation_audio_count`.

| local_annotation_audio_count | target per species |
|---:|---:|
| > 100 | 100 |
| 50-100 inclusive | 50 |
| < 50 | 10 |

The generated quota table lives at:
`acoustic_ai/layers/layer_c/attempts/burger__mvp_2__retrieval_v2_library/data/media_asset_bank/species_quota_v2.csv`.

## Old Sample Reuse

For species that already have Layer C final-pass or human-pass samples:

1. Treat those samples as preferred candidates.
2. Re-run v2 validation rather than accepting them blindly.
3. Recreate crops, spectrograms, metadata, and review files using the v2
   layout.
4. Count only samples that still pass v2 review toward the species target.
5. Fill any shortfall from local annotations first, then S3-backed source data.

## Candidate Ranking

Within each species, rank candidates by:

1. Existing human `Pass` verdict from prior Layer C review.
2. Higher sample-level `quality_score`.
3. Higher annotation/model confidence score.
4. Clearer complete target call boundaries.
5. Less non-target species overlap.
6. Less wind, rain, human voice, vehicle, machinery, and broadband clutter.
7. Cleaner target-band spectrogram structure.
8. Diversity across recording IDs and dates.

If fields are missing, preserve the sample for manual review but mark the
missing field in `review.json`.

## Hard Rejections

Reject a candidate when any of the following is true:

- Source audio cannot be resolved locally or from S3.
- The species name or `audio_event_id` is missing.
- The target event time range is invalid or outside the source recording.
- The target call cannot be confidently heard or seen.
- The crop cannot include the complete target call without severe overlap.
- Heavy wind, rain, human voice, vehicle, machinery, or clipping dominates the
  target call.
- A high-confidence non-target bird overlaps the target band and time window.
- Duration after buffered crop is still unusably short or too broad for a
  single retrieval sample.

## Time Crop Rule

Each selected sample must contain the full target vocalisation plus audible
context buffer.

Default crop:

```text
crop_start_s = max(0, event_start_s - 0.25)
crop_end_s   = min(recording_duration_s, event_end_s + 0.35)
```

Manual adjustment is allowed and expected when annotation boundaries are too
tight. The final crop must not cut off call onset, tail, trills, harmonics, or
repeated syllables that are part of the same event.

Recommended target crop duration:

```text
0.5s <= crop_duration_s <= 8.0s
```

Samples outside this range are allowed only with a review note explaining why
the event remains useful.

## Frequency Crop Rule

Each sample must provide:

- A full-band time crop for audit.
- A target-band crop for retrieval/mixing.

The target-band crop should use the narrowest practical band that preserves the
species vocalisation:

- Low cutoff should sit just below the lowest target-call energy.
- High cutoff should sit just above the highest target-call energy.
- Keep meaningful harmonics when they help identify the species.
- Exclude avoidable wind, insects, non-target birds, and broadband clutter.

Use a species-level default band when possible, then override per sample when
the spectrogram shows a different call type or pitch range.

Record both defaults and overrides in metadata:

```text
species_low_hz
species_high_hz
sample_low_hz
sample_high_hz
bandpass_reason
```

## Directory Layout

Each species gets one folder. Each sample gets one subfolder.

```text
acoustic_ai/layers/layer_c/attempts/burger__mvp_2__retrieval_v2_library/data/media_asset_bank/
  <species_slug>/
    samples/
      <NNN>_audioevent_<audio_event_id>/
        original.wav
        crop_full.wav
        crop_bandpass.wav
        mel_full.png
        mel_bandpass.png
        metadata.json
        review.json
```

Required file meanings:

- `original.wav`: source or local source excerpt used for traceability.
- `crop_full.wav`: full-band buffered event crop.
- `crop_bandpass.wav`: target-band crop used by retrieval.
- `mel_full.png`: spectrogram/mel image for the full crop.
- `mel_bandpass.png`: spectrogram/mel image for the target-band crop.
- `metadata.json`: source IDs, paths, timing, frequency band, scores, and
  provenance.
- `review.json`: pass/borderline/reject verdict and reviewer notes.

## Metadata Minimum Fields

Each `metadata.json` must include:

```text
species_common_name
species_scientific_name
species_slug
audio_event_id
recording_id
source_audio_path
source_is_s3_backed
source_manifest
event_start_s
event_end_s
crop_start_s
crop_end_s
pre_buffer_s
post_buffer_s
species_low_hz
species_high_hz
sample_low_hz
sample_high_hz
score
quality_score
diel_bin
season
reused_from_prior_library
prior_library_path
```

## Review Outputs

Build these tables as the pipeline matures:

```text
species_quota_v2.csv
candidate_samples_v2.csv
selected_samples_v2.csv
rejected_samples_v2.csv
layer_c_retrieval_v2_event_index.csv
```

The final event index should contain only `Pass` samples and point at
`crop_bandpass.wav` for retrieval playback/mixing.
