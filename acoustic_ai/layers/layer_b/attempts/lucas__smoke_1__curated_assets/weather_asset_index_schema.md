# Weather Asset Index Schema

Layer B and Layer E-B share the same weather asset index so that one curated
asset pool can support both generation-time mixing and analysis-time detector
calibration.

The index is component-based. A clip is not forced into a single weather label;
it can contain rain, wind, thunder, or a mixture of those components.

## Current Location

```text
acoustic_ai/layers/layer_b/attempts/lucas__smoke_1__curated_assets/data/weather/asset_index.csv
```

This location is inherited from the current placeholder Layer B attempt. Future
Murphy-owned Layer B attempts should keep the same field contract even if the
asset index moves under a new attempt folder.

## MVP CSV Fields

| Field | Purpose |
|---|---|
| `asset_id` | Stable unique ID for this asset row. |
| `clip_path` | Repo-relative or attempt-relative audio path. |
| `source_type` | `site`, `sound_library`, or `generated`. |
| `source_site_id` | Site ID for site-derived clips, else empty. |
| `source_recording_id` | Source recording ID or external library ID. |
| `start_s`, `end_s`, `duration_s` | Source window timing when available. |
| `recording_group_id` | Group key for all clips from the same source recording. Usually `source_type:source_recording_id`. |
| `near_duplicate_group_id` | Group key for adjacent or highly overlapping clips from the same recording. |
| `selection_rank_in_group` | Rank within a duplicate group after scoring; keep low ranks first. |
| `sample_rate`, `channels` | Audio format after curation when known. |
| `primary_weather` | Display/retrieval summary: `rain`, `wind`, `thunder`, `rain_wind`, `storm`, `ambience`, `no_weather`, `unknown`. |
| `has_rain`, `has_wind`, `has_thunder` | Component booleans. Use `unclear` when the component is plausible but not confidently audible. |
| `rain_intensity`, `wind_intensity`, `thunder_intensity` | Component intensity: `none`, `light`, `medium`, `heavy`, or `unclear`. |
| `mixed_weather` | Whether multiple weather components are present or plausibly present. |
| `layer_d_role` | Mixer role, e.g. `rain_primary`, `wind_primary`, `thunder_primary`, `rain_wind_mixed`, `storm_bed`, `storm_thunder_hit`, `boundary_only`, `do_not_use`. |
| `layer_d_use` | `primary`, `backup`, or `reject`. |
| `analysis_use` | E-B use: `train`, `calibration`, `boundary`, `negative`, or `exclude`. |
| `analysis_label_quality` | `high`, `medium`, `low`, or `ambiguous`. |
| `human_audit_status` | `yes`, `maybe`, `no`, `library_seed`, `unreviewed`, etc. |
| `human_weather_label` | Human label if reviewed. |
| `human_notes` | Short review note. |
| `clap_*` fields | CLAP scores and top label when available. |
| `contamination_*` fields | Contamination classifier result when available. |
| `env_*` fields | Weather prior metadata for site-derived clips. |
| `audio_*` fields | Loudness and clipping metrics. |
| `quality_flags` | Pipe-separated flags such as `clipping`, `bio_contamination`, `low_rms`, `library_seed`. |
| `run_id` | Retrieval or curation run that produced the row. |
| `provenance_json` | Compact JSON-like provenance pointer; avoid commas unless quoted. |

## Layer D Policy

Layer D should treat weather as components:

```text
rain: site-first, sound-library fallback
wind: site-first, sound-library fallback
thunder: sound-library first, site only as boundary/optional evidence
storm: compose rain + wind + thunder; do not require one site clip to contain all components
```

Site audit results so far support this policy:

- site wind is usable;
- site rain is usable but often mixed with wind or biological texture;
- site thunder is not reliable enough for MVP primary use.

Apply a recording diversity gate before promotion:

```text
primary assets: max 1-2 clips per recording
backup assets: max 3 clips per recording
near-duplicate primary clips: keep top 1 per group
near-duplicate backup clips: keep top 1-2 per group
```

For site audio, a near-duplicate group can be defined as:

```text
same source_recording_id and start_s within 60-120 seconds
```

This prevents one rainy recording from producing many adjacent clips that look
like independent assets but sound continuous.

## Layer E-B Policy

E-B weather analysis should use the same asset index but interpret rows
differently:

| Asset class | E-B use |
|---|---|
| Clean rain/wind/thunder library assets | `train` or `calibration` with high label quality. |
| Site `rain_primary` / `wind_primary` rows with human yes | `calibration` or `train`, depending on cleanliness. |
| `rain_wind_mixed` and storm-like rows | `boundary`; useful for mixed-weather detector behaviour. |
| CLAP false positives or no-weather rows | `negative` if clean enough, otherwise `exclude`. |
| Clipped or heavily bio-contaminated rows | `exclude`. |

The key design rule is that Layer B labels should not collapse mixed weather
into a single class. Keep component flags and component intensities even when
`primary_weather` is a single display label.

Report validation at two levels:

```text
clip-level precision = accepted clips / reviewed clips
recording-level precision = accepted recording groups / reviewed recording groups
```

Clip-level precision can be inflated when many clips come from one continuous
rain or wind recording. Recording-level precision is the better signal for
whether the policy generalises across site audio.
