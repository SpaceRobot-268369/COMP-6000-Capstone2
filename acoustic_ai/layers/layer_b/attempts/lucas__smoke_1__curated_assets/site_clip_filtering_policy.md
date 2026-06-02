# Site Clip Filtering Policy for Layer D Candidates

## Purpose

This policy defines the first offline filtering step for using site recordings as
Layer D source material.

The raw site archive is too large and uneven to use directly at runtime. The
system should first build a curated candidate pool from site audio, then run
site-first retrieval against that curated pool.

```text
raw site audio
-> valid S3 coarse clips
-> fine analysis windows
-> env-joined windows
-> weather candidate windows
-> Layer D eligible candidates
```

Runtime retrieval should not scan all raw S3 audio.

## Decision Principle

CLAP/audio embeddings are the primary weather classifier.

Env metadata is only a prior, filter, or tie-breaker. It can propose likely
weather time ranges, but it must not decide that a clip is rain, wind, storm, or
thunder by itself.

The first manual audit showed that env-only selection produces many wrong
weather candidates, especially for light rain, light wind, and storm/thunder.
Use [site_weather_audit_v0.md](site_weather_audit_v0.md) as the calibration
record.

Recommended retrieval/index weighting:

```text
final_score =
  0.65 * CLAP_weather_similarity
+ 0.20 * audio_quality_or_contamination_score
+ 0.15 * env_prior_score
```

## Project Audio Boundaries

Layer B candidates are not considered Layer D-ready just because they have an
MP3 preview or a CLAP analysis WAV.

The CLAP retrieval script may export 48 kHz mono WAV windows for embedding, and
MP3 files for human listening. Those are analysis/review artifacts. Before a
clip can be used by the Layer D mixer, it must be promoted to a Layer D asset:

| Requirement | Value |
|---|---|
| Sample rate | 22,050 Hz |
| Channels | mono |
| Recommended format | WAV for mixer input |
| Required metadata | source S3 URI, offsets, duration, CLAP scores, env prior, pool category, RMS/peak/clipping fields |

The candidate pool manifest must therefore record whether a row is only a
preview/analysis window or whether it has been exported as a Layer D-ready
asset. Do not treat `preview_path` as a runtime asset path.

RMS/peak/clipping metrics are quality controls, not weather labels. RMS can be
used later by the mixer for normalization; clipping or obvious overload should
reject or quarantine a candidate even if the CLAP weather score is high.

Use `script/dataset/promote_site_weather_candidates.py` to promote candidate
rows into Layer D-ready assets. The default promotion mode is conservative:

- export only `pool_decision=accept`;
- do not export backup rows unless explicitly requested;
- do not export site-derived thunder for the MVP;
- write 22,050 Hz mono WAV files plus `layer_d_ready_manifest.csv`.

## CLAP-First Retrieval MVP

Use `script/dataset/build_site_weather_clap_retrieval.py` for the first
server-side CLAP retrieval batch.

The script keeps env metadata in the pipeline, but only for narrowing the search
space:

```text
site item metadata + S3 listing + env rows
-> env prior buckets: rain_env_prior / wind_env_prior / storm_env_prior
-> sampled 15s site windows
-> CLAP weather prompts + contamination prompts
-> retrieval_manifest.csv
```

`clap_weather_label` is the primary machine label. `env_prior_for_clap_label`
is auxiliary evidence, not a label override.

Weather-vs-contamination margins should be treated conservatively. If a
candidate's strongest weather prompt is close to the strongest insect/bird,
human, or machine prompt, route it to `maybe_contamination_close` for listening
review instead of accepting it directly.

Thunder is gated more strictly than rain and wind because the current env data
does not contain direct thunder/lightning annotations. Storm-prior windows must
still pass audio confirmation before they can be accepted. A thunder CLAP label
without at least weak storm/rain env support should be rejected or routed out of
the Layer D weather candidate pool.

## Balanced Retrieval

After the first CLAP audit, global ranking should not be used as the only
manifest selection method because wind tends to dominate the top results.

For manual review batches, prefer target-first balanced retrieval:

```text
rain candidates -> rain prompt score vs contamination/other weather
wind candidates -> wind prompt score vs contamination/other weather
thunder candidates -> thunder prompt score with strict storm-prior gate
```

Recommended MVP review quotas:

```text
rain=30,wind=25,thunder=10
```

Use `retrieval_target` as the audit grouping field. `clap_weather_label` should
still be recorded, but it should not erase the requested target. For example, a
target `rain` clip with top CLAP label `wind` may still be worth hearing if the
rain target score is high and the target-vs-other margin is close.

## Rain Dense Scout Strategy

Server runs on 2026-05-31 showed that broad balanced expansion still produces a
wind-heavy pool. The useful rain yield improved when rain was sampled as a
dense scout:

```text
target-quotas rain=50,wind=0,thunder=0
max-recordings-per-target=30
windows-per-recording=8
no MP3 preview export during scout
```

Observed yield:

| Run | Rain rows | `rain_primary` | Yield |
|---|---:|---:|---:|
| broad rain expansion | 192 | 6 | ~3% |
| rain dense scout | 50 | 10 | 20% |

Use this pattern for future rain expansion:

- sample many windows per rain-prior recording;
- score first, then keep only the best rain rows;
- export listening previews only for the retained sample, not for every scout
  window;
- do not spend more expansion effort on wind until rain is closer to the MVP
  target.

Follow-up rain scout 005 on 2026-06-01 tested whether unseen/manual-unreviewed
rain-prior recordings could improve the site rain pool. The practical exclude
list omitted recordings already manually reviewed or promoted into the MVP
rain-ish pool. The run produced 12 scored windows, but CLAP still preferred wind
for 10/12 rows and no row reached the `candidate` gate. A top-3 listening check
found biological foreground too dominant in every sample.

MVP decision after scout 005:

- stop expanding site-derived rain for this MVP;
- keep the small accepted site `rain_primary` / `rain_wind_mixed` subset as
  site provenance signal;
- use the curated sound-library fallback for clean rain, thunder, and storm
  material;
- do not spend further manual review time on site rain unless new metadata or a
  stronger rain detector is added.

## Candidate Pool Policy

After `audit_002`, do not continue full manual review loops. Use manual audit
results to build an automatic candidate pool, then only spot-check stratified
samples.

Run:

```text
script/dataset/build_site_weather_candidate_pool.py
```

The candidate pool should separate default retrieval material from backup
material:

| Pool category | Default use |
|---|---|
| `rain_primary` | Use by default for Layer D rain retrieval |
| `rain_wind_mixed` | Use when mixed rain/wind is acceptable |
| `wind_primary` | Use by default for Layer D wind retrieval |
| `rain_backup_maybe` | Backup only |
| `wind_with_bio_backup` | Backup only; likely biological contamination |
| `wind_backup_maybe` | Backup only |
| `wind_weak_backup` | Backup only |
| `storm_rain_wind_backup` | Backup only; not thunder |
| `reject` | Do not use |

Thunder is not accepted from site audio in the MVP candidate pool. Route Layer D
thunder needs to the library fallback unless a future thunder detector provides
strong event confirmation.

For thunder-like clips, run
`script/dataset/validate_site_weather_thunder.py` as a second-pass validator.
This validator reads the analysis WAV and computes envelope/spectrogram
features such as low-frequency energy ratio, transient peak-to-median level,
active duration, decay shape, spectral flux, and clipping ratio.

The validator is deliberately conservative:

- it can mark `possible_thunder_burst` for tiny targeted review;
- it should not auto-promote a clip into the default thunder pool;
- it treats peak/clipping risk as `ambiguous_overload_or_thunder`, because wind
  overload can sound like thunder;
- it requires thunder/storm context from CLAP target/label, storm pool, or storm
  env before calling a low-frequency transient a possible thunder burst.

For the MVP, Layer D thunder retrieval should still prefer the sound library
unless a very small reviewed site thunder subset is explicitly promoted later.

Do not use source separation as a prerequisite for Layer B weather retrieval.
Natural site soundscapes have diffuse overlapping sources, and the project
architecture expects direct detection/scoring on the mixture.

Manual review should now be limited to small samples, for example 10-20 clips
per pool category, rather than reviewing every candidate window.

## Scope

This policy is for initial data screening only.

It does not define:

- final CLAP retrieval ranking;
- sound library fallback;
- Layer D timeline mixing;
- S3 download orchestration.

Those steps come after this candidate pool exists.

## Execution Location

This filtering job should run on the server, not on a local laptop.

Server-side execution is required because:

- the raw site audio pool is hundreds of GiB;
- S3 credentials and AWS tooling are available on the server;
- CLAP/audio-quality passes will need stable CPU/GPU resources;
- the curated output should be written back to durable server/S3 storage.

Local development should only edit policy, code, and small metadata fixtures.
It should not download or scan the full raw site archive.

## Observed S3 Data Shape

Observed from Server A (`spacerobot-268369`) on 2026-05-28.

Relevant source prefix:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/downloaded_clips/
```

The raw source is already split into coarse web clips:

```text
site_257_item_<item_id>/site_257_item_<item_id>_clip_<nnn>.webm
```

Current server listing summary:

| Measure | Value |
|---|---:|
| S3 webm clips listed | 243,250 |
| Unique present item folders | 11,229 |
| Total listed size | 471.46 GiB |
| Items with gaps | 1,590 |
| Fully missing items | 1,022 |
| Missing clip rows | 21,459 |
| Files under 1 MB | 860 |
| Files between 1 MB and 3 MB | 242,388 |
| Files at least 3 MB | 2 |

Most S3 clips are roughly 300-second webm chunks. The final chunk for a
recording can be shorter and much smaller.

Existing curated Layer B weather assets on S3 are small:

| Category | Objects |
|---|---:|
| `rain` | 3 |
| `wind` | 3 |
| `thunder` | 2 |

That curated weather asset pool is useful as fallback, but it is too small to be
the primary source for site-first retrieval.

## Server-Side Output Target

The first server job should create a metadata-only candidate index before moving
audio bytes.

Recommended output prefix:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/training_dataset/layer-d/site-weather-candidates-v0/
```

Recommended artifacts:

```text
manifest.csv
policy_version.txt
summary.json
reject_reasons.csv
```

Audio segment export should be a later step. The first pass should keep S3
references and offsets instead of duplicating audio.

## Layer 0: S3 Coarse Clip Policy

Layer 0 validates S3 coarse clips and recording-level metadata before fine audio
windowing.

### Input

Expected fields:

| Field | Meaning |
|---|---|
| `site_id` | Source site identifier |
| `recording_id` | Recording or item id from metadata |
| `item_id` | Item id encoded in the S3 coarse clip path |
| `recorded_date` | Recording start timestamp |
| `duration_seconds` | Full recording duration |
| `sample_rate_hertz` | Source sample rate |
| `channels` | Channel count |
| `media_type` | Source format |
| `s3_key` | Location of source webm or original audio |
| `clip_num` | Coarse S3 clip number |
| `start_offset` | Offset of the coarse clip within the recording |
| `end_offset` | End offset of the coarse clip within the recording |

### Keep

Keep a coarse clip if:

- metadata has `site_id`;
- metadata has a recording timestamp;
- coarse clip duration is longer than the target fine window duration;
- format is expected to be decodable;
- recording status is usable, such as `ready`;
- the S3 key is present.

### Reject

Reject or quarantine a coarse clip if:

- the file is missing or inaccessible;
- it cannot be decoded;
- `site_id` is missing;
- timestamp is missing;
- coarse clip duration is shorter than the target fine window;
- status indicates deleted, failed, or unavailable.

Very small webm files should not be rejected purely by size. Some are valid
short final chunks. Mark them as `small_s3_object` and let decode/duration checks
decide.

### Output

```json
{
  "site_id": 257,
  "recording_id": "5300",
  "item_id": "5300",
  "s3_key": "dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_5300/site_257_item_5300_clip_001.webm",
  "clip_num": "001",
  "start_offset_seconds": 0.0,
  "end_offset_seconds": 300.0,
  "recording_start_utc": "2019-08-13T20:00:00.000Z",
  "coarse_clip_duration_seconds": 300.0,
  "recording_duration_seconds": 7194.749,
  "sample_rate_hertz": 22050,
  "channels": 1,
  "status": "coarse_clip_valid"
}
```

## Layer 1: Fine Window Policy

Layer 1 turns valid S3 coarse clips into fixed fine analysis windows.

These windows are analysis and retrieval units, not necessarily final output
segments.

### Defaults

MVP defaults:

```text
fine_window_duration_seconds = 15
fine_hop_seconds = 10
minimum_window_seconds = 10
```

For the observed Site 257 S3 layout, a normal 300-second webm chunk produces
approximately 29 fine windows at 15s window / 10s hop.

### Keep

Keep a fine window if:

- it decodes successfully;
- duration is at least `minimum_window_seconds`;
- it is not mostly silent;
- it is not heavily clipped;
- it has enough broadband energy to be useful as source material.

### Initial Thresholds

```text
silence_ratio <= 0.70
clipping_ratio <= 0.05
rms_loudness >= configured minimum
```

The RMS threshold should be calibrated after inspecting the first server batch.
Until then, it should be treated as a soft warning rather than a hard reject.

### Reject Reasons

| Reason | Meaning |
|---|---|
| `window_too_short` | Window shorter than minimum duration |
| `decode_failed` | Audio decoder failed on the window |
| `mostly_silent` | Silence ratio is too high |
| `too_much_clipping` | Clipping ratio is too high |
| `too_low_energy` | RMS is below the configured floor |

### Output

```json
{
  "clip_id": "site257_5300_000120_000135",
  "site_id": 257,
  "recording_id": "5300",
  "item_id": "5300",
  "s3_key": "dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_5300/site_257_item_5300_clip_001.webm",
  "coarse_clip_num": "001",
  "recording_start_offset_seconds": 120.0,
  "recording_end_offset_seconds": 135.0,
  "start_time_utc": "2019-08-13T20:02:00.000Z",
  "end_time_utc": "2019-08-13T20:02:15.000Z",
  "duration_seconds": 15.0,
  "silence_ratio": 0.18,
  "clipping_ratio": 0.0,
  "rms_loudness": -31.2,
  "status": "window_valid"
}
```

## Layer 2: Env-Joined Clip Policy

Layer 2 joins each fine audio window to environmental metadata for the same site.

Env data is used as a prior. It does not prove that the target weather is audible
in the clip.

### Join Keys

Preferred join:

```text
site_id + nearest timestamp
```

For the current Site 257 resources, env rows are recording-level and include
`recording_id`. Use the recording row for all fine windows from that recording
and mark the join mode as `recording_level_env`.

### Time Tolerance

MVP tolerance when timestamp join is needed:

```text
env_time_tolerance_minutes = 10
```

### Keep

Keep a clip as env-joined if:

- the env row belongs to the same site;
- the env timestamp is within tolerance, or the metadata is explicitly
  recording-level;
- the fields needed by weather inference are present.

### Missing Env

Do not automatically delete clips with missing env metadata.

Instead:

```text
env_join_status = "missing"
env_prior_score = 0
```

Weather retrieval should rank these lower, but a later CLAP pass may still
recover useful ambience.

For weather-focused Layer D candidates, clips from `env_missing` rows should not
enter the first weather candidate pool unless CLAP later gives a strong weather
score.

### Output

```json
{
  "clip_id": "site257_5300_000120_000135",
  "env_join_status": "matched",
  "env_join_mode": "recording_level_env",
  "env_recorded_date_utc": "2019-08-13T20:00:00.000Z",
  "temperature_c": 2.43,
  "humidity_pct": 81.56,
  "wind_speed_ms": 1.59,
  "precipitation_mm": 0.0,
  "surface_pressure_kpa": 101.4
}
```

## Layer 3: Weather Candidate Policy

Layer 3 converts env metadata into weather priors.

A clip may have multiple weather candidates. For example, rain and wind can both
be true.

Site 257 MVP env distribution is wind-heavy and rain-sparse:

| Env signal | Count in current 287-row env sample |
|---|---:|
| `precipitation_mm > 0` | 42 |
| `wind_speed_ms >= 2` | 209 |
| `wind_speed_ms >= 6` | 9 |
| `wind_speed_ms >= 10` | 0 |

Because the sample has no `wind_speed_ms >= 10`, strong wind should not be
required for MVP site candidate mining. Treat `medium wind` as the strongest
available site-derived wind category until broader data is indexed.

### Rain Rules

```text
precipitation_mm <= 0 -> no rain prior
0 < precipitation_mm < 2 -> light rain
2 <= precipitation_mm < 5 -> medium rain
precipitation_mm >= 5 -> heavy rain
```

Initial prior scores:

| Condition | `rain` prior |
|---|---:|
| `precipitation_mm <= 0` | 0.00 |
| `0 < precipitation_mm < 2` | 0.55 |
| `2 <= precipitation_mm < 5` | 0.75 |
| `precipitation_mm >= 5` | 0.90 |

### Wind Rules

```text
wind_speed_ms < 2 -> no wind prior
2 <= wind_speed_ms < 6 -> light wind
6 <= wind_speed_ms < 10 -> medium wind
wind_speed_ms >= 10 -> strong wind
```

Initial prior scores:

| Condition | `wind` prior |
|---|---:|
| `wind_speed_ms < 2` | 0.00 |
| `2 <= wind_speed_ms < 6` | 0.50 |
| `6 <= wind_speed_ms < 10` | 0.75 |
| `wind_speed_ms >= 10` | 0.90 |

### Thunder and Storm Rules

Thunder should be treated as:

```text
storm prior + CLAP confirmation
```

Use direct thunder metadata when available:

- `thunder_flag`;
- `storm_flag`;
- weather code containing `storm` or `thunder`;
- lightning annotation.

If no direct field exists, do not infer thunder from precipitation alone.

Initial prior scores:

| Condition | `thunder` prior |
|---|---:|
| direct thunder flag | 0.90 |
| storm code or storm flag | 0.75 |
| precipitation only | 0.10 |
| no storm evidence | 0.00 |

### Output

```json
{
  "clip_id": "site257_5300_000120_000135",
  "weather_types": ["wind"],
  "weather_intensity": {
    "wind": "none",
    "rain": "none",
    "thunder": "none"
  },
  "env_prior_score": {
    "wind": 0.0,
    "rain": 0.0,
    "thunder": 0.0
  },
  "weather_candidate_status": "weak_weather_prior"
}
```

## Layer 4: Layer D Eligible Clip Policy

Layer 4 decides whether a clip enters the curated Layer D candidate pool.

This pool is the data source for future site-first retrieval.

### Keep

MVP keep rule before CLAP is available:

```text
audio_quality_score >= 0.60
silence_ratio <= 0.70
clipping_ratio <= 0.05
max(env_prior_score) >= 0.50
```

After CLAP weather scoring is available:

```text
audio_quality_score >= 0.60
silence_ratio <= 0.70
clipping_ratio <= 0.05
and at least one of:
  max(env_prior_score) >= 0.50
  max(clap_weather_score) >= 0.50
  useful_ambience == true
```

For the first server-side weather candidate pass, keep the output metadata-only:

```text
do not export fine audio windows yet
store s3_key + coarse clip offset + fine window offset
```

This prevents unnecessary duplication of a 471 GiB source pool.

### Reject Reasons

| Reason | Meaning |
|---|---|
| `low_audio_quality` | Quality score below threshold |
| `mostly_silent` | Silence ratio too high |
| `too_much_clipping` | Clipping ratio too high |
| `no_weather_or_ambience_value` | No env, CLAP, or ambience signal |
| `human_or_machine_dominant` | Later classifier marks non-ecological dominance |

### Output

```json
{
  "clip_id": "site257_5300_000120_000135",
  "site_id": 257,
  "recording_id": "5300",
  "item_id": "5300",
  "s3_key": "dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_5300/site_257_item_5300_clip_001.webm",
  "coarse_clip_num": "001",
  "recording_start_offset_seconds": 120.0,
  "duration_seconds": 15.0,
  "weather_types": ["rain", "wind"],
  "weather_intensity": {
    "rain": "light",
    "wind": "medium"
  },
  "env_prior_score": {
    "rain": 0.55,
    "wind": 0.75,
    "thunder": 0.0
  },
  "audio_quality_score": 0.72,
  "layer_d_eligible": true,
  "filtering_policy_version": "site_clip_filtering_v0.2"
}
```

## Layer Responsibilities

| Layer | Responsibility | Runtime? |
|---|---|---|
| Layer 0 | Validate S3 coarse clip and recording metadata | Server offline |
| Layer 1 | Split valid coarse clips into fine windows | Server offline |
| Layer 2 | Join windows to env metadata | Server offline |
| Layer 3 | Infer weather priors from env metadata | Server offline |
| Layer 4 | Keep only useful Layer D candidate clips | Server offline |

The runtime retrieval service should query only Layer 4 clips, then rerank them
with CLAP similarity, env prior, and audio quality.

## Initial Implementation Notes

The first implementation should be a server-side indexing job.

It should:

1. read the S3 coarse clip listing;
2. join each coarse clip to recording metadata;
3. decode only the coarse clips selected for inspection;
4. produce fine-window metadata and quality metrics;
5. join env rows at recording level for Site 257;
6. write a metadata-only Layer D candidate manifest.

The job should not download the full archive to local developer machines, and it
should not duplicate S3 audio during the first pass.

## MVP Initial Screening Pass

The first runnable version should be intentionally small and auditable.

Goal:

```text
produce enough site-derived weather candidates to test the policy by listening,
without committing to a full 471 GiB archive scan
```

Recommended MVP scale:

| Bucket | Target candidates before listening |
|---|---:|
| `light_rain` | 20 |
| `medium_or_heavy_rain` | 20 |
| `light_wind` | 20 |
| `medium_wind` | 20 |
| `storm_or_thunder_prior` | 10 |
| `quiet_ambience_control` | 10 |

Total first audit batch: about 100 clips.

For each selected candidate, export a short preview clip for human listening:

```text
preview_duration_seconds = 15
preview_format = wav or mp3
```

Recommended audit output prefix:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/training_dataset/layer-d/site-weather-candidates-v0/audit-previews/
```

The MVP pass should also write:

```text
audit_manifest.csv
```

Minimum audit manifest fields:

| Field | Meaning |
|---|---|
| `clip_id` | Stable fine-window id |
| `preview_path` | Local or S3 path to the exported preview |
| `s3_key` | Source coarse webm key |
| `recording_start_offset_seconds` | Offset within original recording |
| `duration_seconds` | Preview/fine-window duration |
| `candidate_bucket` | Intended bucket such as `light_rain` |
| `env_prior_score` | Env prior used for selection |
| `audio_quality_score` | Initial quality score |
| `human_weather_label` | Filled during listening |
| `human_intensity_label` | Filled during listening |
| `human_accept` | `yes`, `no`, or `maybe` |
| `human_reject_reason` | Short reason if rejected |
| `notes` | Free text listening notes |

## Human Listening Audit Policy

Human listening is required before using the MVP candidate pool for Layer D.

Each candidate should be judged by whether it is useful for mixing, not whether
it is a perfectly isolated weather sound.

### Accept

Accept a clip if:

- the intended weather is clearly audible or plausibly useful as texture;
- the clip is not dominated by speech, traffic, music, or obvious machine noise;
- the clip is stable enough for Layer D mixing;
- clipping, silence, or sudden spikes would not break a generated soundscape.

### Maybe

Mark a clip as `maybe` if:

- weather is present but weak;
- another ecological sound is prominent but not fatal;
- the clip may be useful as quiet ambience rather than weather;
- the intensity label seems wrong but the audio is still usable.

### Reject

Reject a clip if:

- the intended weather is not audible;
- the env prior is contradicted by the audio;
- speech, vehicles, handling noise, or mechanical noise dominates;
- the clip is mostly silent;
- the clip clips, distorts, or has disruptive spikes;
- thunder/storm candidates contain no thunder-like event.

Recommended reject reason values:

```text
wrong_weather
weather_too_weak
mostly_silent
clipping_or_distortion
speech_or_human_noise
vehicle_or_machine_noise
bird_or_insect_dominant
unstable_texture
no_thunder_event
bad_loop_candidate
```

## Policy Update Trigger

Update this policy after the first listening audit if any of these happen:

| Audit result | Policy response |
|---|---|
| Many rain candidates sound dry | Raise rain env threshold or require CLAP rain confirmation |
| Many wind candidates are just leaves/insects | Add CLAP wind confirmation or spectral wind heuristic |
| Thunder candidates lack thunder events | Do not admit thunder without CLAP/event confirmation |
| Many clips are too quiet | Raise RMS floor or lower priority of quiet windows |
| Many clips have spikes/clipping | Tighten clipping and transient-spike checks |
| Good clips are rejected by env prior | Add `useful_ambience` escape path with CLAP/audio evidence |

The policy should not be treated as fixed until the audit batch has been heard.
