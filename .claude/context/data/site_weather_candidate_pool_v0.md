# Site Weather Candidate Pool v0

Purpose: convert scored CLAP retrieval manifests into a curated candidate pool
for later Layer D site-first retrieval and mixing.

This step replaces full-batch manual listening. Manual review should now be a
small spot check per pool category.

## Builder

```text
script/dataset/build_site_weather_candidate_pool.py
```

Inputs:

- scored retrieval manifest from `build_site_weather_clap_retrieval.py`;
- CLAP weather scores;
- contamination scores;
- env prior fields;
- S3 source URI and offsets.

Outputs:

- `candidate_pool_manifest.csv`
- `summary.json`
- `manual_review_sample.csv`
- `policy_version.txt`

Important manifest fields for Layer D handoff:

| Field | Meaning |
|---|---|
| `pool_decision` | `accept`, `backup`, or `reject` |
| `pool_category` | retrieval bucket used by Layer D |
| `pool_label` | weather label exposed to retrieval |
| `layer_d_export_action` | whether/how to promote this row to mixer-ready audio |
| `layer_d_target_sample_rate_hz` | must be `22050` for mixer-ready export |
| `layer_d_target_channels` | must be `1` |
| `layer_d_recommended_format` | `wav` |
| `analysis_rms_dbfs` | loudness sanity check from CLAP analysis WAV, when available |
| `analysis_peak_dbfs` | peak sanity check from CLAP analysis WAV, when available |
| `analysis_clipping_ratio` | overload rejection/quarantine signal |

## Policy Version

```text
site_weather_candidate_pool_v0.2
```

`v0.2` adds the project audio-boundary requirements from `prerequisites.md`:

- CLAP analysis WAVs and MP3 previews are not Layer D runtime assets.
- Accepted/backup rows must be exported later as 22,050 Hz mono WAV before they
  are used by the mixer.
- Candidate manifests carry RMS, peak, and clipping fields when available.
- Rows with clipping/overload risk are rejected or quarantined even if their
  weather score is high.

## Dry Run on Audit 002

Input:

```text
debug/site_weather_clap_retrieval_v0_audit_002/retrieval_manifest.csv
```

Output:

```text
debug/site_weather_candidate_pool_v0_from_audit_002/
```

Summary:

| Decision | Count |
|---|---:|
| `accept` | 20 |
| `backup` | 24 |
| `reject` | 21 |

Pool categories:

| Category | Count | Use |
|---|---:|---|
| `rain_primary` | 3 | Default rain retrieval |
| `rain_wind_mixed` | 4 | Use when mixed rain/wind is acceptable |
| `rain_backup_maybe` | 5 | Backup only |
| `wind_primary` | 13 | Default wind retrieval |
| `wind_backup_maybe` | 4 | Backup only |
| `wind_with_bio_backup` | 7 | Backup only; biological contamination risk |
| `wind_weak_backup` | 1 | Backup only |
| `storm_rain_wind_backup` | 7 | Backup only; not thunder |
| `reject` | 21 | Do not use |

## Server MVP Pool 001

Run on Server A (`spacerobot-268369`) on 2026-05-30.

Server working directory:

```text
/home/ubuntu/layer_b_site_weather_job/
```

Retrieval output:

```text
runs/mvp_pool_20260530_001/
```

Candidate pool output:

```text
runs/mvp_pool_20260530_001_candidate_pool/
```

Local spot-check page:

```text
debug/site_weather_candidate_pool_mvp_001/listen.html
```

Retrieval summary:

| Metric | Count |
|---|---:|
| Total retrieval windows | 178 |
| Target rain | 80 |
| Target wind | 80 |
| Target thunder | 18 |
| CLAP rain label | 24 |
| CLAP wind label | 147 |
| CLAP thunder label | 7 |

Retrieval gates:

| Gate | Count |
|---|---:|
| `candidate` | 85 |
| `maybe_target_confused_with_other_weather` | 31 |
| `maybe_contamination_close` | 2 |
| `reject_target_outcompeted_by_other_weather` | 43 |
| `reject_contamination_dominant` | 4 |
| `reject_thunder_without_clear_audio_confirmation` | 13 |

Candidate pool summary:

| Decision | Count |
|---|---:|
| `accept` | 63 |
| `backup` | 46 |
| `reject` | 69 |

Pool categories:

| Category | Count | Use |
|---|---:|---|
| `rain_primary` | 4 | Default rain retrieval |
| `rain_wind_mixed` | 6 | Use when mixed rain/wind is acceptable |
| `rain_backup_maybe` | 12 | Backup only |
| `wind_primary` | 53 | Default wind retrieval |
| `wind_backup_maybe` | 9 | Backup only |
| `wind_with_bio_backup` | 17 | Backup only; biological contamination risk |
| `storm_rain_wind_backup` | 8 | Backup only; not thunder |
| `reject` | 69 | Do not use |

Interpretation:

- Wind remains the strongest site-derived Layer B pool.
- Rain is usable but limited; use `rain_primary` first and allow
  `rain_wind_mixed` only when mixed weather is acceptable.
- Site-derived thunder is still not reliable; keep Layer D thunder on library
  fallback for the MVP.
- Continue with spot-check review instead of full manual review. The generated
  sample has 67 clips across pool categories.

## Retrieval Implication

Layer D site-first retrieval should query only accepted pools by default:

```text
rain -> rain_primary, optionally rain_wind_mixed
wind -> wind_primary
thunder -> library fallback
```

Backup pools are useful when default pools have insufficient coverage, but they
should have lower priority than sound-library fallback if the requested layer
needs a clean, isolated weather texture.

Before Layer D uses an accepted or backup row, run a promotion/export step that
creates the actual 22.05 kHz mono WAV asset and writes its durable storage URI
back to the candidate pool index. Until that exists, the row is a retrieval
candidate, not a mixer input.

## Layer D Asset Promotion 001

Run on Server A (`spacerobot-268369`) on 2026-05-31.

Script:

```text
script/dataset/promote_site_weather_candidates.py
```

Input:

```text
runs/mvp_pool_20260530_001_candidate_pool/candidate_pool_manifest.csv
```

Output:

```text
runs/mvp_pool_20260530_001_layer_d_assets_accept_only/
debug/site_weather_layer_d_assets_mvp_001/
```

Policy version:

```text
site_weather_candidate_promotion_v0.1
```

Promotion mode:

```text
accept-only, no backup, no thunder
```

Summary:

| Pool category | Promoted WAV assets |
|---|---:|
| `rain_primary` | 4 |
| `rain_wind_mixed` | 6 |
| `wind_primary` | 53 |
| **Total** | **63** |

Verification:

| Requirement | Result |
|---|---|
| Sample rate | 22,050 Hz |
| Channels | mono |
| Format | WAV |
| Manifest rows | 63 |

The generated `layer_d_ready_manifest.csv` is the first MVP manifest that points
to mixer-ready site-weather assets. Backup rows remain candidates only and are
not promoted by default.

## Thunder Validator 001

Run on Server A (`spacerobot-268369`) on 2026-05-30.

Script:

```text
script/dataset/validate_site_weather_thunder.py
```

Input:

```text
runs/mvp_pool_20260530_001_candidate_pool/candidate_pool_manifest.csv
```

Output:

```text
runs/mvp_pool_20260530_001_thunder_validator_v002/
debug/site_weather_thunder_validator_mvp_002/
```

Policy version:

```text
site_weather_thunder_validator_v0.2
```

Summary:

| Validator label | Count | Use |
|---|---:|---|
| `possible_thunder_burst` | 10 | Possible thunder-like transient; hold for tiny targeted review only |
| `ambiguous_storm_rumble` | 14 | Storm/rain/wind rumble, not enough to call thunder |
| `possible_storm_rumble_not_thunder` | 2 | Burst-like mixed weather without thunder context |
| `ambiguous_overload_or_thunder` | 4 | Low-frequency burst with clipping/peak risk; do not auto-accept |
| `likely_wind_or_rain` | 2 | Not thunder |

Interpretation:

- Spectrogram/envelope features help detect low-frequency transient bursts, but
  they are not a standalone thunder classifier.
- Strong wind, microphone overload, and storm rain/wind can share the same
  low-frequency burst shape as thunder.
- Site thunder should remain out of the default MVP pool. Use the 10
  `possible_thunder_burst` rows only as a small research/spot-check set.
- Any `ambiguous_overload_or_thunder` row is unsafe for automatic Layer D
  thunder because overload can sound like thunder and can clip the mixer.

## Manual Review Going Forward

Do not review every candidate. Instead:

1. Run candidate-pool policy on a larger server batch.
2. Auto-accept only strict `rain_primary` and `wind_primary` rows.
3. Keep `rain_wind_mixed` and backup pools available but lower-priority.
4. Use at most a tiny targeted thunder spot check from
   `possible_thunder_burst`; otherwise use library fallback for thunder.
5. Tune thresholds only if a category has a systematic failure.
6. Write the final candidate pool manifest to durable server/S3 storage.
