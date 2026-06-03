# Layer E MVP 3 - Site257 Weather Audit Dataset

## Scope

This attempt expands E-B beyond the current 63 audited weather seed clips.

The target is an E-B-owned, Site257-only, multi-attribute audit dataset for
weather-layer analysis. It should support ordinary Site257 uploads, not only
curated weather-pool clips.

This attempt follows the main-branch Layer E synthesis policy:

- E-B labels weather as an authoritative audio observation.
- E-B does not infer season or diel in production.
- Human audit buckets are used to calibrate continuous weather summaries:
  intensity, label, coverage, variability, and confidence.

This attempt does not replace the existing seed set. The seed set is used as:

```text
audited calibration seed -> candidate mining guide -> expanded Site257 audit manifest
```

## Why

The current E-B baseline can run and produce wind/rain intensity + confidence,
but the data is too small for a final MVP claim. A realistic weather detector
needs:

- many no-weather clips to control false positives,
- enough examples per wind intensity bucket,
- mixed weather + ecology examples,
- random Site257 holdout clips,
- labels that allow multiple attributes per clip.

Each audio clip can carry multiple labels at once. For example:

```text
wind=moderate, rain=none, bird_activity=medium, insect_activity=low,
background_noise=medium, mixed_weather=false
```

## Current Method

The first script mines candidate clips from existing Site257 manifests and the
current audited weather seed policy snapshot:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/layers/layer_e/attempts/liting__mvp_3__site257_weather_audit_dataset/code/mine_site257_weather_candidates.py
```

Output:

```text
debug/e_b_site257_audit_candidates/candidate_manifest.csv
debug/e_b_site257_audit_candidates/summary.json
```

This first pass is metadata-first. It does not download audio and does not train
a model. It prepares the next Server B run by selecting:

- existing audited seed weather clips,
- candidate wind buckets from Site257 manifests,
- likely no-weather candidates,
- mixed/boundary candidates,
- random holdout candidates.

After DVC audio is materialised on Server B, this manifest becomes the listening
and audit queue.

## Audit Output

The final audited manifest should follow:

```text
data/audit_manifest_template.csv
```

The most important fields are:

- `wind_intensity`
- `rain_intensity`
- `thunder_status`
- `mixed_weather`
- `bird_activity`
- `insect_activity`
- `background_noise`
- `audit_status`
- `notes`

## Success Bar

This attempt is useful when it produces a reviewable candidate queue with:

- 100+ candidates for each wind bucket where Site257 supports it,
- a large no-weather candidate set,
- mixed weather examples,
- random holdout clips,
- traceability back to Site257 source recordings and clip paths.

The first run is allowed to be imperfect. The purpose is to make manual audit
targeted rather than listening to every Site257 clip blindly.
