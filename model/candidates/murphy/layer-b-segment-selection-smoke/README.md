# Layer B Segment Selection Smoke

## Purpose

This candidate records Murphy's Layer B segment-selection smoke test attempt.

Shared source code stays in:

```text
acoustic_ai/modules/weather/
```

This folder stores run metadata, smoke-test configuration, manual audit notes,
and optional DVC pointers for generated review artifacts.

## Run

From the repository root:

```bash
python acoustic_ai/tests/layer_b_segment_selection_smoke.py --segments-per-type 1
```

By default, the smoke test writes generated local review artifacts here:

```text
model/candidates/murphy/layer-b-segment-selection-smoke/outputs/
```

## Current Smoke Cases

| Case | Query |
|---|---|
| breeze | light breeze through forest leaves |
| storm | strong storm wind with distant thunder |
| light_raining | light raining under forest canopy |
| heavy_raining | heavy raining in dense forest |

## Manual Audit

Final dev-container smoke test passed with `--segments-per-type 1` on
2026-05-24 for the comparable breeze, storm, light-raining, and heavy-raining
cases.

Manual listening passed for all exported top-1 clips:

| Case | Manual result |
|---|---|
| breeze | Pass - light wind texture |
| storm wind | Pass - strong wind texture |
| storm thunder | Pass - exported thunder clip matches storm thunder |
| light_raining | Pass - light rain texture |
| heavy_raining | Pass - heavy rain texture |

Current selector behavior combines:

- CLAP semantic file retrieval
- `asset_index.csv` intensity labels for light/medium/strong filtering
- segment-level audio validation for silence, clipping, and texture stability
- light-wind segment ranking that prefers lower-energy breeze clips
- multi-type smoke export so storm writes both wind and thunder clips

## Artifact Policy

Generated WAV clips and local `report.json` files are ignored by git. If these
artifacts need to be shared across machines, track them with DVC and commit only
the resulting `.dvc` pointer files plus small metadata.

## Limitations

This smoke attempt validates top-1 segment quality only. Returning multiple
segments still needs a diversity constraint so adjacent windows from the same
source file are not selected as separate useful candidates.
