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
| wind | strong natural forest wind ambience |
| rain | light drizzle under forest canopy |
| thunder | distant rolling thunderstorm ambience |

## Manual Audit

Dev-container smoke test passed with `--segments-per-type 1` on 2026-05-24.

Manual listening notes:

- `wind_0_windy_forest_176.0s.wav`: pass; matches wind description, no severe clipping.
- `rain_0_rainforest_rain_32.0s.wav`: pass; matches rain description, no severe clipping.
- `thunder_0_rain_thunder_40.0s.wav`: pass; matches thunder/rain-thunder description, no severe clipping.

## Artifact Policy

Generated WAV clips and local `report.json` files are ignored by git. If these
artifacts need to be shared across machines, track them with DVC and commit only
the resulting `.dvc` pointer files plus small metadata.

## Limitations

This smoke attempt validates top-1 segment quality only. Returning multiple
segments still needs a diversity constraint so adjacent windows from the same
source file are not selected as separate useful candidates.
