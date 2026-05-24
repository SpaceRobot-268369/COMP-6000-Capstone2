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

Initial dev-container smoke test passed with `--segments-per-type 1` on
2026-05-24 for the previous wind/rain/thunder cases. The comparable
breeze/storm/light-raining/heavy-raining cases exposed that CLAP-only ranking
does not reliably separate weather intensity.

Current selector behavior combines:

- CLAP semantic file retrieval
- `asset_index.csv` intensity labels for light/medium/strong filtering
- segment-level audio validation for silence, clipping, and texture stability

Re-run manual listening after each selector change and record pass/fail notes in
the local `outputs/manual_audit.md`.

## Artifact Policy

Generated WAV clips and local `report.json` files are ignored by git. If these
artifacts need to be shared across machines, track them with DVC and commit only
the resulting `.dvc` pointer files plus small metadata.

## Limitations

This smoke attempt validates top-1 segment quality only. Returning multiple
segments still needs a diversity constraint so adjacent windows from the same
source file are not selected as separate useful candidates.
