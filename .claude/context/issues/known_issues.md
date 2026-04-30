# Known Issues

## Unrecoverable Clips — A2O API 422 Errors

**Date discovered:** 2026-04-23
**Status:** Permanent — server-side, cannot be fixed by re-downloading

The following 12 clips return `422 Unprocessable Entity` from the A2O API regardless
of retries. The audio data for these specific time offsets is corrupted or missing on
the A2O server. All other 6,148 clips downloaded successfully.

| CSV Count | Recording ID | Failed Clip | Start (s) | End (s) |
|-----------|-------------|-------------|-----------|---------|
| 216 | 1678484 | 021 | 6000.000 | 6300.000 |
| 219 | 1678513 | 006 | 1500.000 | 1685.955 |
| 222 | 1681319 | 009 | 2400.000 | 2681.998 |
| 248 | 1679394 | 024 | 6900.000 | 7194.749 |
| 249 | 1676521 | 005 | 1200.000 | 1500.000 |
| 252 | 1676441 | 021 | 6000.000 | 6300.000 |
| 254 | 1676444 | 018 | 5100.000 | 5400.000 |
| 256 | 1670355 | 024 | 6900.000 | 7194.749 |
| 266 | 1672094 | 011 | 3000.000 | 3300.000 |
| 268 | 1681455 | 001 | 0.000 | 300.000 |
| 270 | 1676142 | 023 | 6600.000 | 6900.000 |
| 281 | 1672466 | 011 | 3000.000 | 3226.645 |

**Impact:** 12 / 6,160 clips missing (0.2%) — negligible for training.
Each affected recording still has the majority of its clips present.

**Action:** Exclude these clips from the training pipeline. Do not attempt to
re-download — they will always fail with 422.
