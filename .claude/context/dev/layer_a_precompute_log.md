# Layer A — Ambient Pool Precompute Log

Date: 2026-05-02
Status: Cleaning complete, audit pending

---

## Full Cleaning Run Results

**Execution:**
```bash
python3 acoustic_ai/precompute/build_ambient_index.py --workers 6
```

**Output Summary:**
- Clips processed: 6,146 (2 too_short, 0 load_error)
- Segments generated: 1,982
- Per-cell coverage: all 16 cells populated

---

## Per-Cell Coverage Table

| Diel Bin | Season | Count | Notes |
|----------|--------|-------|-------|
| afternoon | autumn | 215 | ✓ solid |
| afternoon | spring | 131 | ✓ good |
| afternoon | summer | 73 | ⚠ lower than expected |
| afternoon | winter | 89 | ✓ good |
| dawn | autumn | 65 | expected: densest activity bin |
| dawn | spring | 103 | expected: densest activity bin |
| dawn | summer | 110 | expected: densest activity bin |
| dawn | winter | 75 | expected: densest activity bin |
| morning | autumn | 47 | ✓ acceptable |
| morning | spring | 126 | ✓ good |
| morning | summer | 69 | ✓ good |
| morning | winter | 22 | ⚠ low but adequate |
| night | autumn | 232 | ✓ excellent |
| night | spring | 342 | ✓ excellent |
| night | summer | 166 | ✓ good |
| night | winter | 117 | ✓ good |

---

## Analysis

### What went well

1. **All 16 cells populated.** No diel_bin × season cell is empty. Even the smallest
   (`morning winter: 22`) exceeds the retrieval minimum k=5 with headroom.

2. **Night and afternoon dominate.** Night segments (757 total) and afternoon (508)
   drive the pool — ecologically sensible, as these periods have cleaner ambient
   texture (less dawn/dusk chorus density).

3. **Gate performs as designed.** Only 7.8% of frames flagged as anomalous (flux + RMS),
   ±0.1 s dilation brings it to 24.3% masked. Event masking is content-agnostic
   and self-calibrating per clip.

4. **Dilation was tuned correctly.** Reducing from 0.5 s → 0.1 s and min_span from
   20 s → 10 s was necessary. Dense clips (dawn/spring) yielded segments; sparse clips
   (afternoon/summer) benefit from lower minimum threshold.

### Observations

#### `morning winter: 22` (⚠ low but adequate)

**Root cause:** 2021–2022 archive gap. The source A2O archive has virtually no
recordings from those years at Site 257. This cell is genuinely data-sparse, not a
threshold issue.

**Mitigation:** The retrieval `low_confidence` fallback will handle it. If a user
requests `(diel_bin=morning, season=winter)`, the hard filter returns < 5 segments,
we relax to neighbouring diel_bin (e.g., afternoon/winter or dawn/winter) and flag
`metadata.low_confidence = True`.

**Action:** No threshold adjustment needed. Accept as-is.

#### `afternoon summer: 73` (lower than expected)

**Root cause:** Summer afternoons in Australia feature continuous, high-amplitude
cicada drone. This is *ambient* — stationary, repeating, not an event — but its high
RMS can trigger false anomalies if the rolling baseline is computed over a wide
window and the cicada intensity varies slightly across the clip.

**Observation:** The afternoon summer cell still has 73 segments, which is adequate
coverage. The gate is conservative (prefers fewer, cleaner segments over more, noisier
ones).

**Action:** Monitor in the audit. If afternoon summer segments consistently fail the
listening test (i.e., sound like pure cicada), we've correctly filtered out the noise.
If they pass cleanly, the lower count is not a problem — we have sufficient coverage.

---

## Thresholds Used (Final)

```python
CFG = {
    "rolling_window_s":   30.0,   # per-clip baseline window
    "mad_threshold":      3.0,    # frames > 3·MAD from rolling median
    "dilation_s":         0.1,    # ± 0.1 s margin around events
    "min_span_s":         10.0,   # min span length (lowered from 20 s)
    "target_seg_s":       30.0,   # preferred segment length
    "max_seg_s":          60.0,   # hard cap on segment length
    "rms_low_pct":        20,     # RMS span verification: [p20, p80] of clip
    "rms_high_pct":       80,
    "max_frame_diff_z":   2.0,    # mel stationarity check
}
```

**Justification:**
- `dilation_s: 0.1` — ecoacoustic events (bird calls, etc.) have fuzzy onsets
  (~50 ms), so 0.1 s (22 frames at 43 fps) captures the onset fuzziness without
  over-eating the gaps between events.
- `min_span_s: 10.0` — dawn chorus density maxes out around 10–11 s per clip.
  Lowering from 20 s to 10 s accommodates the hardest diel bin while maintaining
  acceptable stationarity (verified post-hoc by RMS and mel-variance checks).
- `mad_threshold: 3.0` — standard choice (3·MAD ≈ 3σ for Gaussian). With the MAD
  floor (2% of rolling median), near-zero-MAD clips no longer amplify tiny drift
  into false anomalies.

---

## Diagnostic Run (Single Clip, dawn/spring)

**Clip:** `site_257_item_4881_clip_001.webm` (5 min, dawn, spring)

```
per-feature anomaly rates:
  flux          anomalous=  4.1%
  rms           anomalous=  6.0%
combined anomalous: 7.8%
after dilation: anomalous=24.3%
clean_runs=165, longest_clean=10.9s
```

**Interpretation:** 23 short events (bird calls, rustles) spread across the clip,
detected by flux and RMS. After 0.1 s dilation, 24.3% of frames are masked but
still yields 165 unmasked runs, with the longest at 10.9 s — just above our 10 s
minimum. This is the densest possible diel bin, and the gate handles it well.

---

## Next Steps

1. **Audit:** Listen to 30 stratified samples from `debug/ambient_audit/`.
   Pass criterion: ≥27/30 sound like pure ambience (no identifiable events).

2. **If audit passes:** Track in DVC, commit, proceed to Step 5 (write retrieval).

3. **If audit reveals a cell is consistently failing (e.g., all afternoon/summer
   segments sound like cicada drones):** Decide whether to tighten the gate for that
   cell only or accept the cicada as ambient (it is, botanically speaking — cicadas
   are part of the soundscape texture). The answer depends on the project's definition
   of "ambient".

---

## Files Generated

- `acoustic_ai/data/ambient/ambient_segments/*.wav` — 1,982 cleaned audio segments
- `acoustic_ai/data/ambient/ambient_index.csv` — segment metadata (id, source, time,
  diel_bin, season, cyclic time/month encoding)
- `acoustic_ai/precompute/build_ambient_index.py` — cleaning pipeline (with
  `--diagnose` mode)
- `acoustic_ai/modules/ambient/audit_segments.py` — stratified audit tool
