# Skill: Sample MVP Dataset — Site 257 Bowra-Dry-A

## Purpose

Reduce the full 6-year audio archive (~12,251 files) down to a representative MVP
training subset. The goal is maximum temporal diversity at minimum download cost,
so the AI pipeline can be validated quickly before scaling to the full dataset.

---

## Sampling Policy: Complete-Day Diel Sampling, Stratified by Year-Month

### Rationale

Bioacoustic activity is highly structured by:
- **Season / month** — breeding periods, migration, vegetation state
- **Year** — long-term ecological change, climate variation
- **Time of day (diel cycle)** — dawn chorus, midday baseline, nocturnal species

Rather than sampling time bins independently (which could mix days), we pick
**whole days** so that all four bins come from the same acoustic conditions
(same weather, same site state). This preserves within-day coherence while
still covering every part of the diel cycle.

---

### Strata

| Dimension | Groups | Detail |
|-----------|--------|--------|
| Year | 6 | 2019 – 2025 (partial years included) |
| Month | 12 | Jan – Dec |
| **Year-Month total** | **72** | Each is an independent stratum |
| Days per stratum | 2 | One from week 1–2, one from week 3–4 |
| Bins per day | 4 | Dawn, Morning, Afternoon, Night |
| **Max recordings** | **~576** | (gaps/missing months reduce this slightly) |

Days within each half-month are chosen with a fixed random seed (`seed=42`)
for full reproducibility.

---

### Time-of-Day Bins

All recordings are stored in UTC. Site timezone is **Australia/Brisbane (UTC+10,
no DST)**. Bins are defined in local AEST and mapped back to UTC for filtering.

| Bin | Local AEST | UTC (same calendar day) | Why |
|-----|-----------|--------------------------|-----|
| **Dawn** | 05:00–07:00 | 19:00–21:00 (prev UTC day) | Dawn chorus — peak species richness |
| **Morning** | 08:00–10:00 | 22:00–00:00 | Post-dawn general activity |
| **Afternoon** | 13:00–15:00 | 03:00–05:00 | Midday / heat-of-day acoustic baseline |
| **Night** | 22:00–00:00 | 12:00–14:00 | Nocturnal and crepuscular species |

Fallback: if the exact bin window has no recording for a given day, expand the
search by ±1 hour. If still empty, that bin is skipped (not filled with a
recording from a different day).

---

### Storage Estimates

| Format | Per file | ~576 files |
|--------|----------|------------|
| Original FLAC (as downloaded) | ~150 MB | **~84 GB** |
| Optimized / resampled clips | ~49 MB | **~28 GB** |

The optimized size is the working target for MVP training. Original files should
be retained as the source of truth.

---

## Implementation

The sampling logic lives in `script/sample_mvp_dataset.py`. It:

1. Reads `resources/site_257_bowra-dry-a/site_257_all_items.csv`
2. Parses `recorded_date` (UTC ISO-8601) and converts to AEST
3. Assigns each row to `(year, month, local_date, time_bin)`
4. For each year-month stratum, splits available days into first/second half
5. Randomly picks 1 day per half (seed=42), falls back to any available day if
   the half is empty
6. Selects 1 recording per (day, bin), applying the ±1 hr fallback
7. Writes `resources/site_257_bowra-dry-a/site_257_mvp_sample.csv` — same
   schema as the source CSV, directly usable by `download_site_257_originals.py`

### Running the sampler

```bash
# From project root
python3 script/sample_mvp_dataset.py

# Preview without writing output
python3 script/sample_mvp_dataset.py --dry-run

# Custom seed or output path
python3 script/sample_mvp_dataset.py --seed 7 --output resources/site_257_bowra-dry-a/site_257_mvp_sample.csv
```

### Then download the sampled files

```bash
# Download using the sampled CSV (all rows, 10 parallel workers)
python3 script/download_site_257_originals.py \
  --csv-path resources/site_257_bowra-dry-a/site_257_mvp_sample.csv \
  --start-item 1 \
  --end-item 9999 \
  --workers 6
```

---

## Output Schema

`site_257_mvp_sample.csv` has the same columns as `site_257_all_items.csv`, with
two additional columns appended:

| Column | Value |
|--------|-------|
| `sample_bin` | `dawn` / `morning` / `afternoon` / `night` |
| `sample_local_date` | Local AEST date string (`YYYY-MM-DD`) |

---

## Scaling Beyond MVP

| Phase | Days / stratum | Est. recordings | Optimized size |
|-------|---------------|-----------------|----------------|
| MVP (current) | 2 | ~576 | ~28 GB |
| Stage 3 pilot | 5 | ~1,440 | ~71 GB |
| Full dataset | all | ~12,251 | ~601 GB |
