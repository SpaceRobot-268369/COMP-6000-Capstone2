# Environmental Data Features for AI Training

Source script: `script/fetch_nasa_env_data.py`
Output file: `resources/site_257_bowra-dry-a/site_257_env_data.csv`
Site: Bowra Dry-A, coordinates (-30.04, 145.87), timezone UTC+10 (AEST, no DST)

---

## Trial Download — Issues Found & Fix Checklist

| # | Issue | Status |
|---|-------|--------|
| 1 | `cloud_clearness_index` passed NASA -999 sentinel through as a numeric value (98 nighttime rows affected) | ✅ Fixed — `clean()` helper strips sentinel; nighttime rows now empty string |
| 2 | `sunrise_utc` / `sunset_utc` stored as bare UTC `HH:MM`, causing sunrise (≈20:52) to appear *after* sunset (≈07:50) due to UTC midnight wrap | ✅ Fixed — renamed to `sunrise_local` / `sunset_local`, stored in AEST; `sunrise < sunset` now always true |
| 3 | `--ndvi` flag referenced in CLAUDE.md Data Commands but not implemented in script (would crash with `unrecognized arguments`) | ✅ Fixed — argument added; prints a clear `[WARN] not yet implemented` and continues |
| 4 | `days_since_rain` passed AEST `local_date` to `compute_days_since_rain()`, but `daily_lookup` is keyed by UTC date — off-by-one day for dawn/morning recordings | ✅ Fixed — now passes `dt.date()` (UTC) to match daily_lookup key space; `days_since_rain` changed from 37→36 on affected rows |

---

## Data Sources

| Source | Resolution | Purpose |
|--------|-----------|---------|
| NASA POWER API | Hourly | Instantaneous conditions at recording time |
| NASA POWER API | Daily | Day-level aggregates + drought tracking |
| Astral (local compute) | Per local date | Sunrise/sunset/daylight duration |
| Derived (local compute) | Per recording | Temporal context features |

---

## Feature Groups

### Hourly Conditions (NASA POWER — matched to recording start time, UTC key)

| Column | Unit | Description | Notes |
|--------|------|-------------|-------|
| `temperature_c` | °C | Air temperature at 2m | |
| `humidity_pct` | % | Relative humidity at 2m | |
| `wind_speed_ms` | m/s | Wind speed at 2m | |
| `wind_direction_deg` | ° | Wind direction at 2m | |
| `precipitation_mm` | mm/hr | Corrected precipitation rate | |
| `solar_radiation_wm2` | W/m² | All-sky surface shortwave radiation | 0 at night |
| `cloud_clearness_index` | 0–1 | Cloud clearness (1 = fully clear sky) | **Empty for nighttime** (NASA -999 stripped) |
| `surface_pressure_kpa` | kPa | Surface atmospheric pressure | |

### Daily Aggregates (NASA POWER — keyed by UTC date)

| Column | Unit | Description | Notes |
|--------|------|-------------|-------|
| `temp_max_c` | °C | Daily maximum temperature | |
| `temp_min_c` | °C | Daily minimum temperature | |
| `precipitation_daily_mm` | mm/day | Total daily precipitation | |
| `wind_max_ms` | m/s | Daily maximum wind speed | |
| `days_since_rain` | days | Consecutive dry days before recording (90-day lookback, threshold ≥ 1mm) | Uses UTC date for lookup |

### Sun / Light (Astral — AEST local times)

| Column | Format | Description | Notes |
|--------|--------|-------------|-------|
| `sunrise_local` | HH:MM | Sunrise time in AEST | Always < `sunset_local` |
| `sunset_local` | HH:MM | Sunset time in AEST | Always > `sunrise_local` |
| `daylight_hours` | hrs | Total daylight duration | |

### Temporal Context (Derived)

| Column | Description |
|--------|-------------|
| `hour_utc` | Recording start hour (UTC, 0–23) |
| `hour_local` | Recording start hour (AEST, 0–23) — diel cycle position |
| `month` | Local month (1–12) — seasonal cycle |
| `day_of_year` | Local day of year (1–365) — fine-grained seasonal position |
| `season` | Southern hemisphere: `summer` / `autumn` / `winter` / `spring` |

---

## Identity Columns (linking back to recordings)

| Column | Description |
|--------|-------------|
| `count` | Row index |
| `recording_id` | A2O recording ID (joins to `site_257_filtered_items.csv`) |
| `recorded_date_utc` | UTC ISO-8601 timestamp of recording |
| `sample_bin` | Diel bin: `dawn` / `morning` / `afternoon` / `night` |
| `sample_local_date` | Local AEST date (YYYY-MM-DD) |

---

## Total Feature Count

| Group | Features |
|-------|---------|
| Hourly conditions | 8 |
| Daily aggregates | 5 |
| Sun / light | 3 |
| Temporal context | 4 |
| **Total conditioning inputs** | **20** |

---

## Notes for AI Training

- `cloud_clearness_index` is empty for nighttime recordings (dawn + night bins) — treat as a masked/conditional feature, not fill with 0.
- `days_since_rain` uses UTC date throughout to stay consistent with NASA POWER key space. Values reflect drought conditions up to 90 days prior.
- `sunrise_local` / `sunset_local` are AEST `HH:MM` strings — convert to decimal hours for model input.
- Temporal features (`month`, `day_of_year`, `season`) should be encoded cyclically (sin/cos) to preserve continuity across year/day boundaries.
- `--ndvi` (MODIS NDVI vegetation index) is listed as a future feature — not yet fetched.
