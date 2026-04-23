#!/usr/bin/env python3
"""Fetch environmental data for site 257 recordings from NASA POWER.

Data sources:
  - NASA POWER hourly: temperature, humidity, wind, precipitation, solar radiation,
    cloud clearness index, surface pressure
  - NASA POWER daily: temp max/min, daily precip total, max wind speed
  - Derived locally: days_since_rain, sunrise/sunset/daylight hours (astral),
    season, day_of_year, hour_local

All NASA POWER data is in UTC. recorded_date in the CSV is also UTC — no
timezone conversion needed for hourly/daily joins.

Dependencies:
  pip install requests astral

Run from repository root:
  python3 script/fetch_nasa_env_data.py
  python3 script/fetch_nasa_env_data.py --csv-path resources/site_257_bowra-dry-a/site_257_filtered_items.csv
  python3 script/fetch_nasa_env_data.py --output resources/site_257_bowra-dry-a/site_257_env_data.csv
  python3 script/fetch_nasa_env_data.py --ndvi
"""

import argparse
import csv
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Site
# ---------------------------------------------------------------------------
SITE_LAT = -30.04
SITE_LON = 145.87
UTC_OFFSET_HOURS = 10  # Australia/Brisbane — no DST

# ---------------------------------------------------------------------------
# NASA POWER
# ---------------------------------------------------------------------------
POWER_BASE = "https://power.larc.nasa.gov/api/temporal/{resolution}/point"

HOURLY_PARAMS = ",".join([
    "T2M",               # Air temperature at 2m (°C)
    "RH2M",              # Relative humidity at 2m (%)
    "WS2M",              # Wind speed at 2m (m/s)
    "WD2M",              # Wind direction at 2m (°)
    "PRECTOTCORR",       # Precipitation corrected (mm/hr)
    "ALLSKY_SFC_SW_DWN", # Solar radiation all-sky (W/m²)
    "ALLSKY_KT",         # Cloud clearness index (0–1; 1=clear)
    "PS",                # Surface pressure (kPa)
])

DAILY_PARAMS = ",".join([
    "T2M_MAX",           # Daily max temperature (°C)
    "T2M_MIN",           # Daily min temperature (°C)
    "PRECTOTCORR",       # Daily total precipitation (mm/day)
    "WS2M_MAX",          # Daily max wind speed (m/s)
])

NASA_SENTINEL = -999.0   # NASA POWER missing-data marker

# ---------------------------------------------------------------------------
# Retry config
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 90
RETRY_DELAY = 5
MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_csv = (
        project_root / "resources" / "site_257_bowra-dry-a" / "site_257_filtered_items.csv"
    )
    default_output = (
        project_root / "resources" / "site_257_bowra-dry-a" / "site_257_env_data.csv"
    )
    parser = argparse.ArgumentParser(
        description="Fetch environmental data from NASA POWER and join to filtered recordings CSV."
    )
    parser.add_argument("--csv-path", type=Path, default=default_csv)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--lat", type=float, default=SITE_LAT)
    parser.add_argument("--lon", type=float, default=SITE_LON)
    parser.add_argument(
        "--ndvi",
        action="store_true",
        help="Also fetch MODIS NDVI 16-day vegetation index (not yet implemented).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_recorded_date(value: str) -> datetime:
    value = value.strip().rstrip("Z")
    if "." in value:
        value = value.split(".")[0]
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def clean(val) -> Optional[float]:
    """Return None for NASA sentinel (-999) or missing values, else the float."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if f <= NASA_SENTINEL + 1 else f


def v(val, decimals=2) -> str:
    """Format a value for CSV output: empty string for None/sentinel, else rounded."""
    c = clean(val)
    return "" if c is None else str(round(c, decimals))


def get_with_retry(url: str, label: str) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"  [FETCH] {label} (attempt {attempt})...")
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            print(f"  [OK]    {label}")
            return data
        except Exception as exc:
            print(f"  [WARN]  {label} attempt {attempt} failed: {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"Failed after {MAX_ATTEMPTS} attempts: {label}")


def season_for_month(month: int) -> str:
    """Southern hemisphere seasons."""
    return {12: "summer", 1: "summer",  2: "summer",
             3: "autumn",  4: "autumn",  5: "autumn",
             6: "winter",  7: "winter",  8: "winter",
             9: "spring", 10: "spring", 11: "spring"}[month]


# ---------------------------------------------------------------------------
# NASA POWER fetchers
# ---------------------------------------------------------------------------
def fetch_power_year(resolution: str, params: str, year: int,
                     lat: float, lon: float) -> dict:
    """Return {param_name: {key: value}} for one year."""
    start, end = f"{year}0101", f"{year}1231"
    url = (
        POWER_BASE.format(resolution=resolution)
        + f"?parameters={params}&community=RE"
        + f"&latitude={lat}&longitude={lon}"
        + f"&start={start}&end={end}&format=JSON"
        + ("&time-standard=UTC" if resolution == "hourly" else "")
    )
    data = get_with_retry(url, f"NASA POWER {resolution} year={year}")
    return data["properties"]["parameter"]


def build_hourly_lookup(years: set, lat: float, lon: float) -> dict:
    """Fetch all hourly params for each year.
    Returns {YYYYMMDDHH: {param: value, ...}}.
    """
    print(f"\n[STEP] Fetching NASA POWER hourly data for years: {sorted(years)}")
    raw: dict = {}
    for year in sorted(years):
        yearly = fetch_power_year("hourly", HOURLY_PARAMS, year, lat, lon)
        for param, records in yearly.items():
            raw.setdefault(param, {}).update(records)

    all_keys: set = set()
    for records in raw.values():
        all_keys.update(records.keys())

    lookup = {key: {param: raw[param].get(key) for param in raw} for key in all_keys}
    print(f"  [INFO] Hourly lookup: {len(lookup)} keys across {len(raw)} params")
    return lookup


def build_daily_lookup(min_date: date, max_date: date, lat: float, lon: float) -> dict:
    """Fetch all daily params for the full date range (needed for days_since_rain).
    Returns {YYYYMMDD: {param: value, ...}}.
    """
    years = set(range(min_date.year, max_date.year + 1))
    print(f"\n[STEP] Fetching NASA POWER daily data for years: {sorted(years)}")

    raw: dict = {}
    for year in sorted(years):
        yearly = fetch_power_year("daily", DAILY_PARAMS, year, lat, lon)
        for param, records in yearly.items():
            raw.setdefault(param, {}).update(records)

    all_keys: set = set()
    for records in raw.values():
        all_keys.update(records.keys())

    lookup = {key: {param: raw[param].get(key) for param in raw} for key in all_keys}
    print(f"  [INFO] Daily lookup: {len(lookup)} keys across {len(raw)} params")
    return lookup


def compute_days_since_rain(daily_lookup: dict, utc_date: date,
                             threshold_mm: float = 1.0) -> Optional[int]:
    """Count consecutive dry days before utc_date (up to 90-day lookback).

    Uses UTC date throughout to match daily_lookup keys, which are keyed by
    UTC date (YYYYMMDD) from NASA POWER. Passing local AEST date would cause
    an off-by-one error for dawn/morning recordings where the AEST date is one
    day ahead of the UTC date.
    """
    count = 0
    for offset in range(1, 91):
        d = utc_date - timedelta(days=offset)
        key = d.strftime("%Y%m%d")
        day = daily_lookup.get(key)
        if day is None:
            return None
        precip = clean(day.get("PRECTOTCORR"))
        if precip is None:
            return None
        if precip >= threshold_mm:
            return count
        count += 1
    return count  # 90+ dry days


# ---------------------------------------------------------------------------
# Sunrise / sunset (astral) — returns local AEST times
# ---------------------------------------------------------------------------
def get_sun_times(local_date: date, lat: float, lon: float):
    """Return (sunrise_local, sunset_local, daylight_hours) for a local AEST date.

    Times are in AEST (HH:MM) so that sunrise always precedes sunset on the
    same calendar day. Storing UTC equivalents causes apparent inversion because
    AEST sunrise (~05:xx local = ~19:xx UTC previous day) appears after sunset
    (~17:xx local = ~07:xx UTC same day) when compared as bare HH:MM strings.
    """
    try:
        from astral import LocationInfo
        from astral.sun import sun as astral_sun

        loc = LocationInfo(latitude=lat, longitude=lon, timezone="Australia/Brisbane")
        aware_tz = timezone(timedelta(hours=UTC_OFFSET_HOURS))
        s = astral_sun(loc.observer, date=local_date, tzinfo=aware_tz)

        # s["sunrise"] and s["sunset"] are already in AEST (aware_tz)
        sunrise_local = s["sunrise"].strftime("%H:%M")
        sunset_local  = s["sunset"].strftime("%H:%M")
        daylight      = round((s["sunset"] - s["sunrise"]).total_seconds() / 3600, 2)
        return sunrise_local, sunset_local, daylight
    except ImportError:
        return "", "", ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.ndvi:
        print("[WARN] --ndvi flag is not yet implemented. MODIS NDVI fetch will be added in a future update.")
        print("[WARN] Continuing without NDVI data.")

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    recordings: list = []
    with args.csv_path.open("r", encoding="utf-8", newline="") as f:
        recordings = list(csv.DictReader(f))
    print(f"[INFO] Loaded {len(recordings)} recordings from {args.csv_path.name}")

    dts = [parse_recorded_date(r["recorded_date"]) for r in recordings]
    years = {dt.year for dt in dts}
    min_date = min(dt.date() for dt in dts)
    max_date = max(dt.date() for dt in dts)
    print(f"[INFO] Date range: {min_date} → {max_date}, years: {sorted(years)}")

    hourly_lookup = build_hourly_lookup(years, args.lat, args.lon)
    daily_lookup  = build_daily_lookup(min_date, max_date, args.lat, args.lon)

    print("\n[STEP] Joining environmental data to recordings...")
    output_rows = []
    missing_hourly = 0
    missing_daily  = 0

    for row, dt in zip(recordings, dts):
        hourly_key = dt.strftime("%Y%m%d%H")
        daily_key  = dt.strftime("%Y%m%d")           # UTC date — matches daily_lookup keys
        local_dt   = dt + timedelta(hours=UTC_OFFSET_HOURS)
        local_date = local_dt.date()                  # AEST date — used for astral only

        h = hourly_lookup.get(hourly_key, {})
        d = daily_lookup.get(daily_key, {})

        if not h:
            print(f"  [WARN] No hourly data for recording {row['id']} at {hourly_key}")
            missing_hourly += 1
        if not d:
            print(f"  [WARN] No daily data for recording {row['id']} at {daily_key}")
            missing_daily += 1

        sunrise, sunset, daylight = get_sun_times(local_date, args.lat, args.lon)

        # Fix: pass UTC date so days_since_rain lookback uses same key space as daily_lookup
        days_rain = compute_days_since_rain(daily_lookup, dt.date())

        output_rows.append({
            # --- Identity ---
            "count":                  row["count"],
            "recording_id":           row["id"],
            "recorded_date_utc":      row["recorded_date"],
            "sample_bin":             row.get("sample_bin", ""),
            "sample_local_date":      row.get("sample_local_date", ""),

            # --- Hourly (NASA POWER) ---
            "temperature_c":          v(h.get("T2M")),
            "humidity_pct":           v(h.get("RH2M")),
            "wind_speed_ms":          v(h.get("WS2M")),
            "wind_direction_deg":     v(h.get("WD2M"), 1),
            "precipitation_mm":       v(h.get("PRECTOTCORR")),
            "solar_radiation_wm2":    v(h.get("ALLSKY_SFC_SW_DWN")),
            "cloud_clearness_index":  v(h.get("ALLSKY_KT")),   # empty for nighttime (no solar)
            "surface_pressure_kpa":   v(h.get("PS")),

            # --- Daily (NASA POWER) ---
            "temp_max_c":             v(d.get("T2M_MAX")),
            "temp_min_c":             v(d.get("T2M_MIN")),
            "precipitation_daily_mm": v(d.get("PRECTOTCORR")),
            "wind_max_ms":            v(d.get("WS2M_MAX")),
            "days_since_rain":        days_rain if days_rain is not None else "",

            # --- Sun times (AEST local — sunrise always precedes sunset) ---
            "sunrise_local":          sunrise,
            "sunset_local":           sunset,
            "daylight_hours":         daylight,

            # --- Derived temporal ---
            "hour_utc":               dt.hour,
            "hour_local":             local_dt.hour,
            "month":                  local_dt.month,
            "day_of_year":            local_dt.timetuple().tm_yday,
            "season":                 season_for_month(local_dt.month),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n[DONE] Written {len(output_rows)} rows to {args.output}")
    if missing_hourly:
        print(f"[WARN] {missing_hourly} recording(s) missing hourly data")
    if missing_daily:
        print(f"[WARN] {missing_daily} recording(s) missing daily data")


if __name__ == "__main__":
    main()
