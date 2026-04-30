#!/usr/bin/env python3
"""Generate a filtered MVP sample CSV from site_257_all_items.csv.

Sampling policy: Complete-Day Diel Sampling, Stratified by Year-Month.
  - 72 strata (6 years × 12 months)
  - 2 representative days per stratum: 1 from first half of month (days 1-14),
    1 from second half (days 15-end). Selected with a fixed random seed.
  - 4 time-of-day bins per day (AEST local time):
      Dawn      05:00 – 07:00
      Morning   08:00 – 10:00
      Afternoon 13:00 – 15:00
      Night     22:00 – 00:00
  - If the exact bin window has no recording for that day, expand ±1 hour.
    If still empty, skip that (day, bin) — do NOT fill from a different day.
  - Max ~576 recordings.

Output adds two columns to the source schema:
  sample_bin        dawn / morning / afternoon / night
  sample_local_date YYYY-MM-DD (AEST)

Usage:
  python3 script/sample_mvp_dataset.py
  python3 script/sample_mvp_dataset.py --dry-run
  python3 script/sample_mvp_dataset.py --seed 7 --output path/to/out.csv
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

AEST = timezone(timedelta(hours=10))

# Primary bin windows: (start_hour_inclusive, end_hour_exclusive) in local AEST
BINS_PRIMARY = {
    "dawn":      (5,  7),
    "morning":   (8,  10),
    "afternoon": (13, 15),
    "night":     (22, 24),
}

# Fallback windows (±1 hour each side)
BINS_FALLBACK = {
    "dawn":      (4,  8),
    "morning":   (7,  11),
    "afternoon": (12, 16),
    "night":     (21, 24),
}

BIN_ORDER = ["dawn", "morning", "afternoon", "night"]


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.rstrip("Z")).replace(tzinfo=timezone.utc)


def to_aest(utc_dt: datetime) -> datetime:
    return utc_dt.astimezone(AEST)


def local_date_str(utc_dt: datetime) -> str:
    return to_aest(utc_dt).strftime("%Y-%m-%d")


def local_hour(utc_dt: datetime) -> int:
    return to_aest(utc_dt).hour


def in_range(hour: int, window: tuple) -> bool:
    start, end = window
    return start <= hour < end


def main():
    parser = argparse.ArgumentParser(description="Sample MVP dataset from site 257.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--input",
        default="resources/site_257_bowra-dry-a/site_257_all_items.csv",
    )
    parser.add_argument(
        "--output",
        default="resources/site_257_bowra-dry-a/site_257_filtered_items.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # ── Load ────────────────────────────────────────────────────────────────
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"Loaded {len(rows):,} rows from {input_path}")

    # ── Enrich with local date / hour ──────────────────────────────────────
    enriched = []
    for row in rows:
        utc_dt = parse_utc(row["recorded_date"])
        enriched.append({
            "row": row,
            "utc_dt": utc_dt,
            "local_date": local_date_str(utc_dt),
            "local_hour": local_hour(utc_dt),
        })

    # Index: (year, month) → set of local dates
    ym_days: dict[tuple, set] = defaultdict(set)
    # Index: local_date → sorted list of enriched items
    day_items: dict[str, list] = defaultdict(list)

    for item in enriched:
        ld = item["local_date"]
        dt = datetime.fromisoformat(ld)
        ym_days[(dt.year, dt.month)].add(ld)
        day_items[ld].append(item)

    for ld in day_items:
        day_items[ld].sort(key=lambda x: x["utc_dt"])

    # ── Stratified sampling ─────────────────────────────────────────────────
    rng = random.Random(args.seed)
    selected: list[dict] = []
    stats = Counter()

    for (year, month) in sorted(ym_days):
        days = sorted(ym_days[(year, month)])

        first_half  = [d for d in days if int(d[8:10]) <= 14]
        second_half = [d for d in days if int(d[8:10]) >= 15]

        chosen_days = []
        for half in (first_half, second_half):
            if half:
                chosen_days.append(rng.choice(half))

        for chosen_day in chosen_days:
            day_recs = day_items[chosen_day]

            for bin_name in BIN_ORDER:
                # Try primary window
                candidates = [r for r in day_recs if in_range(r["local_hour"], BINS_PRIMARY[bin_name])]

                used_fallback = False
                if not candidates:
                    candidates = [r for r in day_recs if in_range(r["local_hour"], BINS_FALLBACK[bin_name])]
                    used_fallback = bool(candidates)

                if not candidates:
                    stats["skipped"] += 1
                    continue

                picked = candidates[0]
                selected.append({
                    "item": picked,
                    "bin": bin_name,
                    "local_date": chosen_day,
                    "fallback": used_fallback,
                })
                stats["selected"] += 1
                if used_fallback:
                    stats["fallback"] += 1

    # ── Report ──────────────────────────────────────────────────────────────
    ym_counts = Counter()
    bin_counts = Counter()
    fallback_bins = Counter()
    for s in selected:
        ym = s["local_date"][:7]
        ym_counts[ym] += 1
        bin_counts[s["bin"]] += 1
        if s["fallback"]:
            fallback_bins[s["bin"]] += 1

    print(f"\nSampling results:")
    print(f"  Strata covered : {len(ym_counts):>4}  (of 72 possible year-months)")
    print(f"  Recordings     : {stats['selected']:>4}")
    print(f"  Bins skipped   : {stats['skipped']:>4}  (no recording in window or fallback)")
    print(f"  Fallback used  : {stats['fallback']:>4}  bins needed ±1 hr expansion")
    print(f"\nBy time-of-day bin:")
    for b in BIN_ORDER:
        fb = f"  ({fallback_bins[b]} via fallback)" if fallback_bins[b] else ""
        print(f"  {b:<12} {bin_counts[b]:>4}{fb}")

    if args.dry_run:
        print("\nDry run — no file written.")
        return

    # ── Write ───────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fieldnames = fieldnames + ["sample_bin", "sample_local_date"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for i, s in enumerate(selected, start=1):
            row = dict(s["item"]["row"])
            row["count"] = i
            row["sample_bin"] = s["bin"]
            row["sample_local_date"] = s["local_date"]
            writer.writerow(row)

    print(f"\nWritten {len(selected)} rows → {output_path}")


if __name__ == "__main__":
    main()
