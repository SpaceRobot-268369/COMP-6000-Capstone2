#!/usr/bin/env python3
"""Sample ADDITIONAL site-257 items to enlarge the Layer E ambient pool.

Same policy as sample_mvp_dataset.py (Complete-Day Diel Sampling, stratified by
year-month, 4 diel bins/day), but:
  - picks more days per half-month (`--days-per-half`, default 3),
  - over-weights winter (months 6/7/8) by `--winter-extra-days`,
  - EXCLUDES recordings already in an existing filtered set (`--exclude`), so the
    output is disjoint and can be appended (existing downloads/ambient stay valid).

Output schema matches site_257_filtered_items.csv (adds sample_bin,
sample_local_date). Seed 42.

Usage:
  python3 script/dataset/sample_mvp_dataset_ext.py --dry-run
  python3 script/dataset/sample_mvp_dataset_ext.py \
      --days-per-half 3 --winter-extra-days 3 \
      --exclude resources/site_257_bowra-dry-a/site_257_filtered_items.csv \
      --output resources/site_257_bowra-dry-a/site_257_filtered_items_ext.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

AEST = timezone(timedelta(hours=10))
BINS_PRIMARY = {"dawn": (5, 7), "morning": (8, 10), "afternoon": (13, 15), "night": (22, 24)}
BINS_FALLBACK = {"dawn": (4, 8), "morning": (7, 11), "afternoon": (12, 16), "night": (21, 24)}
BIN_ORDER = ["dawn", "morning", "afternoon", "night"]
WINTER_MONTHS = {6, 7, 8}


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.rstrip("Z")).replace(tzinfo=timezone.utc)


def to_aest(utc_dt: datetime) -> datetime:
    return utc_dt.astimezone(AEST)


def local_date_str(utc_dt: datetime) -> str:
    return to_aest(utc_dt).strftime("%Y-%m-%d")


def local_hour(utc_dt: datetime) -> int:
    return to_aest(utc_dt).hour


def in_range(hour: int, window: tuple) -> bool:
    return window[0] <= hour < window[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--days-per-half", type=int, default=3)
    ap.add_argument("--winter-extra-days", type=int, default=3)
    ap.add_argument("--input", default="resources/site_257_bowra-dry-a/site_257_all_items.csv")
    ap.add_argument("--exclude", default="resources/site_257_bowra-dry-a/site_257_filtered_items.csv")
    ap.add_argument("--output", default="resources/site_257_bowra-dry-a/site_257_filtered_items_ext.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    print(f"Loaded {len(rows):,} items from {args.input}")

    exclude_ids: set[str] = set()
    if args.exclude and Path(args.exclude).exists():
        with open(args.exclude, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                exclude_ids.add(r["id"])
    print(f"Excluding {len(exclude_ids)} already-selected recording ids")

    enriched = []
    for row in rows:
        if row["id"] in exclude_ids:
            continue
        utc_dt = parse_utc(row["recorded_date"])
        enriched.append({"row": row, "utc_dt": utc_dt,
                         "local_date": local_date_str(utc_dt), "local_hour": local_hour(utc_dt)})

    ym_days: dict[tuple, set] = defaultdict(set)
    day_items: dict[str, list] = defaultdict(list)
    for item in enriched:
        ld = item["local_date"]
        dt = datetime.fromisoformat(ld)
        ym_days[(dt.year, dt.month)].add(ld)
        day_items[ld].append(item)
    for ld in day_items:
        day_items[ld].sort(key=lambda x: x["utc_dt"])

    rng = random.Random(args.seed)
    selected: list[dict] = []
    used_ids: set[str] = set(exclude_ids)
    stats = Counter()

    for (year, month) in sorted(ym_days):
        days = sorted(ym_days[(year, month)])
        first_half = [d for d in days if int(d[8:10]) <= 14]
        second_half = [d for d in days if int(d[8:10]) >= 15]
        n_days = args.days_per_half + (args.winter_extra_days if month in WINTER_MONTHS else 0)

        chosen_days = []
        for half in (first_half, second_half):
            if half:
                k = min(n_days, len(half))
                chosen_days.extend(rng.sample(half, k))

        for chosen_day in chosen_days:
            day_recs = day_items[chosen_day]
            for bin_name in BIN_ORDER:
                cands = [r for r in day_recs if in_range(r["local_hour"], BINS_PRIMARY[bin_name])]
                used_fb = False
                if not cands:
                    cands = [r for r in day_recs if in_range(r["local_hour"], BINS_FALLBACK[bin_name])]
                    used_fb = bool(cands)
                cands = [r for r in cands if r["row"]["id"] not in used_ids]
                if not cands:
                    stats["skipped"] += 1
                    continue
                picked = cands[0]
                used_ids.add(picked["row"]["id"])
                selected.append({"item": picked, "bin": bin_name, "local_date": chosen_day, "fallback": used_fb})
                stats["selected"] += 1
                if used_fb:
                    stats["fallback"] += 1

    season_of_month = {**{m: "summer" for m in (12, 1, 2)}, **{m: "autumn" for m in (3, 4, 5)},
                       **{m: "winter" for m in (6, 7, 8)}, **{m: "spring" for m in (9, 10, 11)}}
    season_counts = Counter()
    bin_counts = Counter()
    for s in selected:
        mo = int(s["local_date"][5:7])
        season_counts[season_of_month[mo]] += 1
        bin_counts[s["bin"]] += 1

    print(f"\nNew items selected: {stats['selected']}  (skipped {stats['skipped']}, fallback {stats['fallback']})")
    print("By season:", dict(season_counts))
    print("By diel:  ", dict(bin_counts))

    if args.dry_run:
        print("\nDry run — no file written.")
        return

    out_fieldnames = fieldnames + ["sample_bin", "sample_local_date"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames)
        w.writeheader()
        for i, s in enumerate(selected, start=1):
            row = dict(s["item"]["row"])
            row["count"] = i
            row["sample_bin"] = s["bin"]
            row["sample_local_date"] = s["local_date"]
            w.writerow(row)
    print(f"\nWrote {len(selected)} new items → {args.output}")


if __name__ == "__main__":
    main()
