#!/usr/bin/env python3
"""Merge site-weather candidate pools into a conservative Layer B MVP pool.

This script does not create audio. It consolidates existing candidate-pool
manifests, removes duplicate clips, applies recording-level diversity caps, and
writes the final site-derived weather pool that can later be promoted/exported
for Layer D.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


POLICY_VERSION = "site_weather_mvp_pool_v0.1"
DEFAULT_CATEGORY_LIMITS = {
    "rain_primary": 12,
    "rain_wind_mixed": 12,
    "wind_primary": 60,
}
DEFAULT_RECORDING_CAPS = {
    "rain_primary": 1,
    "rain_wind_mixed": 1,
    "wind_primary": 2,
}
CATEGORY_PRIORITY = {
    "rain_primary": 100,
    "wind_primary": 90,
    "rain_wind_mixed": 80,
    "rain_backup_maybe": 40,
    "wind_backup_maybe": 35,
    "wind_with_bio_backup": 30,
    "storm_rain_wind_backup": 20,
    "reject": 0,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except ValueError:
        return default


def parse_map(text: str, defaults: dict[str, int]) -> dict[str, int]:
    values = dict(defaults)
    if not text:
        return values
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Expected key=value entry, got: {part}")
        values[key.strip()] = int(value.strip())
    return values


def safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def near_duplicate_group(row: dict[str, str], bucket_seconds: int) -> str:
    recording_id = row.get("recording_id", "")
    start = safe_int(row.get("recording_start_offset_seconds", "0"))
    bucket = start // max(1, bucket_seconds)
    return f"site257_{recording_id}_{bucket_seconds}s_{bucket:05d}"


def score_key(row: dict[str, str]) -> tuple[float, ...]:
    category = row.get("pool_category", "")
    return (
        float(CATEGORY_PRIORITY.get(category, 0)),
        parse_float(row, "final_score"),
        parse_float(row, "target_clap_score", parse_float(row, "clap_weather_score")),
        parse_float(row, "weather_margin"),
        parse_float(row, "target_vs_other_weather_margin"),
        -parse_float(row, "contamination_score"),
        -parse_float(row, "analysis_clipping_ratio"),
    )


def best_rows_by_clip(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        clip_id = row.get("clip_id", "")
        if not clip_id:
            continue
        if clip_id not in best or score_key(row) > score_key(best[clip_id]):
            best[clip_id] = row
    return list(best.values())


def component_fields(category: str, label: str) -> dict[str, str]:
    has_rain = category in {"rain_primary", "rain_wind_mixed"} or "rain" in label
    has_wind = category in {"wind_primary", "rain_wind_mixed"} or "wind" in label
    return {
        "source_type": "site",
        "has_rain": str(has_rain).lower(),
        "has_wind": str(has_wind).lower(),
        "has_thunder": "false",
        "primary_weather": "rain+wind" if has_rain and has_wind else ("rain" if has_rain else "wind"),
        "mixed_weather": str(has_rain and has_wind).lower(),
        "thunder_mvp_source": "library_fallback",
    }


def select_mvp_rows(
    rows: list[dict[str, str]],
    *,
    category_limits: dict[str, int],
    recording_caps: dict[str, int],
    near_duplicate_seconds: int,
) -> list[dict[str, object]]:
    rows = [
        row
        for row in best_rows_by_clip(rows)
        if row.get("pool_category", "") in category_limits
        and row.get("pool_decision", "") == "accept"
        and row.get("pool_category", "") != "reject"
    ]
    rows.sort(key=score_key, reverse=True)

    selected: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    recording_counts: dict[str, Counter[str]] = defaultdict(Counter)
    near_duplicate_counts: Counter[str] = Counter()

    for row in rows:
        category = row.get("pool_category", "")
        recording_id = row.get("recording_id", "")
        duplicate_group = near_duplicate_group(row, near_duplicate_seconds)
        if category_counts[category] >= category_limits[category]:
            continue
        if recording_counts[category][recording_id] >= recording_caps.get(category, 1):
            continue
        if near_duplicate_counts[duplicate_group] >= 1:
            continue

        category_counts[category] += 1
        recording_counts[category][recording_id] += 1
        near_duplicate_counts[duplicate_group] += 1
        selected.append(
            {
                **row,
                **component_fields(category, row.get("pool_label", "")),
                "mvp_pool_policy_version": POLICY_VERSION,
                "mvp_pool_decision": "default_site_pool",
                "mvp_pool_category": category,
                "recording_group_id": f"site257_{recording_id}",
                "near_duplicate_group_id": duplicate_group,
                "selection_rank_in_category": category_counts[category],
                "site_retrieval_role": "primary" if category in {"rain_primary", "wind_primary"} else "mixed_texture",
                "fallback_required": "true" if category != "wind_primary" else "false",
                "fallback_reason": "pure_weather_library_needed" if category != "wind_primary" else "",
            }
        )

    selected.sort(
        key=lambda row: (
            row["mvp_pool_category"],
            safe_int(str(row["selection_rank_in_category"])),
        )
    )
    return selected


def write_summary(path: Path, selected: list[dict[str, object]], source_rows: int) -> None:
    summary: dict[str, object] = {
        "policy_version": POLICY_VERSION,
        "source_rows": source_rows,
        "selected_rows": len(selected),
        "selected_category_counts": {},
        "selected_recording_counts": {},
        "component_counts": {
            "has_rain": 0,
            "has_wind": 0,
            "has_thunder": 0,
        },
    }
    recordings_by_category: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        category = str(row.get("mvp_pool_category", ""))
        counts = summary["selected_category_counts"]  # type: ignore[index]
        counts[category] = counts.get(category, 0) + 1
        recordings_by_category[category].add(str(row.get("recording_id", "")))
        for key in ["has_rain", "has_wind", "has_thunder"]:
            if row.get(key) == "true":
                component_counts = summary["component_counts"]  # type: ignore[index]
                component_counts[key] += 1

    summary["selected_recording_counts"] = {
        category: len(recordings)
        for category, recordings in sorted(recordings_by_category.items())
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category-limits", default="")
    parser.add_argument("--recording-caps", default="")
    parser.add_argument("--near-duplicate-seconds", type=int, default=60)
    args = parser.parse_args()

    category_limits = parse_map(args.category_limits, DEFAULT_CATEGORY_LIMITS)
    recording_caps = parse_map(args.recording_caps, DEFAULT_RECORDING_CAPS)
    rows: list[dict[str, str]] = []
    for path in args.candidate_pool_manifest:
        for row in read_csv(path):
            row["source_candidate_pool_manifest"] = str(path)
            rows.append(row)

    selected = select_mvp_rows(
        rows,
        category_limits=category_limits,
        recording_caps=recording_caps,
        near_duplicate_seconds=args.near_duplicate_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "site_weather_mvp_pool_manifest.csv", selected)
    write_summary(args.output_dir / "summary.json", selected, len(rows))
    (args.output_dir / "policy_version.txt").write_text(POLICY_VERSION + "\n", encoding="utf-8")
    print(f"Selected {len(selected)} MVP site-weather rows into {args.output_dir}")


if __name__ == "__main__":
    main()
