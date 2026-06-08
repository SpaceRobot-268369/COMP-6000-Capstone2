"""Build an env-first Site257 rain listening queue.

This script does not download audio and does not label clips as rain. It uses
Site257 environmental metadata to prioritise recordings that are most worth
listening to, then writes a traceable human-audit manifest.

Run from the repository root:

    ./acoustic_ai/.venv/bin/python acoustic_ai/layers/layer_e/attempts/liting__mvp_3__site257_weather_audit_dataset/code/build_env_rain_scout_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_TRAINING_MANIFEST = (
    PROJECT_ROOT
    / "resources"
    / "site_257_bowra-dry-a"
    / "site_257_training_manifest.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "debug" / "e_b_site257_env_rain_scout"

FIELDNAMES = [
    "rank",
    "clip_id",
    "audio_path",
    "source_site_id",
    "source_recording_id",
    "clip_index",
    "clip_start_seconds",
    "clip_end_seconds",
    "duration_s",
    "sample_bin",
    "sample_local_date",
    "hour_local",
    "season",
    "precipitation_mm",
    "precipitation_daily_mm",
    "wind_speed_ms",
    "wind_max_ms",
    "humidity_pct",
    "temperature_c",
    "rain_env_bucket",
    "wind_env_bucket",
    "selection_reason",
    "review_priority",
    "expected_local_path_after_dvc_pull",
    "human_rain_intensity",
    "human_wind_intensity",
    "bird_activity",
    "insect_activity",
    "background_noise",
    "audit_status",
    "auditor",
    "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-recordings", type=int, default=30)
    parser.add_argument("--clips-per-recording", type=int, default=8)
    parser.add_argument("--min-hourly-rain-mm", type=float, default=0.25)
    parser.add_argument("--min-daily-rain-mm", type=float, default=1.0)
    args = parser.parse_args()

    rows = read_csv(args.training_manifest)
    if not rows:
        raise SystemExit(f"No rows found in {args.training_manifest}")

    recording_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        recording_groups[row["recording_id"]].append(row)

    scored_recordings = []
    for recording_id, group in recording_groups.items():
        first = group[0]
        hourly = parse_float(first, "precipitation_mm")
        daily = parse_float(first, "precipitation_daily_mm")
        wind = parse_float(first, "wind_speed_ms")
        wind_max = parse_float(first, "wind_max_ms")
        if hourly < args.min_hourly_rain_mm and daily < args.min_daily_rain_mm:
            continue
        rain_score = hourly + 0.25 * daily
        wind_penalty = max(0.0, wind - 4.0) * 0.15 + max(0.0, wind_max - 6.0) * 0.05
        score = rain_score - wind_penalty
        scored_recordings.append((score, hourly, daily, wind, wind_max, recording_id, group))

    scored_recordings.sort(reverse=True)
    selected_recordings = scored_recordings[: args.top_recordings]

    out_rows: list[dict[str, str]] = []
    for _, hourly, daily, wind, wind_max, recording_id, group in selected_recordings:
        group = sorted(group, key=lambda item: int(float(item.get("clip_index", "0") or 0)))
        selected_clips = choose_review_clips(group, args.clips_per_recording)
        for row in selected_clips:
            clip_index = int(float(row.get("clip_index", "0") or 0))
            clip_id = f"site257_{recording_id}_clip{clip_index:03d}"
            out_rows.append(
                {
                    "rank": "",
                    "clip_id": clip_id,
                    "audio_path": row.get("clip_path", ""),
                    "source_site_id": "257",
                    "source_recording_id": recording_id,
                    "clip_index": str(clip_index),
                    "clip_start_seconds": row.get("clip_start_seconds", ""),
                    "clip_end_seconds": row.get("clip_end_seconds", ""),
                    "duration_s": row.get("clip_duration_seconds", ""),
                    "sample_bin": row.get("sample_bin", ""),
                    "sample_local_date": row.get("sample_local_date", ""),
                    "hour_local": row.get("hour_local", ""),
                    "season": row.get("season", ""),
                    "precipitation_mm": f"{hourly:g}",
                    "precipitation_daily_mm": f"{daily:g}",
                    "wind_speed_ms": f"{wind:g}",
                    "wind_max_ms": f"{wind_max:g}",
                    "humidity_pct": row.get("humidity_pct", ""),
                    "temperature_c": row.get("temperature_c", ""),
                    "rain_env_bucket": rain_bucket(hourly, daily),
                    "wind_env_bucket": wind_bucket(wind, wind_max),
                    "selection_reason": "env rain prior; requires audio/CLAP/human confirmation",
                    "review_priority": review_priority(hourly, daily, wind, wind_max),
                    "expected_local_path_after_dvc_pull": row.get("clip_path", ""),
                    "human_rain_intensity": "",
                    "human_wind_intensity": "",
                    "bird_activity": "",
                    "insect_activity": "",
                    "background_noise": "",
                    "audit_status": "pending",
                    "auditor": "",
                    "notes": "",
                }
            )

    for index, row in enumerate(out_rows, start=1):
        row["rank"] = str(index)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "env_rain_scout_manifest.csv"
    out_json = args.out_dir / "summary.json"
    write_csv(out_csv, out_rows)
    summary = {
        "total_rows": len(out_rows),
        "selected_recordings": len(selected_recordings),
        "top_recordings": [
            {
                "recording_id": recording_id,
                "score": round(score, 4),
                "precipitation_mm": hourly,
                "precipitation_daily_mm": daily,
                "wind_speed_ms": wind,
                "wind_max_ms": wind_max,
                "clips_available": len(group),
            }
            for score, hourly, daily, wind, wind_max, recording_id, group in selected_recordings
        ],
        "note": "Env metadata is only a scout prior; audio must be confirmed with CLAP and human review.",
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Rain env scout manifest written to: {out_csv}")
    print(f"Summary written to: {out_json}")
    print(f"Rows: {len(out_rows)}, recordings: {len(selected_recordings)}")
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def choose_review_clips(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return [rows[len(rows) // 2]]

    indexes = []
    max_index = len(rows) - 1
    for step in range(limit):
        indexes.append(round(step * max_index / (limit - 1)))
    seen = set()
    selected = []
    for index in indexes:
        if index in seen:
            continue
        seen.add(index)
        selected.append(rows[index])
    return selected


def rain_bucket(hourly: float, daily: float) -> str:
    if hourly >= 20 or daily >= 20:
        return "very_high_rain_env"
    if hourly >= 5 or daily >= 5:
        return "high_rain_env"
    if hourly >= 1 or daily >= 2:
        return "moderate_rain_env"
    return "light_rain_env"


def wind_bucket(wind: float, wind_max: float) -> str:
    wind_signal = max(wind, wind_max)
    if wind >= 6 or wind_signal >= 8:
        return "strong_wind_env"
    if wind >= 3 or wind_signal >= 5:
        return "moderate_wind_env"
    if wind > 0:
        return "light_wind_env"
    return "no_wind_env"


def review_priority(hourly: float, daily: float, wind: float, wind_max: float) -> str:
    rain = rain_bucket(hourly, daily)
    wind_level = wind_bucket(wind, wind_max)
    if rain in {"very_high_rain_env", "high_rain_env"} and wind_level in {"no_wind_env", "light_wind_env"}:
        return "priority_a_possible_clean_rain"
    if rain in {"very_high_rain_env", "high_rain_env"}:
        return "priority_b_rain_with_possible_wind"
    if rain == "moderate_rain_env" and wind_level in {"no_wind_env", "light_wind_env"}:
        return "priority_c_light_or_moderate_rain"
    return "priority_d_backup"


if __name__ == "__main__":
    raise SystemExit(main())
