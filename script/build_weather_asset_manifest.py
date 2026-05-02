#!/usr/bin/env python3
"""Build the Layer B weather asset manifest from downloaded clips.

This script does not download or commit audio. It analyses existing clips from
the training manifest and writes a small CSV index with lightweight audio
features and wind/rain scores for retrieval.
"""

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from layer_b import analyse_weather_features  # noqa: E402


DEFAULT_RESOURCE_DIR = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build analysed weather asset manifest for Layer B.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_RESOURCE_DIR / "site_257_training_manifest.csv",
        help="Training manifest with clip paths and environmental fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESOURCE_DIR / "weather_asset_manifest.csv",
        help="Output CSV path for Layer B weather asset index.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=800,
        help="Maximum candidate clips to analyse. Use 0 for all candidates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print candidate counts without analysing audio or writing output.",
    )
    return parser.parse_args()


def to_float(value, default=0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_priority(row: dict) -> float:
    wind = max(to_float(row.get("wind_speed_ms")), to_float(row.get("wind_max_ms")) * 0.65)
    rain = max(to_float(row.get("precipitation_mm")), to_float(row.get("precipitation_daily_mm")) / 8.0)
    humidity = to_float(row.get("humidity_pct"), 50)
    recent_rain = max(0.0, 1.0 - to_float(row.get("days_since_rain"), 30) / 14.0)
    return max(wind / 10.0, rain / 4.0 + humidity / 300.0 + recent_rain * 0.25)


def likely_weather_candidate(row: dict) -> bool:
    wind = max(to_float(row.get("wind_speed_ms")), to_float(row.get("wind_max_ms")) * 0.65)
    rain = max(to_float(row.get("precipitation_mm")), to_float(row.get("precipitation_daily_mm")) / 8.0)
    humidity = to_float(row.get("humidity_pct"), 50)
    return wind >= 2.0 or rain > 0.05 or humidity >= 85.0


def clip_exists(row: dict) -> bool:
    path = PROJECT_ROOT / row["clip_path"]
    return path.exists() and path.stat().st_size > 0


def load_candidates(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = [row for row in rows if likely_weather_candidate(row) and clip_exists(row)]
    candidates.sort(key=candidate_priority, reverse=True)
    return candidates


def output_fieldnames(rows: list[dict]) -> list[str]:
    base = [
        "clip_path",
        "recording_id",
        "clip_index",
        "clip_start_seconds",
        "clip_end_seconds",
        "clip_duration_seconds",
        "sample_bin",
        "sample_local_date",
        "month",
        "month_range",
        "temperature_c",
        "humidity_pct",
        "wind_speed_ms",
        "wind_direction_deg",
        "wind_max_ms",
        "precipitation_mm",
        "precipitation_daily_mm",
        "days_since_rain",
        "analysis_status",
        "analysis_error",
    ]
    feature_cols = [
        "rms_db",
        "low_ratio",
        "mid_ratio",
        "high_ratio",
        "spectral_centroid_hz",
        "spectral_flatness",
        "zero_crossing_rate",
        "gustiness_db",
        "transient_rate",
        "wind_audio_score",
        "rain_audio_score",
        "wind_intensity_audio",
        "rain_intensity_audio",
    ]
    return base + feature_cols


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Training manifest not found: {args.manifest}")

    candidates = load_candidates(args.manifest)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    print(f"Candidates: {len(candidates)} from {args.manifest}")
    if args.dry_run:
        return

    out_rows = []
    for row in tqdm(candidates, desc="Analysing weather candidates", unit="clip"):
        out = {key: row.get(key, "") for key in output_fieldnames([])}
        try:
            source = PROJECT_ROOT / row["clip_path"]
            offset = 0.0
            duration = min(45.0, max(5.0, to_float(row.get("clip_duration_seconds"), 45.0)))
            features = analyse_weather_features(source, duration=duration, offset=offset)
            out.update(features)
            out["analysis_status"] = "ok"
            out["analysis_error"] = ""
        except Exception as exc:
            out["analysis_status"] = "failed"
            out["analysis_error"] = str(exc) or exc.__class__.__name__
        out_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames(out_rows))
        writer.writeheader()
        writer.writerows(out_rows)

    ok = sum(1 for row in out_rows if row["analysis_status"] == "ok")
    failed = len(out_rows) - ok
    print(f"Written {len(out_rows)} rows -> {args.output}")
    print(f"Analysis ok: {ok} | failed: {failed}")


if __name__ == "__main__":
    main()
