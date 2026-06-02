#!/usr/bin/env python3
"""Build a small server-side audit batch for site-derived weather clips.

The script uses recording metadata, env metadata, and an S3 listing of already
chunked Site 257 webm clips. It can export short listening previews from S3 on
the server, then writes an audit manifest with blank human review fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


POLICY_VERSION = "site_clip_filtering_v0.2"
DEFAULT_BUCKET = "eco-acoustic-data.store.adelaideuni.cloud"
DEFAULT_SOURCE_PREFIX = "dataset/original/site_257_bowra-dry-a/downloaded_clips"
COARSE_CLIP_SECONDS = 300.0

S3_LISTING_RE = re.compile(
    r"^\S+\s+\S+\s+(?P<size>\d+)\s+"
    r"(?P<path>site_257_item_(?P<item_id>\d+)/"
    r"site_257_item_(?P=item_id)_clip_(?P<clip_num>\d+)\.webm)$"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except ValueError:
        return default


def parse_s3_listing(path: Path) -> dict[str, list[dict[str, object]]]:
    by_item: dict[str, list[dict[str, object]]] = {}
    with path.open() as f:
        for line in f:
            match = S3_LISTING_RE.match(line.strip())
            if not match:
                continue
            item_id = match.group("item_id")
            entry = {
                "item_id": item_id,
                "clip_num": int(match.group("clip_num")),
                "path": match.group("path"),
                "size_bytes": int(match.group("size")),
            }
            by_item.setdefault(item_id, []).append(entry)

    for clips in by_item.values():
        clips.sort(key=lambda item: int(item["clip_num"]))

    return by_item


def stable_int(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def infer_bucket(row: dict[str, str], bucket: str) -> bool:
    precipitation = parse_float(row, "precipitation_mm")
    daily_precip = parse_float(row, "precipitation_daily_mm")
    wind = parse_float(row, "wind_speed_ms")
    wind_max = parse_float(row, "wind_max_ms")

    if bucket == "light_rain":
        return 0 < precipitation < 2
    if bucket == "medium_or_heavy_rain":
        return precipitation >= 2
    if bucket == "light_wind":
        return precipitation <= 0 and 2 <= wind < 6
    if bucket == "medium_wind":
        return precipitation <= 0 and (wind >= 6 or wind_max >= 6)
    if bucket == "storm_or_thunder_prior":
        return precipitation >= 5 or (daily_precip >= 10 and (wind >= 4 or wind_max >= 6))
    if bucket == "quiet_ambience_control":
        return precipitation <= 0 and wind < 2

    raise ValueError(f"Unknown bucket: {bucket}")


def intensity_for(row: dict[str, str], bucket: str) -> str:
    precipitation = parse_float(row, "precipitation_mm")
    wind = parse_float(row, "wind_speed_ms")

    if "rain" in bucket or bucket == "storm_or_thunder_prior":
        if precipitation >= 5:
            return "heavy"
        if precipitation >= 2:
            return "medium"
        if precipitation > 0:
            return "light"
    if "wind" in bucket:
        if wind >= 10:
            return "strong"
        if wind >= 6:
            return "medium"
        if wind >= 2:
            return "light"
    return "none"


def env_prior_for(row: dict[str, str], bucket: str) -> float:
    precipitation = parse_float(row, "precipitation_mm")
    wind = parse_float(row, "wind_speed_ms")
    wind_max = parse_float(row, "wind_max_ms")

    if "rain" in bucket:
        if precipitation >= 5:
            return 0.90
        if precipitation >= 2:
            return 0.75
        if precipitation > 0:
            return 0.55
    if "wind" in bucket:
        wind_signal = max(wind, wind_max)
        if wind_signal >= 10:
            return 0.90
        if wind_signal >= 6:
            return 0.75
        if wind_signal >= 2:
            return 0.50
    if bucket == "storm_or_thunder_prior":
        if precipitation >= 5:
            return 0.50
        return 0.35
    if bucket == "quiet_ambience_control":
        return 0.40
    return 0.0


def choose_window(
    recording_id: str,
    bucket: str,
    variant: int,
    clips: list[dict[str, object]],
    recording_duration: float,
    preview_seconds: float,
) -> tuple[dict[str, object], float, float]:
    usable = [
        clip
        for clip in clips
        if (int(clip["clip_num"]) - 1) * COARSE_CLIP_SECONDS + preview_seconds
        <= recording_duration
    ]
    if not usable:
        usable = clips

    index = stable_int(recording_id, bucket, variant) % len(usable)
    coarse = usable[index]
    coarse_start = (int(coarse["clip_num"]) - 1) * COARSE_CLIP_SECONDS
    remaining = max(0.0, min(COARSE_CLIP_SECONDS, recording_duration - coarse_start))
    max_inner_start = max(0.0, remaining - preview_seconds)
    inner_start = 0.0
    if max_inner_start > 0:
        inner_start = float(stable_int(bucket, recording_id, variant, "offset") % int(max_inner_start + 1))

    recording_start = coarse_start + inner_start
    return coarse, inner_start, recording_start


def build_candidates(
    items_rows: list[dict[str, str]],
    env_rows: list[dict[str, str]],
    listing: dict[str, list[dict[str, object]]],
    preview_seconds: float,
) -> list[dict[str, object]]:
    items_by_id = {row["id"]: row for row in items_rows}
    env_by_recording = {row["recording_id"]: row for row in env_rows}

    bucket_targets = {
        "light_rain": 20,
        "medium_or_heavy_rain": 20,
        "light_wind": 20,
        "medium_wind": 20,
        "storm_or_thunder_prior": 10,
        "quiet_ambience_control": 10,
    }

    candidates: list[dict[str, object]] = []
    used_clip_ids: set[str] = set()

    for bucket, target in bucket_targets.items():
        matching = [
            row
            for row in env_rows
            if row["recording_id"] in items_by_id
            and row["recording_id"] in listing
            and infer_bucket(row, bucket)
        ]
        matching.sort(
            key=lambda row: (
                -env_prior_for(row, bucket),
                row.get("recorded_date_utc", ""),
                row["recording_id"],
            )
        )

        bucket_count = 0
        max_variants = max(1, math.ceil(target / max(1, len(matching))))
        for variant in range(max_variants):
            for env_row in matching:
                if bucket_count >= target:
                    break
                recording_id = env_row["recording_id"]
                item = items_by_id[recording_id]
                recording_duration = parse_float(item, "duration_seconds")
                coarse, inner_start, recording_start = choose_window(
                    recording_id=recording_id,
                    bucket=bucket,
                    variant=variant,
                    clips=listing[recording_id],
                    recording_duration=recording_duration,
                    preview_seconds=preview_seconds,
                )
                recording_end = recording_start + preview_seconds
                clip_id = (
                    f"site257_{recording_id}_{int(recording_start):06d}_"
                    f"{int(recording_end):06d}_{bucket}"
                )
                if clip_id in used_clip_ids:
                    continue
                used_clip_ids.add(clip_id)
                bucket_count += 1

                weather_type = "ambience"
                if "rain" in bucket:
                    weather_type = "rain"
                elif "wind" in bucket:
                    weather_type = "wind"
                elif "storm" in bucket or "thunder" in bucket:
                    weather_type = "storm"

                candidates.append(
                    {
                        "clip_id": clip_id,
                        "candidate_bucket": bucket,
                        "site_id": item.get("site_id", "257"),
                        "recording_id": recording_id,
                        "item_id": recording_id,
                        "recorded_date_utc": env_row.get("recorded_date_utc", item.get("recorded_date", "")),
                        "sample_bin": env_row.get("sample_bin", item.get("sample_bin", "")),
                        "sample_local_date": env_row.get("sample_local_date", item.get("sample_local_date", "")),
                        "s3_key": coarse["path"],
                        "coarse_clip_num": f"{int(coarse['clip_num']):03d}",
                        "coarse_size_bytes": coarse["size_bytes"],
                        "coarse_inner_start_seconds": round(inner_start, 3),
                        "recording_start_offset_seconds": round(recording_start, 3),
                        "duration_seconds": preview_seconds,
                        "weather_type": weather_type,
                        "weather_intensity": intensity_for(env_row, bucket),
                        "env_prior_score": env_prior_for(env_row, bucket),
                        "precipitation_mm": parse_float(env_row, "precipitation_mm"),
                        "precipitation_daily_mm": parse_float(env_row, "precipitation_daily_mm"),
                        "wind_speed_ms": parse_float(env_row, "wind_speed_ms"),
                        "wind_max_ms": parse_float(env_row, "wind_max_ms"),
                        "humidity_pct": parse_float(env_row, "humidity_pct"),
                        "temperature_c": parse_float(env_row, "temperature_c"),
                        "audio_quality_score": "",
                        "human_weather_label": "",
                        "human_intensity_label": "",
                        "human_accept": "",
                        "human_reject_reason": "",
                        "notes": "",
                    }
                )

    return candidates


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def export_previews(
    candidates: list[dict[str, object]],
    output_dir: Path,
    bucket: str,
    source_prefix: str,
    preview_format: str,
    aws_extra_args: list[str],
) -> None:
    preview_dir = output_dir / "previews"
    cache_dir = output_dir / "cache"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for index, candidate in enumerate(candidates, start=1):
        s3_key = str(candidate["s3_key"])
        source_uri = f"s3://{bucket}/{source_prefix}/{s3_key}"
        local_source = cache_dir / s3_key.replace("/", "__")
        preview_name = f"{index:03d}_{candidate['candidate_bucket']}_{candidate['clip_id']}.{preview_format}"
        preview_path = preview_dir / preview_name

        if not local_source.exists():
            run_command(
                [
                    "aws",
                    "s3",
                    "cp",
                    source_uri,
                    str(local_source),
                    "--only-show-errors",
                    *aws_extra_args,
                ]
            )

        ffmpeg_args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(candidate["coarse_inner_start_seconds"]),
            "-t",
            str(candidate["duration_seconds"]),
            "-i",
            str(local_source),
        ]
        if preview_format == "mp3":
            ffmpeg_args.extend(["-ac", "1", "-ar", "22050", "-b:a", "96k"])
        elif preview_format == "wav":
            ffmpeg_args.extend(["-ac", "1", "-ar", "22050"])
        else:
            raise ValueError(f"Unsupported preview format: {preview_format}")

        ffmpeg_args.append(str(preview_path))
        run_command(ffmpeg_args)
        candidate["preview_path"] = str(preview_path)
        candidate["source_s3_uri"] = source_uri


def write_manifest(path: Path, candidates: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "preview_path",
        "source_s3_uri",
        "s3_key",
        "candidate_bucket",
        "site_id",
        "recording_id",
        "item_id",
        "recorded_date_utc",
        "sample_bin",
        "sample_local_date",
        "coarse_clip_num",
        "coarse_size_bytes",
        "coarse_inner_start_seconds",
        "recording_start_offset_seconds",
        "duration_seconds",
        "weather_type",
        "weather_intensity",
        "env_prior_score",
        "precipitation_mm",
        "precipitation_daily_mm",
        "wind_speed_ms",
        "wind_max_ms",
        "humidity_pct",
        "temperature_c",
        "audio_quality_score",
        "human_weather_label",
        "human_intensity_label",
        "human_accept",
        "human_reject_reason",
        "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({field: candidate.get(field, "") for field in fieldnames})


def write_summary(path: Path, candidates: list[dict[str, object]]) -> None:
    counts: dict[str, int] = {}
    for candidate in candidates:
        bucket = str(candidate["candidate_bucket"])
        counts[bucket] = counts.get(bucket, 0) + 1

    summary = {
        "policy_version": POLICY_VERSION,
        "total_candidates": len(candidates),
        "bucket_counts": counts,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-csv", type=Path, required=True)
    parser.add_argument("--env-csv", type=Path, required=True)
    parser.add_argument("--s3-listing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--preview-seconds", type=float, default=15.0)
    parser.add_argument("--preview-format", choices=["mp3", "wav"], default="mp3")
    parser.add_argument("--export-previews", action="store_true")
    parser.add_argument("--aws-region", default="ap-southeast-2")
    args = parser.parse_args()

    items_rows = read_csv(args.items_csv)
    env_rows = read_csv(args.env_csv)
    listing = parse_s3_listing(args.s3_listing)

    candidates = build_candidates(
        items_rows=items_rows,
        env_rows=env_rows,
        listing=listing,
        preview_seconds=args.preview_seconds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.export_previews:
        export_previews(
            candidates=candidates,
            output_dir=args.output_dir,
            bucket=args.bucket,
            source_prefix=args.source_prefix,
            preview_format=args.preview_format,
            aws_extra_args=["--region", args.aws_region],
        )

    write_manifest(args.output_dir / "audit_manifest.csv", candidates)
    write_summary(args.output_dir / "summary.json", candidates)
    (args.output_dir / "policy_version.txt").write_text(POLICY_VERSION + "\n")

    print(f"Wrote {len(candidates)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
