#!/usr/bin/env python3
"""Promote curated site-weather candidates into Layer D-ready WAV assets.

The candidate-pool manifest records analysis windows and review previews. This
script creates the actual mixer-ready assets by resampling accepted windows to
22,050 Hz mono WAV and writing a Layer D-ready manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import wave
from pathlib import Path


POLICY_VERSION = "site_weather_candidate_promotion_v0.1"
TARGET_SAMPLE_RATE_HZ = 22050
TARGET_CHANNELS = 1


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


def should_promote(row: dict[str, str], include_backup: bool) -> tuple[bool, str]:
    decision = row.get("pool_decision", "")
    category = row.get("pool_category", "")
    label = row.get("pool_label", "")
    action = row.get("layer_d_export_action", "")
    clipping = parse_float(row, "analysis_clipping_ratio")

    if action != "export_22050hz_wav_after_spot_check":
        return False, "export_action_not_enabled"
    if decision == "accept":
        pass
    elif decision == "backup" and include_backup:
        pass
    else:
        return False, "decision_not_selected"
    if category == "reject" or not label:
        return False, "reject_or_missing_label"
    if label == "thunder":
        return False, "site_thunder_not_promoted_in_mvp"
    if clipping >= 0.005:
        return False, "clipping_or_overload_risk"
    return True, "selected"


def output_name(row: dict[str, str]) -> str:
    category = row.get("pool_category", "unknown")
    label = row.get("pool_label", "unknown").replace("+", "_")
    clip_id = row.get("clip_id", "clip")
    return f"{category}__{label}__{clip_id}.wav"


def run_ffmpeg(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE_HZ),
        str(destination),
    ]
    subprocess.run(args, check=True)


def wav_info(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
    return {
        "layer_d_asset_sample_rate_hz": sample_rate,
        "layer_d_asset_channels": channels,
        "layer_d_asset_duration_seconds": round(frames / sample_rate, 3) if sample_rate else 0,
    }


def promote_rows(
    rows: list[dict[str, str]],
    output_dir: Path,
    include_backup: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    asset_dir = output_dir / "assets_wav_22050_mono"
    promoted: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for row in rows:
        selected, reason = should_promote(row, include_backup)
        if not selected:
            skipped.append({**row, "promotion_skip_reason": reason, "promotion_policy_version": POLICY_VERSION})
            continue

        source = Path(row.get("wav_path", ""))
        if not source.exists():
            skipped.append(
                {
                    **row,
                    "promotion_skip_reason": "source_wav_missing",
                    "promotion_policy_version": POLICY_VERSION,
                }
            )
            continue

        destination = asset_dir / row.get("pool_category", "unknown") / output_name(row)
        run_ffmpeg(source, destination)
        info = wav_info(destination)

        if info["layer_d_asset_sample_rate_hz"] != TARGET_SAMPLE_RATE_HZ or info["layer_d_asset_channels"] != TARGET_CHANNELS:
            skipped.append(
                {
                    **row,
                    "promotion_skip_reason": "export_verification_failed",
                    "promotion_policy_version": POLICY_VERSION,
                    **info,
                }
            )
            continue

        promoted.append(
            {
                **row,
                "layer_d_ready": "true",
                "layer_d_asset_status": "ready",
                "layer_d_asset_path": str(destination),
                "layer_d_asset_format": "wav",
                "promotion_policy_version": POLICY_VERSION,
                **info,
            }
        )

    return promoted, skipped


def write_summary(path: Path, promoted: list[dict[str, object]], skipped: list[dict[str, object]]) -> None:
    summary: dict[str, object] = {
        "policy_version": POLICY_VERSION,
        "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "target_channels": TARGET_CHANNELS,
        "promoted_rows": len(promoted),
        "skipped_rows": len(skipped),
        "promoted_pool_category_counts": {},
        "promoted_pool_label_counts": {},
        "skip_reason_counts": {},
    }
    for row in promoted:
        for field, key in [
            ("pool_category", "promoted_pool_category_counts"),
            ("pool_label", "promoted_pool_label_counts"),
        ]:
            value = str(row.get(field, ""))
            counts = summary[key]  # type: ignore[index]
            counts[value] = counts.get(value, 0) + 1
    for row in skipped:
        value = str(row.get("promotion_skip_reason", ""))
        counts = summary["skip_reason_counts"]  # type: ignore[index]
        counts[value] = counts.get(value, 0) + 1

    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-backup", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.candidate_pool_manifest)
    promoted, skipped = promote_rows(rows, args.output_dir, args.include_backup)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "layer_d_ready_manifest.csv", promoted)
    write_csv(args.output_dir / "promotion_skipped_manifest.csv", skipped)
    write_summary(args.output_dir / "summary.json", promoted, skipped)
    (args.output_dir / "policy_version.txt").write_text(POLICY_VERSION + "\n", encoding="utf-8")
    print(f"Promoted {len(promoted)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
