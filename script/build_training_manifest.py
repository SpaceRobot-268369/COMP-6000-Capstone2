#!/usr/bin/env python3
"""Build a training manifest CSV linking every clip file to its env features.

One row per clip. Joins:
  - site_257_filtered_items.csv  → recording metadata + diel bin
  - site_257_env_data.csv        → 20 environmental conditioning features
  - downloaded_clips/            → actual clip files on disk

Clip start/end offsets are derived from the clip filename index and the
MAX_CLIP_SECONDS=300 segmentation used by download_site_257_clips.py.

Output: resources/site_257_bowra-dry-a/site_257_training_manifest.csv

Usage:
  python3 script/build_training_manifest.py
  python3 script/build_training_manifest.py --dry-run
"""

import argparse
import csv
from pathlib import Path

MAX_CLIP_SECONDS = 300.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    base = root / "resources" / "site_257_bowra-dry-a"
    parser = argparse.ArgumentParser(description="Build training manifest CSV.")
    parser.add_argument("--filtered-csv",  type=Path, default=base / "site_257_filtered_items.csv")
    parser.add_argument("--env-csv",       type=Path, default=base / "site_257_env_data.csv")
    parser.add_argument("--clips-dir",     type=Path, default=base / "downloaded_clips")
    parser.add_argument("--output",        type=Path, default=base / "site_257_training_manifest.csv")
    parser.add_argument("--dry-run",       action="store_true")
    return parser.parse_args()


def build_segments(duration: float) -> list[tuple[float, float]]:
    segments, start = [], 0.0
    while start < duration:
        end = min(start + MAX_CLIP_SECONDS, duration)
        segments.append((start, end))
        start = end
    return segments


def main() -> None:
    args = parse_args()

    # ── Load filtered items ──────────────────────────────────────────────────
    filtered = {r["id"]: r for r in csv.DictReader(open(args.filtered_csv))}
    print(f"Filtered items   : {len(filtered)}")

    # ── Load env data ────────────────────────────────────────────────────────
    env = {r["recording_id"]: r for r in csv.DictReader(open(args.env_csv))}
    print(f"Env data rows    : {len(env)}")

    env_cols = [c for c in next(iter(env.values())).keys()
                if c not in ("count", "recording_id", "recorded_date_utc",
                             "sample_bin", "sample_local_date")]

    # ── Walk clip folders ────────────────────────────────────────────────────
    rows = []
    missing_env = []
    missing_folder = []
    empty_folders = []
    unknown_clips = []

    for recording_id, item in sorted(filtered.items(), key=lambda x: int(x[0])):
        folder = args.clips_dir / f"site_257_item_{recording_id}"
        env_row = env.get(recording_id)

        if not folder.exists():
            missing_folder.append(recording_id)
            continue
        if env_row is None:
            missing_env.append(recording_id)

        clips = sorted(folder.glob("*.webm"))
        if not clips:
            empty_folders.append(recording_id)
            continue

        duration = float(item["duration_seconds"])
        expected_segments = build_segments(duration)

        for clip_path in clips:
            # Parse clip index from filename: site_257_item_{id}_clip_{NNN}.webm
            stem = clip_path.stem  # e.g. site_257_item_1313184_clip_001
            try:
                clip_index = int(stem.rsplit("_", 1)[-1])  # 1-based
            except ValueError:
                unknown_clips.append(str(clip_path))
                continue

            seg_idx = clip_index - 1
            if seg_idx < len(expected_segments):
                clip_start, clip_end = expected_segments[seg_idx]
            else:
                # Clip index beyond expected — use position-based calculation
                clip_start = seg_idx * MAX_CLIP_SECONDS
                clip_end = min(clip_start + MAX_CLIP_SECONDS, duration)

            row = {
                "clip_path":            str(clip_path.relative_to(args.clips_dir.parent.parent.parent)),
                "recording_id":         recording_id,
                "clip_index":           clip_index,
                "clip_start_seconds":   round(clip_start, 3),
                "clip_end_seconds":     round(clip_end, 3),
                "clip_duration_seconds": round(clip_end - clip_start, 3),
                "sample_bin":           item.get("sample_bin", ""),
                "sample_local_date":    item.get("sample_local_date", ""),
            }
            # Append env features
            for col in env_cols:
                row[col] = env_row[col] if env_row else ""

            rows.append(row)

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\nManifest rows    : {len(rows)}")
    print(f"Missing folders  : {len(missing_folder)}")
    print(f"Empty folders    : {len(empty_folders)}  → {empty_folders}")
    print(f"Missing env data : {len(missing_env)}")
    print(f"Unknown clips    : {len(unknown_clips)}")

    from collections import Counter
    bin_counts = Counter(r["sample_bin"] for r in rows)
    print(f"\nClips by diel bin:")
    for b in ["dawn", "morning", "afternoon", "night"]:
        print(f"  {b:<12} {bin_counts[b]:>5} clips")

    if args.dry_run:
        print("\nDry run — no file written.")
        return

    # ── Write ────────────────────────────────────────────────────────────────
    if not rows:
        print("No rows to write.")
        return

    fieldnames = list(rows[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten → {args.output}")


if __name__ == "__main__":
    main()
