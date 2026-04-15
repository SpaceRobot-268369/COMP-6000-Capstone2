#!/usr/bin/env python3
"""Download site 257 audio clips from CSV with 300-second chunking."""

import argparse
import csv
from pathlib import Path
from typing import Iterable

import requests

BASE_MEDIA_URL = "https://api.acousticobservatory.org/audio_recordings/{recording_id}/media.webm"
MAX_CLIP_SECONDS = 300.0
REQUEST_TIMEOUT_SECONDS = 120


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    default_csv = (
        project_root
        / "resources"
        / "site_257_bowra-dry-a"
        / "site_257_all_items.csv"
    )
    default_output = (
        project_root / "resources" / "site_257_bowra-dry-a" / "downloaded_clips"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Download clips for site 257 items from CSV. "
            "Large items are split into clips of at most 300 seconds."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where item folders and clips are written.",
    )
    parser.add_argument(
        "--start-item",
        type=int,
        required=True,
        help="1-based start item index in the CSV (data rows only).",
    )
    parser.add_argument(
        "--end-item",
        type=int,
        required=True,
        help="1-based end item index in the CSV (inclusive, data rows only).",
    )
    return parser.parse_args()


def build_segments(duration: float, max_clip_seconds: float = MAX_CLIP_SECONDS) -> list[tuple[float, float]]:
    if duration <= 0:
        return []

    segments: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + max_clip_seconds, duration)
        segments.append((start, end))
        start = end
    return segments


def iter_selected_rows(csv_path: Path, start_item: int, end_item: int) -> Iterable[tuple[int, dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for idx, row in enumerate(reader, start=1):
            if idx < start_item:
                continue
            if idx > end_item:
                break
            yield idx, row


def download_segment(
    session: requests.Session,
    recording_id: str,
    start_offset: float,
    end_offset: float,
    output_path: Path,
) -> None:
    url = BASE_MEDIA_URL.format(recording_id=recording_id)
    params = {"start_offset": f"{start_offset:.3f}", "end_offset": f"{end_offset:.3f}"}
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def main() -> None:
    args = parse_args()

    if args.start_item < 1:
        raise ValueError("--start-item must be >= 1")
    if args.end_item < args.start_item:
        raise ValueError("--end-item must be >= --start-item")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        for csv_index, row in iter_selected_rows(args.csv_path, args.start_item, args.end_item):
            item_id = (row.get("id") or "").strip()
            duration_raw = (row.get("duration_seconds") or "").strip()

            if not item_id:
                print(f"[SKIP] row {csv_index}: missing id")
                continue
            if not duration_raw:
                print(f"[SKIP] row {csv_index} item {item_id}: missing duration_seconds")
                continue

            try:
                duration_seconds = float(duration_raw)
            except ValueError:
                print(
                    f"[SKIP] row {csv_index} item {item_id}: invalid duration_seconds={duration_raw!r}"
                )
                continue

            item_folder = args.output_dir / f"site_257_item_{item_id}"
            if item_folder.exists():
                print(f"[SKIP] row {csv_index} item {item_id}: folder exists ({item_folder})")
                continue

            item_folder.mkdir(parents=True, exist_ok=False)
            segments = build_segments(duration_seconds, MAX_CLIP_SECONDS)
            if not segments:
                print(f"[SKIP] row {csv_index} item {item_id}: non-positive duration")
                continue

            print(
                f"[ITEM] row {csv_index} item {item_id}: duration={duration_seconds:.3f}s clips={len(segments)}"
            )

            try:
                for clip_num, (start_offset, end_offset) in enumerate(segments, start=1):
                    clip_name = f"site_257_item_{item_id}_clip_{clip_num:03d}.webm"
                    clip_path = item_folder / clip_name
                    download_segment(
                        session=session,
                        recording_id=item_id,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        output_path=clip_path,
                    )
                    print(
                        "  "
                        f"downloaded {clip_name} "
                        f"(start={start_offset:.3f}, end={end_offset:.3f})"
                    )
            except Exception as exc:
                print(f"[ERROR] row {csv_index} item {item_id}: {exc}")
                print("        keeping partial downloads in place for inspection")


if __name__ == "__main__":
    main()
