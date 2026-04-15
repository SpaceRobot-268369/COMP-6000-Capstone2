#!/usr/bin/env python3
"""Download site 257 audio clips from CSV with chunking and retries."""
# Run from repository root:
# python3 script/download_site_257_clips.py --start-item 1 --end-item 200
# Optional concurrency override:
# python3 script/download_site_257_clips.py --start-item 1 --end-item 200 --workers 10

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import requests

BASE_MEDIA_URL = "https://api.acousticobservatory.org/audio_recordings/{recording_id}/media.webm"
MAX_CLIP_SECONDS = 300.0
REQUEST_TIMEOUT_SECONDS = 120
MAX_DOWNLOAD_ATTEMPTS = 10
DEFAULT_WORKERS = 10


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
        help="Start item count in the CSV (inclusive).",
    )
    parser.add_argument(
        "--end-item",
        type=int,
        required=True,
        help="End item count in the CSV (inclusive).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel worker processes for clip downloads (default: 10).",
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


def iter_selected_rows(
    csv_path: Path, start_item: int, end_item: int
) -> Iterable[tuple[int, dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if "count" not in (reader.fieldnames or []):
            raise ValueError("CSV must include a 'count' column")

        for row in reader:
            count_raw = (row.get("count") or "").strip()
            if not count_raw:
                raise ValueError("CSV row has missing count value")
            try:
                count = int(count_raw)
            except ValueError as exc:
                raise ValueError(f"CSV row has invalid count value: {count_raw!r}") from exc

            if count < start_item:
                continue
            if count > end_item:
                break
            yield count, row


def download_segment(
    recording_id: str,
    start_offset: float,
    end_offset: float,
    output_path: Path,
) -> tuple[bool, int, str]:
    url = BASE_MEDIA_URL.format(recording_id=recording_id)
    params = {"start_offset": f"{start_offset:.3f}", "end_offset": f"{end_offset:.3f}"}
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    last_error = ""

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()

            if not response.content:
                raise ValueError("response content is empty")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(response.content)
            if tmp_path.stat().st_size == 0:
                raise ValueError("written file is zero-byte")

            tmp_path.replace(output_path)
            if output_path.stat().st_size == 0:
                raise ValueError("final file is zero-byte")

            return True, attempt, ""
        except Exception as exc:
            last_error = str(exc)
            if tmp_path.exists():
                tmp_path.unlink()
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink()

    return False, MAX_DOWNLOAD_ATTEMPTS, last_error


def download_job(job: tuple[int, str, int, float, float, str, str]) -> tuple[bool, int, str, int, str]:
    item_count, item_id, clip_num, start_offset, end_offset, clip_name, output_path_str = job
    output_path = Path(output_path_str)
    ok, attempts, err = download_segment(
        recording_id=item_id,
        start_offset=start_offset,
        end_offset=end_offset,
        output_path=output_path,
    )

    if ok:
        return (
            True,
            item_count,
            item_id,
            clip_num,
            (
                f"[OK] count {item_count} item {item_id} clip {clip_num:03d}: "
                f"{clip_name} (start={start_offset:.3f}, end={end_offset:.3f}, tries={attempts})"
            ),
        )

    return (
        False,
        item_count,
        item_id,
        clip_num,
        (
            f"[FAIL] count {item_count} item {item_id} clip {clip_num:03d}: "
            f"{clip_name} (start={start_offset:.3f}, end={end_offset:.3f}, tries={attempts}) error={err}"
        ),
    )


def main() -> None:
    args = parse_args()

    if args.start_item < 1:
        raise ValueError("--start-item must be >= 1")
    if args.end_item < args.start_item:
        raise ValueError("--end-item must be >= --start-item")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[int, str, int, float, float, str, str]] = []
    for item_count, row in iter_selected_rows(args.csv_path, args.start_item, args.end_item):
        item_id = (row.get("id") or "").strip()
        duration_raw = (row.get("duration_seconds") or "").strip()

        if not item_id:
            print(f"[SKIP] count {item_count}: missing id")
            continue
        if not duration_raw:
            print(f"[SKIP] count {item_count} item {item_id}: missing duration_seconds")
            continue

        try:
            duration_seconds = float(duration_raw)
        except ValueError:
            print(
                f"[SKIP] count {item_count} item {item_id}: invalid duration_seconds={duration_raw!r}"
            )
            continue

        item_folder = args.output_dir / f"site_257_item_{item_id}"
        if item_folder.exists():
            print(f"[SKIP] count {item_count} item {item_id}: folder exists ({item_folder})")
            continue

        segments = build_segments(duration_seconds, MAX_CLIP_SECONDS)
        if not segments:
            print(f"[SKIP] count {item_count} item {item_id}: non-positive duration")
            continue

        item_folder.mkdir(parents=True, exist_ok=False)
        print(
            f"[ITEM] count {item_count} item {item_id}: duration={duration_seconds:.3f}s clips={len(segments)}"
        )

        for clip_num, (start_offset, end_offset) in enumerate(segments, start=1):
            clip_name = f"site_257_item_{item_id}_clip_{clip_num:03d}.webm"
            clip_path = item_folder / clip_name
            jobs.append(
                (
                    item_count,
                    item_id,
                    clip_num,
                    start_offset,
                    end_offset,
                    clip_name,
                    str(clip_path),
                )
            )

    if not jobs:
        print("[DONE] No clips scheduled for download.")
        return

    print(f"[START] Downloading {len(jobs)} clips with {args.workers} workers...")

    success_count = 0
    failure_count = 0
    failed_items: dict[tuple[int, str], set[int]] = defaultdict(set)
    worker_exception_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_job, job) for job in jobs]
        for future in as_completed(futures):
            try:
                ok, item_count, item_id, clip_num, message = future.result()
            except Exception as exc:
                failure_count += 1
                worker_exception_count += 1
                print(f"[FAIL] worker exception: {exc}")
                continue

            print(message)
            if ok:
                success_count += 1
            else:
                failure_count += 1
                failed_items[(item_count, item_id)].add(clip_num)

    print(
        f"[DONE] completed downloads: success={success_count} failed={failure_count} total={len(jobs)}"
    )

    if failed_items or worker_exception_count:
        print("[REPORT] Failed items:")
        for item_count, item_id in sorted(failed_items):
            clip_nums = sorted(failed_items[(item_count, item_id)])
            clip_nums_text = ", ".join(f"{clip_num:03d}" for clip_num in clip_nums)
            print(
                f"[REPORT] count {item_count} item {item_id}: "
                f"failed_clips={len(clip_nums)} ({clip_nums_text})"
            )
        if worker_exception_count:
            print(
                f"[REPORT] worker exceptions={worker_exception_count} "
                "(clip details unavailable for these failures)"
            )
    else:
        print("[REPORT] No failed items.")


if __name__ == "__main__":
    main()
