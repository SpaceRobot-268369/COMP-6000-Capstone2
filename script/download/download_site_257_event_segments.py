#!/usr/bin/env python3
"""Download per-event audio segments (with buffer) for site 257 annotations.

Reads the annotation CSVs under
``resources/site_257_bowra-dry-a/all_items_annotation/site_257_item_<rid>/``,
computes a buffered window around each event
(``event_start_seconds - BUFFER`` to ``event_end_seconds + BUFFER``, clamped to
the recording duration), and downloads that segment from the A2O media API.

Output layout (one file per event) — segments live alongside their annotation CSV:

    all_items_annotation/
      site_257_item_<rid>/
        site_257_item_<rid>.csv                        (existing annotation CSV)
        site_257_item_<rid>_audioevent_<event_id>/
          site_257_item_<rid>_audioevent_<event_id>.webm

Run from repository root, e.g.::

    python3 script/download/download_site_257_event_segments.py \\
        --start-item 1 --end-item 100 --workers 10
    python3 script/download/download_site_257_event_segments.py \\
        --event-manifest resources/site_257_bowra-dry-a/layer_c_smoke_test/manifest.csv \\
        --output-dir resources/site_257_bowra-dry-a/layer_c_smoke_test/segments \\
        --dry-run

Optional filters (applied per event before scheduling)::

    --min-score 0.7
    --min-duration 1.0
    --max-duration 10.0
    --event-manifest resources/site_257_bowra-dry-a/layer_c_smoke_test/manifest.csv
"""

# Run from repository root:
# python3 script/download/download_site_257_event_segments.py --start-item 1 --end-item 100
# Optional concurrency override:
# python3 script/download/download_site_257_event_segments.py --start-item 1 --end-item 100 --workers 10

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import requests

BASE_MEDIA_URL = "https://api.acousticobservatory.org/audio_recordings/{recording_id}/media.webm"
BUFFER_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 120
MAX_DOWNLOAD_ATTEMPTS = 10
DEFAULT_WORKERS = 10


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    default_csv = (
        project_root
        / "resources"
        / "site_257_bowra-dry-a"
        / "site_257_all_items.csv"
    )
    default_anno = (
        project_root
        / "resources"
        / "site_257_bowra-dry-a"
        / "all_items_annotation"
    )
    # Segments are written next to their annotation CSV — same folder.
    default_output = default_anno

    parser = argparse.ArgumentParser(
        description=(
            "Download buffered audio segments for each annotated event. "
            "Each event clip is event_duration + 2*BUFFER seconds long."
        )
    )
    parser.add_argument(
        "--items-csv",
        type=Path,
        default=default_csv,
        help="Path to site_257_all_items.csv (used to look up recording durations).",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=default_anno,
        help="Directory containing site_257_item_<rid>/ annotation subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=(
            "Directory where per-item / per-event folders are written. "
            "Defaults to the annotations directory so segments sit next to their CSVs."
        ),
    )
    parser.add_argument(
        "--start-item",
        type=int,
        default=None,
        help="Start item count in items-csv (inclusive). Filters which recordings are processed.",
    )
    parser.add_argument(
        "--end-item",
        type=int,
        default=None,
        help="End item count in items-csv (inclusive).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel worker processes for segment downloads (default: 10).",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=BUFFER_SECONDS,
        help=f"Seconds of context to add on each side of the event (default: {BUFFER_SECONDS}).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional: drop events with BirdNET score below this value.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Optional: drop events shorter than this (seconds, pre-buffer).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional: drop events longer than this (seconds, pre-buffer).",
    )
    parser.add_argument(
        "--event-manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing an audio_event_id column. When provided, "
            "only those exact events are scheduled. Useful for Layer C smoke tests."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the download plan without fetching media.",
    )
    return parser.parse_args()


def iter_selected_rows(
    csv_path: Path, start_item: int, end_item: int
) -> Iterable[tuple[int, dict[str, str]]]:
    """Yield (count, row) from items-csv for counts in [start_item, end_item]."""
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


def read_annotation_events(anno_csv: Path) -> list[dict[str, str]]:
    """Read all event rows from one annotation CSV. Returns [] if file is empty."""
    if not anno_csv.exists() or anno_csv.stat().st_size == 0:
        return []
    with anno_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_event_id_filter(manifest_path: Path | None) -> set[str] | None:
    """Load exact event IDs from a smoke/test manifest, if provided."""
    if manifest_path is None:
        return None
    if not manifest_path.exists():
        raise FileNotFoundError(f"event manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "audio_event_id" not in (reader.fieldnames or []):
            raise ValueError("event manifest must include an 'audio_event_id' column")
        event_ids = {
            (row.get("audio_event_id") or "").strip()
            for row in reader
            if (row.get("audio_event_id") or "").strip()
        }

    if not event_ids:
        raise ValueError(f"event manifest contains no audio_event_id values: {manifest_path}")
    return event_ids


def buffered_window(
    event_start: float,
    event_end: float,
    recording_duration: float,
    buffer_seconds: float,
) -> tuple[float, float]:
    """Apply ±buffer and clamp to [0, recording_duration]."""
    start = max(0.0, event_start - buffer_seconds)
    end = min(recording_duration, event_end + buffer_seconds)
    return start, end


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


def download_job(
    job: tuple[int, str, str, float, float, str, str],
) -> tuple[bool, int, str, str, str]:
    item_count, item_id, event_id, start_offset, end_offset, segment_name, output_path_str = job
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
            event_id,
            (
                f"[OK] count {item_count} item {item_id} event {event_id}: "
                f"{segment_name} (start={start_offset:.3f}, end={end_offset:.3f}, tries={attempts})"
            ),
        )

    return (
        False,
        item_count,
        item_id,
        event_id,
        (
            f"[FAIL] count {item_count} item {item_id} event {event_id}: "
            f"{segment_name} (start={start_offset:.3f}, end={end_offset:.3f}, tries={attempts}) error={err}"
        ),
    )


def event_passes_filters(
    event_duration: float,
    score: float | None,
    args: argparse.Namespace,
) -> bool:
    if args.min_score is not None and (score is None or score < args.min_score):
        return False
    if args.min_duration is not None and event_duration < args.min_duration:
        return False
    if args.max_duration is not None and event_duration > args.max_duration:
        return False
    return True


def main() -> None:
    args = parse_args()

    if args.start_item is None and args.end_item is None:
        args.start_item = 1
        args.end_item = 10**12
    elif args.start_item is None or args.end_item is None:
        raise ValueError("--start-item and --end-item must be provided together")
    elif args.start_item < 1:
        raise ValueError("--start-item must be >= 1")
    elif args.end_item < args.start_item:
        raise ValueError("--end-item must be >= --start-item")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.buffer_seconds < 0:
        raise ValueError("--buffer-seconds must be >= 0")
    if not args.items_csv.exists():
        raise FileNotFoundError(f"items CSV not found: {args.items_csv}")
    if not args.annotations_dir.exists():
        raise FileNotFoundError(f"annotations dir not found: {args.annotations_dir}")

    event_id_filter = load_event_id_filter(args.event_manifest)

    jobs: list[tuple[int, str, str, float, float, str, str]] = []
    skipped_existing = 0
    skipped_filtered = 0
    skipped_invalid = 0
    items_with_no_annotations = 0

    for item_count, row in iter_selected_rows(args.items_csv, args.start_item, args.end_item):
        item_id = (row.get("id") or "").strip()
        duration_raw = (row.get("duration_seconds") or "").strip()

        if not item_id:
            print(f"[SKIP] count {item_count}: missing id")
            continue
        if not duration_raw:
            print(f"[SKIP] count {item_count} item {item_id}: missing duration_seconds")
            continue
        try:
            recording_duration = float(duration_raw)
        except ValueError:
            print(f"[SKIP] count {item_count} item {item_id}: invalid duration_seconds={duration_raw!r}")
            continue
        if recording_duration <= 0:
            print(f"[SKIP] count {item_count} item {item_id}: non-positive duration")
            continue

        anno_subdir = args.annotations_dir / f"site_257_item_{item_id}"
        anno_csv = anno_subdir / f"site_257_item_{item_id}.csv"
        if not anno_csv.exists():
            flat_anno_csv = args.annotations_dir / f"annotations_{item_id}.csv"
            if flat_anno_csv.exists():
                anno_csv = flat_anno_csv
        events = read_annotation_events(anno_csv)

        if not events:
            items_with_no_annotations += 1
            continue

        scheduled_for_item = 0
        for ev in events:
            event_id = (ev.get("audio_event_id") or "").strip()
            if not event_id:
                skipped_invalid += 1
                continue
            if event_id_filter is not None and event_id not in event_id_filter:
                skipped_filtered += 1
                continue

            try:
                event_start = float(ev.get("event_start_seconds") or "")
                event_end = float(ev.get("event_end_seconds") or "")
            except ValueError:
                skipped_invalid += 1
                continue
            if event_end <= event_start:
                skipped_invalid += 1
                continue

            event_duration = event_end - event_start
            score_raw = (ev.get("score") or "").strip()
            try:
                score = float(score_raw) if score_raw else None
            except ValueError:
                score = None

            if not event_passes_filters(event_duration, score, args):
                skipped_filtered += 1
                continue

            start_offset, end_offset = buffered_window(
                event_start, event_end, recording_duration, args.buffer_seconds
            )
            if end_offset <= start_offset:
                skipped_invalid += 1
                continue

            item_folder = args.output_dir / f"site_257_item_{item_id}"
            segment_name = f"site_257_item_{item_id}_audioevent_{event_id}.webm"
            event_folder = item_folder / f"site_257_item_{item_id}_audioevent_{event_id}"
            segment_path = event_folder / segment_name

            if segment_path.exists() and segment_path.stat().st_size > 0:
                skipped_existing += 1
                continue

            jobs.append(
                (
                    item_count,
                    item_id,
                    event_id,
                    start_offset,
                    end_offset,
                    segment_name,
                    str(segment_path),
                )
            )
            scheduled_for_item += 1

        if scheduled_for_item:
            print(
                f"[ITEM] count {item_count} item {item_id}: events_scheduled={scheduled_for_item} "
                f"(total_in_csv={len(events)}, recording_duration={recording_duration:.1f}s)"
            )

    print(
        f"[PLAN] scheduled={len(jobs)} "
        f"skipped_existing={skipped_existing} "
        f"skipped_filtered={skipped_filtered} "
        f"skipped_invalid={skipped_invalid} "
        f"items_with_no_annotations={items_with_no_annotations}"
    )

    if not jobs:
        print("[DONE] No segments scheduled for download.")
        return

    if args.dry_run:
        print("[DRY-RUN] Planned event downloads:")
        for item_count, item_id, event_id, start_offset, end_offset, segment_name, output_path in jobs:
            print(
                f"[DRY-RUN] count {item_count} item {item_id} event {event_id}: "
                f"{segment_name} start={start_offset:.3f} end={end_offset:.3f} output={output_path}"
            )
        print("[DRY-RUN] No media downloaded.")
        return

    print(f"[START] Downloading {len(jobs)} event segments with {args.workers} workers...")

    success_count = 0
    failure_count = 0
    failed_events: dict[tuple[int, str], set[str]] = defaultdict(set)
    worker_exception_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_job, job) for job in jobs]
        for future in as_completed(futures):
            try:
                ok, item_count, item_id, event_id, message = future.result()
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
                failed_events[(item_count, item_id)].add(event_id)

    print(
        f"[DONE] completed downloads: success={success_count} failed={failure_count} total={len(jobs)}"
    )

    if failed_events or worker_exception_count:
        print("[REPORT] Failed events:")
        for item_count, item_id in sorted(failed_events):
            event_ids = sorted(failed_events[(item_count, item_id)])
            event_ids_text = ", ".join(event_ids)
            print(
                f"[REPORT] count {item_count} item {item_id}: "
                f"failed_events={len(event_ids)} ({event_ids_text})"
            )
        if worker_exception_count:
            print(
                f"[REPORT] worker exceptions={worker_exception_count} "
                "(event details unavailable for these failures)"
            )
    else:
        print("[REPORT] No failed events.")


if __name__ == "__main__":
    main()
