#!/usr/bin/env python3
"""Download audio event annotations for site 257 recordings via authenticated A2O API.

Logs in with POST /security, then fetches each recording's annotations from
GET /audio_recordings/:id/audio_events/download.csv?startOffset=0&endOffset=<duration>

Run from repository root, for example:
  python3 script/download_site_257_annotations.py --start-item 1 --end-item 100
  python3 script/download_site_257_annotations.py --start-item 1 --end-item 100 --workers 4
  python3 script/download_site_257_annotations.py --recording-id 4951 4952

Existing annotation files are skipped by default. Use --force to re-download.

Secrets are intentionally embedded for this local workflow only; do not publish or share
the repository if you keep credentials in this file.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests

API_BASE = "https://api.acousticobservatory.org"
SECURITY_URL = f"{API_BASE}/security"
ANNOTATIONS_URL = (
    f"{API_BASE}/audio_recordings/{{recording_id}}/audio_events/download.csv"
    "?startOffset=0&endOffset={end_offset}"
)

# Hard-coded portal login (JSON field is "email" per API docs; username-style logins work).
A2O_LOGIN_EMAIL = "liting"
A2O_PASSWORD = "88888888"

REQUEST_TIMEOUT = (60, 120)
MAX_ATTEMPTS = 5
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
        project_root / "resources" / "site_257_bowra-dry-a" / "downloaded_annotations"
    )

    parser = argparse.ArgumentParser(
        description="Download annotation CSVs for site 257 recordings (authenticated)."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv,
        help="CSV exported from the audio_recordings filter (must include count, id, duration_seconds).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to write downloaded annotation CSV files.",
    )
    parser.add_argument(
        "--start-item",
        type=int,
        default=None,
        help="First CSV row count (inclusive). Required with --end-item unless --recording-id is set.",
    )
    parser.add_argument(
        "--end-item",
        type=int,
        default=None,
        help="Last CSV row count (inclusive). Required with --start-item unless --recording-id is set.",
    )
    parser.add_argument(
        "--recording-id",
        type=int,
        nargs="*",
        default=(),
        help="If set, download annotations for these recording IDs only (ignores CSV range).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the annotation file already exists (default is to skip).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without contacting the API.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel worker processes (default: 10).",
    )
    return parser.parse_args()


def iter_csv_rows_by_count(
    csv_path: Path, start_item: int, end_item: int
) -> Iterable[Tuple[int, dict]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "count" not in (reader.fieldnames or []):
            raise ValueError("CSV must include a 'count' column")
        for row in reader:
            count_raw = (row.get("count") or "").strip()
            if not count_raw:
                raise ValueError("CSV row has missing count value")
            count = int(count_raw)
            if count < start_item:
                continue
            if count > end_item:
                break
            yield count, row


def login(session: requests.Session) -> str:
    response = session.post(
        SECURITY_URL,
        json={"user": {"email": A2O_LOGIN_EMAIL, "password": A2O_PASSWORD}},
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://data.acousticobservatory.org",
            "Referer": "https://data.acousticobservatory.org/",
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()
    meta = body.get("meta") or {}
    if meta.get("status") not in (200, "200", None):
        raise RuntimeError(f"Login failed: {body}")
    data = body.get("data") or {}
    token = data.get("auth_token")
    if not token:
        raise RuntimeError(f"Login response missing auth_token: {body}")
    return str(token)


def annotation_output_path(output_dir: Path, recording_id: str) -> Path:
    return output_dir / f"annotations_{recording_id}.csv"


def download_annotations(
    session: requests.Session,
    auth_header: str,
    recording_id: str,
    duration_seconds: float,
    output_dir: Path,
    skip_existing: bool,
) -> Tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = annotation_output_path(output_dir, recording_id)

    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return True, f"[SKIP] exists {dest.name}"

    url = ANNOTATIONS_URL.format(
        recording_id=recording_id,
        end_offset=f"{duration_seconds:.3f}",
    )
    headers = {
        "Authorization": auth_header,
        "Accept": "text/csv",
    }

    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        tmp = dest.with_suffix(".csv.part")
        try:
            response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                return False, f"HTTP 404 for recording {recording_id}"
            response.raise_for_status()

            tmp.write_bytes(response.content)
            if tmp.stat().st_size == 0:
                tmp.unlink(missing_ok=True)
                raise ValueError("downloaded annotation file is empty")

            tmp.replace(dest)
            return True, f"[OK] {dest.name} ({dest.stat().st_size} bytes, tries={attempt})"

        except Exception as exc:
            last_error = str(exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    return False, f"[FAIL] recording {recording_id} after {MAX_ATTEMPTS} tries: {last_error}"


def download_annotations_worker(
    job: Tuple[int, str, float, str, str, bool],
) -> Tuple[bool, int, str, str]:
    """Picklable worker for ProcessPoolExecutor."""
    item_count, recording_id, duration_seconds, output_dir_str, auth_header, skip_existing = job
    with requests.Session() as session:
        ok, message = download_annotations(
            session,
            auth_header,
            recording_id,
            duration_seconds,
            Path(output_dir_str),
            skip_existing,
        )
    label = f"count {item_count} item {recording_id}: " if item_count >= 0 else f"item {recording_id}: "
    for tag in ("[OK] ", "[SKIP] ", "[FAIL] "):
        if message.startswith(tag):
            message = tag + label + message[len(tag):]
            break
    return ok, item_count, recording_id, message


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    if not args.recording_id:
        if args.start_item is None or args.end_item is None:
            raise ValueError("CSV mode requires --start-item and --end-item")
        if args.start_item < 1:
            raise ValueError("--start-item must be >= 1")
        if args.end_item < args.start_item:
            raise ValueError("--end-item must be >= --start-item")

    # Build job list: (item_count, recording_id, duration_seconds)
    # item_count is -1 when sourced from --recording-id (not CSV)
    if args.recording_id:
        # No duration available from CLI — use a large sentinel so the full recording is covered.
        # The API will simply return all events up to the actual duration.
        jobs: List[Tuple[int, str, float]] = [
            (-1, str(rid), 86400.0) for rid in sorted(set(args.recording_id))
        ]
    else:
        if not args.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {args.csv_path}")
        jobs = []
        for count, row in iter_csv_rows_by_count(
            args.csv_path, args.start_item, args.end_item
        ):
            rid = (row.get("id") or "").strip()
            duration_raw = (row.get("duration_seconds") or "").strip()
            if not rid:
                print(f"[SKIP] count {count}: missing id")
                continue
            if not duration_raw:
                print(f"[SKIP] count {count} item {rid}: missing duration_seconds")
                continue
            try:
                duration = float(duration_raw)
            except ValueError:
                print(f"[SKIP] count {count} item {rid}: invalid duration_seconds={duration_raw!r}")
                continue
            jobs.append((count, rid, duration))

    if not jobs:
        print("[DONE] No recordings scheduled.")
        return

    if args.dry_run:
        print(
            f"[DRY-RUN] Would download {len(jobs)} annotation file(s) to {args.output_dir} "
            f"with {args.workers} workers"
        )
        for count, rid, duration in jobs:
            label = f"count {count} " if count >= 0 else ""
            print(f"  {label}id {rid} duration={duration:.3f}s")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    skip_existing = not args.force
    pending: List[Tuple[int, str, float]] = []
    preflight_skips: List[str] = []

    for count, rid, duration in jobs:
        if skip_existing:
            dest = annotation_output_path(args.output_dir, rid)
            if dest.exists() and dest.stat().st_size > 0:
                label = f"count {count} " if count >= 0 else ""
                preflight_skips.append(f"[SKIP] {label}item {rid}: exists on disk {dest.name}")
                continue
        pending.append((count, rid, duration))

    for line in preflight_skips:
        print(line)

    if not pending:
        print(
            f"[DONE] completed downloads: success={len(preflight_skips)} failed=0 "
            f"total={len(jobs)} (all skipped, existing files)"
        )
        print("[REPORT] No failed items.")
        return

    with requests.Session() as session:
        token = login(session)
        auth_header = f'Token token="{token}"'
        print(f"[AUTH] Logged in as {A2O_LOGIN_EMAIL!r}.")

    out_dir_str = str(args.output_dir.resolve())
    pool_jobs = [
        (count, rid, duration, out_dir_str, auth_header, skip_existing)
        for count, rid, duration in pending
    ]

    print(
        f"[START] Downloading {len(pool_jobs)} annotation file(s) with {args.workers} workers "
        f"({len(preflight_skips)} already on disk skipped)..."
    )

    success_count = len(preflight_skips)
    failure_count = 0
    failed_items = []
    worker_exception_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_annotations_worker, job) for job in pool_jobs]
        for future in as_completed(futures):
            try:
                ok, item_count, recording_id, message = future.result()
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
                failed_items.append((item_count, recording_id))

    print(
        f"[DONE] completed downloads: success={success_count} failed={failure_count} "
        f"total={len(jobs)}"
    )

    if failed_items or worker_exception_count:
        print("[REPORT] Failed items:")
        for item_count, recording_id in sorted(
            failed_items, key=lambda t: (t[0] if t[0] >= 0 else 10**12, int(t[1]))
        ):
            if item_count < 0:
                print(f"[REPORT] item {recording_id}: annotation download failed (--recording-id)")
            else:
                print(f"[REPORT] count {item_count} item {recording_id}: annotation download failed")
        if worker_exception_count:
            print(
                f"[REPORT] worker exceptions={worker_exception_count} "
                "(recording details unavailable for these failures)"
            )
    else:
        print("[REPORT] No failed items.")


if __name__ == "__main__":
    main()
