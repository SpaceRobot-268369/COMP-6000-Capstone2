#!/usr/bin/env python3
"""Download original FLAC files for site 257 via authenticated A2O API.

Logs in with POST /security (see baw-server API auth), then streams each file from
GET /audio_recordings/:id/original.

Run from repository root, for example:
  python3 script/download_site_257_originals.py --start-item 1 --end-item 5
  python3 script/download_site_257_originals.py --start-item 1 --end-item 10 --workers 4
  python3 script/download_site_257_originals.py --recording-id 4951 4952 --workers 2

Existing originals are skipped by default (CSV canonical_file_name or *_<id>.flac on disk).
Use --force to download again.

Secrets are intentionally embedded for this local workflow only; do not publish or share
the repository if you keep credentials in this file.
"""
# https://github.com/QutEcoacoustics/baw-server/wiki/API:-Authentication

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests

API_BASE = "https://api.acousticobservatory.org"
SECURITY_URL = f"{API_BASE}/security"
ORIGINAL_URL = f"{API_BASE}/audio_recordings/{{recording_id}}/original"

# Hard-coded portal login (JSON field is "email" per API docs; username-style logins work).
A2O_LOGIN_EMAIL = "Murphyy"
A2O_PASSWORD = "12345678"

REQUEST_TIMEOUT = (60, 600)
CONNECT_READ_TIMEOUT = REQUEST_TIMEOUT
CHUNK_BYTES = 1024 * 1024
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
        project_root / "resources" / "site_257_bowra-dry-a" / "downloaded_originals"
    )

    parser = argparse.ArgumentParser(
        description="Download original audio files for site 257 (authenticated)."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv,
        help="CSV exported from the audio_recordings filter (must include count, id).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to write downloaded originals.",
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
        help="If set, download these recording IDs only (ignores CSV range).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the output file already exists (default is to skip).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without contacting the download URL.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel worker processes for downloads (default: 10).",
    )
    return parser.parse_args()


def iter_csv_rows_by_count(
    csv_path: Path, start_item: int, end_item: int
) -> Iterable[tuple[int, dict[str, str]]]:
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
        timeout=CONNECT_READ_TIMEOUT,
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


def existing_original_path(
    output_dir: Path, recording_id: str, canonical_file_name: Optional[str]
) -> Optional[Path]:
    """Return path if a non-empty original for this recording is already on disk."""
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        return None
    if canonical_file_name:
        candidate = output_dir / canonical_file_name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    for pattern in (f"*_{recording_id}.flac", f"*_{recording_id}.wav"):
        matches = [
            p for p in output_dir.glob(pattern) if p.is_file() and p.stat().st_size > 0
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def filename_from_content_disposition(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    # Prefer RFC 5987 filename* when present
    m_star = re.search(r"filename\*\s*=\s*([^']*)''([^;\n]+)", value, re.IGNORECASE)
    if m_star:
        from urllib.parse import unquote

        return unquote(m_star.group(2).strip())
    m = re.search(r'filename\s*=\s*"([^"]+)"', value, re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r"filename\s*=\s*([^;\s]+)", value, re.IGNORECASE)
    if m2:
        return m2.group(1).strip().strip('"')
    return None


def download_original(
    session: requests.Session,
    auth_header: str,
    recording_id: str,
    output_dir: Path,
    skip_existing: bool,
) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    url = ORIGINAL_URL.format(recording_id=recording_id)
    headers = {
        "Authorization": auth_header,
        "Accept": "*/*",
    }

    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        tmp: Optional[Path] = None
        try:
            with session.get(
                url, headers=headers, stream=True, timeout=CONNECT_READ_TIMEOUT
            ) as response:
                if response.status_code == 404:
                    return False, f"HTTP 404 for recording {recording_id}"
                response.raise_for_status()
                cd = response.headers.get("Content-Disposition")
                name = filename_from_content_disposition(cd) or f"recording_{recording_id}.bin"
                dest = output_dir / name
                if skip_existing and dest.exists() and dest.stat().st_size > 0:
                    return True, f"[SKIP] exists {dest.name}"

                tmp = dest.with_suffix(dest.suffix + ".part")
                expected = response.headers.get("Content-Length")
                expected_int = int(expected) if expected and expected.isdigit() else None

                with tmp.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        if chunk:
                            out.write(chunk)

                if expected_int is not None and tmp.stat().st_size != expected_int:
                    tmp.unlink(missing_ok=True)
                    raise ValueError(
                        f"size mismatch: got {tmp.stat().st_size} expected {expected_int}"
                    )
                if tmp.stat().st_size == 0:
                    tmp.unlink(missing_ok=True)
                    raise ValueError("downloaded file is empty")

                tmp.replace(dest)
                return True, f"[OK] {dest.name} ({dest.stat().st_size} bytes, tries={attempt})"

        except Exception as exc:
            last_error = str(exc)
            if tmp is not None and tmp.exists():
                tmp.unlink(missing_ok=True)

    return False, f"[FAIL] recording {recording_id} after {MAX_ATTEMPTS} tries: {last_error}"


def download_original_worker(
    job: Tuple[int, str, str, str, bool],
) -> Tuple[bool, int, str, str]:
    """Picklable worker for ProcessPoolExecutor. item_count is -1 when not from CSV."""
    item_count, recording_id, output_dir_str, auth_header, skip_existing = job
    label_count = item_count if item_count >= 0 else None
    with requests.Session() as session:
        ok, message = download_original(
            session,
            auth_header,
            recording_id,
            Path(output_dir_str),
            skip_existing,
        )
    if label_count is not None:
        for tag in ("[OK] ", "[SKIP] ", "[FAIL] "):
            if message.startswith(tag):
                message = (
                    tag
                    + f"count {label_count} item {recording_id}: "
                    + message[len(tag) :]
                )
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

    # (csv count or None, recording id, canonical_file_name from CSV or None)
    if args.recording_id:
        jobs: List[Tuple[Optional[int], str, Optional[str]]] = [
            (None, str(rid), None) for rid in sorted(set(args.recording_id))
        ]
    else:
        if not args.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {args.csv_path}")
        jobs = []
        for count, row in iter_csv_rows_by_count(
            args.csv_path, args.start_item, args.end_item
        ):
            rid = (row.get("id") or "").strip()
            if not rid:
                print(f"[SKIP] count {count}: missing id")
                continue
            canonical = (row.get("canonical_file_name") or "").strip() or None
            jobs.append((count, rid, canonical))

    if not jobs:
        print("[DONE] No recordings scheduled.")
        return

    if args.dry_run:
        print(
            f"[DRY-RUN] Would download {len(jobs)} original(s) to {args.output_dir} "
            f"with {args.workers} workers"
        )
        for count, rid, _canonical in jobs:
            label = f"count {count} " if count is not None else ""
            print(f"  {label}id {rid}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    skip_existing = not args.force
    pending_downloads: List[Tuple[int, str]] = []
    preflight_skip_messages: List[str] = []

    for count, rid, canonical in jobs:
        if skip_existing:
            existing = existing_original_path(args.output_dir, rid, canonical)
            if existing is not None:
                if count is not None:
                    preflight_skip_messages.append(
                        f"[SKIP] count {count} item {rid}: exists on disk {existing.name}"
                    )
                else:
                    preflight_skip_messages.append(
                        f"[SKIP] item {rid}: exists on disk {existing.name} (--recording-id)"
                    )
                continue
        item_count = -1 if count is None else count
        pending_downloads.append((item_count, rid))

    for line in preflight_skip_messages:
        print(line)

    if not pending_downloads:
        print(
            f"[DONE] completed downloads: success={len(preflight_skip_messages)} failed=0 "
            f"total={len(jobs)} (all skipped, existing files)"
        )
        print("[REPORT] No failed items.")
        return

    with requests.Session() as session:
        token = login(session)
        auth_header = f'Token token="{token}"'
        print(f"[AUTH] Logged in as {A2O_LOGIN_EMAIL!r}.")

    out_dir_str = str(args.output_dir.resolve())
    pool_jobs: List[Tuple[int, str, str, str, bool]] = [
        (item_count, rid, out_dir_str, auth_header, skip_existing)
        for item_count, rid in pending_downloads
    ]

    print(
        f"[START] Downloading {len(pool_jobs)} originals with {args.workers} workers "
        f"({len(preflight_skip_messages)} already on disk skipped)..."
    )

    success_count = len(preflight_skip_messages)
    failure_count = 0
    failed_items: set[Tuple[int, str]] = set()
    worker_exception_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_original_worker, job) for job in pool_jobs]
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
                failed_items.add((item_count, recording_id))

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
                print(f"[REPORT] item {recording_id}: original download failed (--recording-id)")
            else:
                print(
                    f"[REPORT] count {item_count} item {recording_id}: original download failed"
                )
        if worker_exception_count:
            print(
                f"[REPORT] worker exceptions={worker_exception_count} "
                "(recording details unavailable for these failures)"
            )
    else:
        print("[REPORT] No failed items.")


if __name__ == "__main__":
    main()
