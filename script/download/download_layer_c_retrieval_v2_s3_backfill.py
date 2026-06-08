#!/usr/bin/env python3
"""Backfill Layer C retrieval v2 review sources from S3 300-second clips.

The shared S3 store keeps site_257 recordings as 300-second webm chunks:

  s3://eco-acoustic-data.store.adelaideuni.cloud/
    dataset/original/site_257_bowra-dry-a/downloaded_clips/
      site_257_item_<recording_id>/
        site_257_item_<recording_id>_clip_<NNN>.webm

This script reads a v2 backfill manifest with exact event windows, downloads
the needed source chunks to a local cache, extracts the buffered event window
with ffmpeg, and writes:

  downloaded_source.webm
  downloaded_source_origin.json

Run the review-package builder after this script to regenerate WAVs and
spectrograms from the S3-backed sources.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_ROOT = (
    REPO_ROOT
    / "acoustic_ai"
    / "layers"
    / "layer_c"
    / "attempts"
    / "burger__mvp_2__retrieval_v2_library"
    / "data"
    / "media_asset_bank"
)
DEFAULT_MANIFEST = LIB_ROOT / "review_package_pilot_v2" / "pilot_backfill_event_manifest_v2.csv"
DEFAULT_CACHE_DIR = Path("/private/tmp/layer_c_retrieval_v2_s3_cache")
S3_CLIP_PREFIX = (
    "s3://eco-acoustic-data.store.adelaideuni.cloud/"
    "dataset/original/site_257_bowra-dry-a/downloaded_clips"
)
CLIP_SECONDS = 300.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{detail}")


def clip_numbers(start_s: float, end_s: float) -> list[int]:
    if end_s <= start_s:
        raise ValueError(f"invalid window: {start_s}-{end_s}")
    first = int(math.floor(start_s / CLIP_SECONDS)) + 1
    last = int(math.floor((end_s - 1e-6) / CLIP_SECONDS)) + 1
    return list(range(first, last + 1))


def s3_uri(prefix: str, recording_id: str, clip_num: int) -> str:
    clip_name = f"site_257_item_{recording_id}_clip_{clip_num:03d}.webm"
    return f"{prefix.rstrip('/')}/site_257_item_{recording_id}/{clip_name}"


def local_clip_path(cache_dir: Path, recording_id: str, clip_num: int) -> Path:
    clip_name = f"site_257_item_{recording_id}_clip_{clip_num:03d}.webm"
    return cache_dir / f"site_257_item_{recording_id}" / clip_name


def ensure_clip(
    *,
    prefix: str,
    cache_dir: Path,
    profile: str,
    recording_id: str,
    clip_num: int,
    dry_run: bool,
) -> Path:
    local_clip = local_clip_path(cache_dir, recording_id, clip_num)
    if local_clip.exists() and local_clip.stat().st_size > 0:
        return local_clip
    local_clip.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "aws",
            "s3",
            "cp",
            s3_uri(prefix, recording_id, clip_num),
            str(local_clip),
            "--profile",
            profile,
        ],
        dry_run=dry_run,
    )
    return local_clip


def extract_window(
    *,
    row: dict[str, str],
    clips: list[Path],
    output_path: Path,
    dry_run: bool,
) -> None:
    start_s = float(row["download_start_s"])
    end_s = float(row["download_end_s"])
    duration_s = max(0.001, end_s - start_s)
    first_clip_index = int(math.floor(start_s / CLIP_SECONDS))
    offset_in_concat = start_s - (first_clip_index * CLIP_SECONDS)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False) as f:
        list_path = Path(f.name)
        for clip in clips:
            f.write(f"file '{clip.resolve()}'\n")
    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-ss",
                f"{offset_in_concat:.3f}",
                "-t",
                f"{duration_s:.3f}",
                "-c",
                "copy",
                str(output_path),
            ],
            dry_run=dry_run,
        )
    finally:
        if list_path.exists():
            list_path.unlink()


def write_origin(row: dict[str, str], output_path: Path, clips: list[Path], prefix: str) -> None:
    payload: dict[str, Any] = {
        "source_kind": "s3_downloaded_clip",
        "clip_prefix": prefix.rstrip("/"),
        "recording_id": row["recording_id"],
        "audio_event_id": row["audio_event_id"],
        "download_start_s": row["download_start_s"],
        "download_end_s": row["download_end_s"],
        "local_clips": [str(p) for p in clips],
        "output_path": str(output_path.relative_to(REPO_ROOT)),
    }
    output_path.with_name("downloaded_source_origin.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--profile", default="capstone2")
    parser.add_argument("--clip-prefix", default=S3_CLIP_PREFIX)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    failures: list[str] = []
    for index, row in enumerate(rows, start=1):
        output_path = REPO_ROOT / row["output_path"]
        if output_path.exists() and output_path.stat().st_size > 0 and not args.force:
            skipped += 1
            continue
        try:
            nums = clip_numbers(float(row["download_start_s"]), float(row["download_end_s"]))
            clips = [
                ensure_clip(
                    prefix=args.clip_prefix,
                    cache_dir=args.cache_dir,
                    profile=args.profile,
                    recording_id=row["recording_id"],
                    clip_num=num,
                    dry_run=args.dry_run,
                )
                for num in nums
            ]
            extract_window(row=row, clips=clips, output_path=output_path, dry_run=args.dry_run)
            if not args.dry_run:
                write_origin(row, output_path, clips, args.clip_prefix)
            ok += 1
            print(
                f"[OK] {index}/{len(rows)} event={row['audio_event_id']} "
                f"clips={','.join(str(n) for n in nums)} output={row['output_path']}"
            )
        except Exception as exc:
            failures.append(f"{row.get('audio_event_id', '<missing>')}: {exc}")
            print(f"[FAIL] {failures[-1]}")

    print(f"[DONE] ok={ok} skipped={skipped} failed={len(failures)} total={len(rows)}")
    if failures:
        for failure in failures[:20]:
            print(f"[FAILURE] {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
