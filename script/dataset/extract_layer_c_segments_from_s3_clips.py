#!/usr/bin/env python3
"""Extract Layer C event segments from S3-hosted 300-second source clips.

The shared bucket stores site 257 audio as 300-second webm chunks under:

  dataset/original/site_257_bowra-dry-a/downloaded_clips/site_257_item_<id>/

Layer C manifests point to short buffered event windows. This script downloads
only the needed source chunks to a temporary cache, trims each event with ffmpeg,
and writes the exact segment layout expected by the Layer C prepare script.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "resources"
    / "site_257_bowra-dry-a"
    / "layer_c_smoke_2_3_species"
    / "manifest.csv"
)
DEFAULT_CACHE_DIR = Path("/private/tmp/layer_c_s3_clip_cache")
S3_CLIP_PREFIX = (
    "s3://eco-acoustic-data.store.adelaideuni.cloud/"
    "dataset/original/site_257_bowra-dry-a/downloaded_clips"
)
CLIP_SECONDS = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Layer C manifest segments from shared S3 clips."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--profile", default="capstone2")
    parser.add_argument("--clip-prefix", default=S3_CLIP_PREFIX)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def download_source_clip(
    row: dict[str, str],
    clip_prefix: str,
    cache_dir: Path,
    profile: str,
    dry_run: bool,
) -> tuple[Path, float]:
    recording_id = row["audio_recording_id"]
    extracted_start = float(row["extracted_start_seconds"])
    extracted_end = float(row["extracted_end_seconds"])
    start_clip_index = int(extracted_start // CLIP_SECONDS)
    end_clip_index = int((extracted_end - 1e-6) // CLIP_SECONDS)
    if start_clip_index != end_clip_index:
        raise ValueError(
            f"event {row['audio_event_id']} crosses source clip boundary; "
            "rebuild manifest with --avoid-clip-boundary"
        )

    clip_num = start_clip_index + 1
    clip_name = f"site_257_item_{recording_id}_clip_{clip_num:03d}.webm"
    source_uri = (
        f"{clip_prefix}/site_257_item_{recording_id}/{clip_name}"
    )
    local_clip = cache_dir / f"site_257_item_{recording_id}" / clip_name
    local_clip.parent.mkdir(parents=True, exist_ok=True)

    if not local_clip.exists() or local_clip.stat().st_size == 0:
        cmd = [
            "aws",
            "s3",
            "cp",
            source_uri,
            str(local_clip),
            "--profile",
            profile,
        ]
        print(f"[S3] {source_uri}")
        if not dry_run:
            run(cmd)

    local_offset = extracted_start - (start_clip_index * CLIP_SECONDS)
    return local_clip, local_offset


def extract_segment(
    row: dict[str, str],
    source_clip: Path,
    local_offset: float,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    output_path = Path(row["segment_path"])
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return output_path

    duration = float(row["extracted_duration_seconds"])
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{local_offset:.3f}",
        "-i",
        str(source_clip),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(output_path),
    ]
    print(
        f"[EXTRACT] event={row['audio_event_id']} species={row['event_type']} "
        f"offset={local_offset:.3f}s duration={duration:.3f}s"
    )
    if not dry_run:
        run(cmd)
    return output_path


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")

    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    ok = 0
    failures: list[str] = []
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        try:
            source_clip, local_offset = download_source_clip(
                row=row,
                clip_prefix=args.clip_prefix.rstrip("/"),
                cache_dir=args.cache_dir,
                profile=args.profile,
                dry_run=args.dry_run,
            )
            output_path = extract_segment(
                row=row,
                source_clip=source_clip,
                local_offset=local_offset,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            ok += 1
            if index <= 5 or ok % 25 == 0:
                print(f"[OK] {ok}/{len(rows)} {output_path}")
        except Exception as exc:
            failures.append(f"{row.get('audio_event_id', '<missing>')}: {exc}")
            print(f"[FAIL] {failures[-1]}", file=sys.stderr)

    print(f"[DONE] extracted={ok} failed={len(failures)} total={len(rows)}")
    if failures:
        for failure in failures[:20]:
            print(f"[FAILURE] {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
