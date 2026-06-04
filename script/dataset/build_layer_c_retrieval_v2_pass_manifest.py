#!/usr/bin/env python3
"""Build the final Layer C retrieval v2 Pass manifest from a review package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_ROOT = REPO_ROOT / "resources" / "site_257_bowra-dry-a" / "layer_c_retrieval_event_library_v2"
DEFAULT_REVIEW_PACKAGE = LIB_ROOT / "review_package_full_v2_s3reuse"
DEFAULT_OUTPUT = LIB_ROOT / "layer_c_retrieval_v2_pass_manifest.csv"

FIELDNAMES = [
    "species_common_name",
    "species_scientific_name",
    "species_slug",
    "species_rank",
    "sample_index",
    "audio_event_id",
    "recording_id",
    "score",
    "quality_score",
    "diel_bin",
    "season",
    "sample_local_date",
    "event_start_s",
    "event_end_s",
    "pre_buffer_s",
    "post_buffer_s",
    "crop_full_duration_s",
    "sample_low_hz",
    "sample_high_hz",
    "review_verdict",
    "item_dir",
    "crop_full_wav",
    "crop_bandpass_wav",
    "mel_full_png",
    "mel_bandpass_png",
    "metadata_json",
    "review_json",
    "source_audio_resolved",
    "source_resolution_kind",
    "listen_url",
    "library_url",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-package", type=Path, default=DEFAULT_REVIEW_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.review_package.is_absolute():
        args.review_package = REPO_ROOT / args.review_package
    if not args.output.is_absolute():
        args.output = REPO_ROOT / args.output

    summary_path = args.review_package / "review_package_summary_v2.csv"
    rows: list[dict[str, str]] = []
    missing_required: list[str] = []

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        for summary_row in csv.DictReader(f):
            item_dir = REPO_ROOT / summary_row["item_dir"]
            metadata_path = item_dir / "metadata.json"
            review_path = item_dir / "review.json"
            crop_full = item_dir / "crop_full.wav"
            crop_bandpass = item_dir / "crop_bandpass.wav"
            mel_full = item_dir / "mel_full.png"
            mel_bandpass = item_dir / "mel_bandpass.png"
            required = [
                metadata_path,
                review_path,
                crop_full,
                crop_bandpass,
                mel_full,
                mel_bandpass,
            ]
            if not all(path.exists() and path.stat().st_size > 0 for path in required):
                missing_required.append(rel(item_dir))
                continue

            metadata = read_json(metadata_path)
            review = read_json(review_path)
            verdict = str(review.get("verdict", "")).strip()
            if verdict.lower() != "pass":
                continue

            rows.append(
                {
                    "species_common_name": str(metadata.get("species_common_name", "")),
                    "species_scientific_name": str(metadata.get("species_scientific_name", "")),
                    "species_slug": str(metadata.get("species_slug", "")),
                    "species_rank": str(metadata.get("species_rank", "")),
                    "sample_index": str(metadata.get("sample_index", "")),
                    "audio_event_id": str(metadata.get("audio_event_id", "")),
                    "recording_id": str(metadata.get("recording_id", "")),
                    "score": str(metadata.get("score", "")),
                    "quality_score": str(metadata.get("quality_score", "")),
                    "diel_bin": str(metadata.get("diel_bin", "")),
                    "season": str(metadata.get("season", "")),
                    "sample_local_date": str(metadata.get("sample_local_date", "")),
                    "event_start_s": str(metadata.get("event_start_s", "")),
                    "event_end_s": str(metadata.get("event_end_s", "")),
                    "pre_buffer_s": str(metadata.get("pre_buffer_s", "")),
                    "post_buffer_s": str(metadata.get("post_buffer_s", "")),
                    "crop_full_duration_s": str(metadata.get("crop_full_duration_s", "")),
                    "sample_low_hz": str(metadata.get("sample_low_hz", "")),
                    "sample_high_hz": str(metadata.get("sample_high_hz", "")),
                    "review_verdict": verdict,
                    "item_dir": rel(item_dir),
                    "crop_full_wav": rel(crop_full),
                    "crop_bandpass_wav": rel(crop_bandpass),
                    "mel_full_png": rel(mel_full),
                    "mel_bandpass_png": rel(mel_bandpass),
                    "metadata_json": rel(metadata_path),
                    "review_json": rel(review_path),
                    "source_audio_resolved": str(metadata.get("source_audio_resolved", "")),
                    "source_resolution_kind": str(metadata.get("source_resolution_kind", "")),
                    "listen_url": str(metadata.get("listen_url", "")),
                    "library_url": str(metadata.get("library_url", "")),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"manifest_rows={len(rows)}")
    print(f"missing_required={len(missing_required)}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
