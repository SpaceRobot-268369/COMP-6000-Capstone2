#!/usr/bin/env python3
"""Build a Layer C retrieval-compatible event index from the v2 pass manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_ROOT = REPO_ROOT / "resources" / "site_257_bowra-dry-a" / "layer_c_retrieval_event_library_v2"
DEFAULT_MANIFEST = LIB_ROOT / "layer_c_retrieval_v2_pass_manifest.csv"
DEFAULT_OUTPUT = LIB_ROOT / "layer_c_retrieval_v2_event_index.csv"

INDEX_COLUMNS = [
    "snippet_id",
    "event_type",
    "species_common_name",
    "species_scientific_name",
    "audio_event_id",
    "audio_path",
    "score",
    "quality_score",
    "diel_bin",
    "season",
    "duration_s",
    "recording_id",
    "event_start_seconds",
    "event_end_seconds",
    "source_manifest",
    "verdict",
    "notes",
]


def normalise_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.pass_manifest.is_absolute():
        args.pass_manifest = REPO_ROOT / args.pass_manifest
    if not args.output.is_absolute():
        args.output = REPO_ROOT / args.output

    rows: list[dict[str, str]] = []
    with args.pass_manifest.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("review_verdict", "").strip().lower() != "pass":
                continue
            audio_path = normalise_path(row["crop_bandpass_wav"])
            if not (REPO_ROOT / audio_path).exists():
                raise FileNotFoundError(f"Missing crop_bandpass_wav: {audio_path}")
            species_slug = row["species_slug"]
            sample_index = int(float(row["sample_index"]))
            low_hz = row.get("sample_low_hz", "")
            high_hz = row.get("sample_high_hz", "")
            rows.append(
                {
                    "snippet_id": f"{species_slug}_v2_{sample_index:03d}_{row['audio_event_id']}",
                    "event_type": species_slug,
                    "species_common_name": row["species_common_name"],
                    "species_scientific_name": row["species_scientific_name"],
                    "audio_event_id": row["audio_event_id"],
                    "audio_path": audio_path,
                    "score": row.get("score", ""),
                    "quality_score": row.get("quality_score") or row.get("score", ""),
                    "diel_bin": row.get("diel_bin", ""),
                    "season": row.get("season", ""),
                    "duration_s": row.get("crop_full_duration_s", ""),
                    "recording_id": row.get("recording_id", ""),
                    "event_start_seconds": row.get("event_start_s", ""),
                    "event_end_seconds": row.get("event_end_s", ""),
                    "source_manifest": normalise_path(str(args.pass_manifest)),
                    "verdict": "Pass",
                    "notes": f"v2 full review pass; bandpass={low_hz}-{high_hz}Hz",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    species_count = len({row["event_type"] for row in rows})
    print(f"rows={len(rows)}")
    print(f"species={species_count}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
