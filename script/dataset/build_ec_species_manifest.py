"""Merge local E-C species manifests into one training manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_LABELS = [
    "ninox_boobook",
    "laughing_kookaburra",
    "rhipidura_leucophrys",
    "psophodes_cristatus",
    "cincloramphus_mathewsi",
    "podargus_strigoides",
    "red_capped_robin",
    "anas_superciliosa",
    "australian_raven",
    "peaceful_dove",
    "galah",
    "crested_bellbird",
    "rainbow_bee_eater",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label order. This determines class_index.",
    )
    args = parser.parse_args()

    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    rows: list[dict[str, str]] = []

    for manifest in args.manifest:
        with manifest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"]
                if label not in label_to_index:
                    raise ValueError(f"Unknown label {label!r} in {manifest}")
                audio_path = Path(row["audio_path"])
                if not audio_path.exists():
                    raise FileNotFoundError(row["audio_path"])
                rows.append({
                    "clip_id": row["clip_id"],
                    "label": label,
                    "class_index": str(label_to_index[label]),
                    "split": row["split"],
                    "audio_path": row["audio_path"],
                    "source_file": row["source_file"],
                    "source_xc_id": row["source_xc_id"],
                    "start_s": row["start_s"],
                    "end_s": row["end_s"],
                    "duration_s": row["duration_s"],
                    "notes": row.get("notes", ""),
                })

    validate_no_source_split_leak(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output, rows)
    print_summary(rows, label_to_index)
    print(f"wrote: {args.output}")
    return 0


def validate_no_source_split_leak(rows: list[dict[str, str]]) -> None:
    by_label_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["label"], row["source_xc_id"])
        by_label_source[key].add(row["split"])
    leaks = {key: splits for key, splits in by_label_source.items() if len(splits) > 1}
    if leaks:
        preview = ", ".join(f"{key}:{sorted(value)}" for key, value in list(leaks.items())[:5])
        raise ValueError(f"Source split leakage detected: {preview}")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "clip_id",
        "label",
        "class_index",
        "split",
        "audio_path",
        "source_file",
        "source_xc_id",
        "start_s",
        "end_s",
        "duration_s",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], label_to_index: dict[str, int]) -> None:
    print("labels:", label_to_index)
    print("total:", len(rows))
    print("by label:", dict(Counter(row["label"] for row in rows)))
    print("by split:", dict(Counter(row["split"] for row in rows)))
    by_pair = Counter((row["label"], row["split"]) for row in rows)
    print("by label/split:")
    for label in label_to_index:
        print(" ", label, {
            split: by_pair[(label, split)]
            for split in ("train", "val", "test")
        })


if __name__ == "__main__":
    raise SystemExit(main())
