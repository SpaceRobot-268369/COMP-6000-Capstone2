#!/usr/bin/env python3
"""Build a Layer B weather-asset candidate index from FSD50K metadata.

This script does not download audio. It reads FSD50K ground-truth CSVs plus
Freesound metadata JSON, filters likely rain/wind clips, and writes a candidate
CSV with source/license fields that can later be audited by listening.

Expected FSD50K metadata inputs:
  - FSD50K.ground_truth/dev.csv
  - FSD50K.ground_truth/eval.csv
  - FSD50K.metadata/dev_clips_info_FSD50K.json
  - FSD50K.metadata/eval_clips_info_FSD50K.json

Usage:
  python3 script/dataset/build_layer_b_fsd50k_candidates.py \
    --fsd50k-root /path/to/FSD50K \
    --output acoustic_ai/data/weather/fsd50k_weather_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("acoustic_ai/data/weather/fsd50k_weather_candidates.csv")

# Keep commercial reuse possible unless the project explicitly chooses a
# research-only asset library.
DEFAULT_ALLOWED_LICENSES = {
    "Creative Commons 0",
    "Attribution",
    "CC0",
    "CC-BY",
}

WIND_LABELS = {
    "Wind",
    "Wind noise (microphone)",
}
RAIN_LABELS = {
    "Rain",
    "Raindrop",
}

WIND_KEYWORDS = {
    "wind",
    "windy",
    "gust",
    "gusts",
    "stormwind",
}
RAIN_KEYWORDS = {
    "rain",
    "rainy",
    "raindrop",
    "raindrops",
    "rainfall",
    "drizzle",
    "downpour",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Layer B FSD50K weather asset candidates."
    )
    parser.add_argument("--fsd50k-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-license",
        action="append",
        default=None,
        help=(
            "Allowed per-clip license. Repeat to allow multiple. Defaults to "
            "CC0/CC-BY style licenses and excludes non-commercial licenses."
        ),
    )
    parser.add_argument(
        "--max-per-bucket",
        type=int,
        default=50,
        help="Maximum candidates per layer/intensity bucket.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed = set(args.allow_license or DEFAULT_ALLOWED_LICENSES)

    rows = []
    rows.extend(_load_split(args.fsd50k_root, "dev", allowed))
    rows.extend(_load_split(args.fsd50k_root, "eval", allowed))
    rows = _dedupe_and_cap(rows, args.max_per_bucket)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} Layer B weather candidates -> {args.output}")


def _load_split(root: Path, split: str, allowed_licenses: set[str]) -> list[dict[str, str]]:
    gt_path = root / "FSD50K.ground_truth" / f"{split}.csv"
    info_path = root / "FSD50K.metadata" / f"{split}_clips_info_FSD50K.json"

    if not gt_path.exists():
        raise FileNotFoundError(f"Missing FSD50K ground-truth CSV: {gt_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"Missing FSD50K clip metadata JSON: {info_path}")

    clip_info = json.loads(info_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []

    with gt_path.open(newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            fname = str(item["fname"]).strip()
            info = _lookup_clip_info(clip_info, fname)
            license_name = _metadata_value(info, "license", "licence")
            if allowed_licenses and license_name not in allowed_licenses:
                continue

            labels = _split_list(item.get("labels", ""))
            tags = _metadata_list(info, "tags")
            text_blob = " ".join(
                labels
                + tags
                + [
                    _metadata_value(info, "title"),
                    _metadata_value(info, "description"),
                ]
            ).lower()

            layer = _infer_layer(labels, text_blob)
            if layer is None:
                continue

            rows.append(
                {
                    "asset_id": f"fsd50k_{fname}",
                    "clip_path": f"{layer}/uncurated/{fname}.wav",
                    "layer": layer,
                    "intensity": "uncurated",
                    "source": "FSD50K/Freesound",
                    "source_url": f"https://freesound.org/s/{fname}/",
                    "license": license_name,
                    "attribution": _metadata_value(info, "username", "user", "uploader"),
                    "fsd50k_split": split,
                    "fsd50k_fname": fname,
                    "labels": ";".join(labels),
                    "tags": ";".join(tags),
                    "notes": "Candidate only; listen and assign light/moderate/heavy before promotion.",
                }
            )

    return rows


def _infer_layer(labels: list[str], text_blob: str) -> str | None:
    label_set = set(labels)
    if label_set & RAIN_LABELS or any(k in text_blob for k in RAIN_KEYWORDS):
        return "rain"
    if label_set & WIND_LABELS or any(k in text_blob for k in WIND_KEYWORDS):
        return "wind"
    return None


def _lookup_clip_info(clip_info: dict[str, Any], fname: str) -> dict[str, Any]:
    value = clip_info.get(fname) or clip_info.get(f"{fname}.wav") or {}
    return value if isinstance(value, dict) else {}


def _metadata_value(info: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value.strip()
        return str(value)
    return ""


def _metadata_list(info: dict[str, Any], key: str) -> list[str]:
    value = info.get(key, [])
    if isinstance(value, str):
        return _split_list(value)
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _split_list(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.replace(",", ";").split(";") if part.strip()]


def _dedupe_and_cap(rows: list[dict[str, str]], max_per_bucket: int) -> list[dict[str, str]]:
    seen = set()
    counts: dict[str, int] = {}
    result = []

    for row in sorted(rows, key=lambda r: (r["layer"], r["asset_id"])):
        key = row["asset_id"]
        bucket = row["layer"]
        if key in seen:
            continue
        if counts.get(bucket, 0) >= max_per_bucket:
            continue
        seen.add(key)
        counts[bucket] = counts.get(bucket, 0) + 1
        result.append(row)

    return result


def _fieldnames() -> list[str]:
    return [
        "asset_id",
        "clip_path",
        "layer",
        "intensity",
        "source",
        "source_url",
        "license",
        "attribution",
        "fsd50k_split",
        "fsd50k_fname",
        "labels",
        "tags",
        "notes",
    ]


if __name__ == "__main__":
    main()
