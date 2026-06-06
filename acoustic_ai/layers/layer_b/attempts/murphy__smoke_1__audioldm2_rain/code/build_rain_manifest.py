#!/usr/bin/env python3
"""Build Layer B rain smoke manifests from debug training pool.

Outputs:
  - data/rain_manifest.csv
  - data/rain_manifest_val.csv

Split policy (recording-group, deterministic):
  - validation recordings: 214659, 1401415
  - remaining recordings -> train
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


TRAIN_CAPTION = "steady rain over dry eucalyptus woodland, Bowra, Australia"
RAIN_WIND_CAPTION = (
    "steady rain with light wind over dry eucalyptus woodland, Bowra, Australia"
)
VAL_RECORDING_IDS = {214659, 1401415}


def _normalize_caption(overall_label: str) -> str:
    label = (overall_label or "").strip().lower()
    if label == "rain":
        return TRAIN_CAPTION
    if label == "rain+wind":
        return RAIN_WIND_CAPTION
    raise ValueError(f"unexpected overall_label={overall_label!r}; expected rain/rain+wind")


def _read_training_meta(path: Path) -> dict[str, dict[str, str]]:
    by_audio_path: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["training_audio_path"].strip()
            by_audio_path[key] = row
    return by_audio_path


def _build_rows(caption_manifest: Path, training_manifest: Path) -> tuple[list[dict], list[dict]]:
    training_meta = _read_training_meta(training_manifest)
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    with caption_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            audio_path = row["audio_path"].strip()
            recording_id = int(str(row["recording_id"]).strip())
            overall_label = str(row["overall_label"]).strip().lower()
            meta = training_meta.get(audio_path, {})

            out = {
                "audio_path": audio_path,
                "caption": _normalize_caption(overall_label),
                "status": "ok",
                "asset_id": f"rain_smoke_v0_{idx:03d}",
                "source_recording_id": str(recording_id),
                "overall_label": overall_label,
                "rain_confidence": str(row.get("rain_confidence", "")).strip(),
                "rain_intensity": str(meta.get("rain_intensity", "")).strip(),
                "wind_intensity": str(meta.get("wind_intensity", "")).strip(),
                "pool_selection_rule": str(row.get("pool_selection_rule", "")).strip(),
            }
            if recording_id in VAL_RECORDING_IDS:
                val_rows.append(out)
            else:
                train_rows.append(out)

    return train_rows, val_rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "audio_path",
        "caption",
        "status",
        "asset_id",
        "source_recording_id",
        "overall_label",
        "rain_confidence",
        "rain_intensity",
        "wind_intensity",
        "pool_selection_rule",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict], name: str) -> None:
    rec_counts = Counter(r["source_recording_id"] for r in rows)
    label_counts = Counter(r["overall_label"] for r in rows)
    print(f"[{name}] rows={len(rows)} recordings={dict(sorted(rec_counts.items()))}")
    print(f"[{name}] labels={dict(sorted(label_counts.items()))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rain smoke train/val manifests")
    parser.add_argument(
        "--caption-manifest",
        type=Path,
        default=Path(
            "debug/murphy_layer_b_rain_smoke_training_pool_v0_20260606/caption_manifest.csv"
        ),
    )
    parser.add_argument(
        "--training-pool-manifest",
        type=Path,
        default=Path(
            "debug/murphy_layer_b_rain_smoke_training_pool_v0_20260606/training_pool_manifest.csv"
        ),
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path(
            "acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__audioldm2_rain/data/rain_manifest.csv"
        ),
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path(
            "acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__audioldm2_rain/data/rain_manifest_val.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows, val_rows = _build_rows(args.caption_manifest, args.training_pool_manifest)
    _write_manifest(args.train_output, train_rows)
    _write_manifest(args.val_output, val_rows)
    _summarize(train_rows, "train")
    _summarize(val_rows, "val")


if __name__ == "__main__":
    main()
