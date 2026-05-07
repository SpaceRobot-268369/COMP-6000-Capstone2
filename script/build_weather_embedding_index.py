#!/usr/bin/env python3
"""Build the optional Layer B weather embedding index.

This creates a compact local .npz artifact from the existing
weather_asset_manifest.csv. It does not download audio, train a model, or
generate audio. The artifact is ignored by git and can be rebuilt locally.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from layer_b import (  # noqa: E402
    DEFAULT_ASSET_MANIFEST,
    DEFAULT_EMBEDDING_INDEX,
    WEATHER_EMBEDDING_VERSION,
    weather_embedding_feature_names,
    weather_row_embedding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact optional weather embedding index for Layer B reranking."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ASSET_MANIFEST,
        help="Layer B weather asset manifest with analysed audio features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EMBEDDING_INDEX,
        help="Output .npz path. This artifact should stay local and gitignored.",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=0,
        help="Optional cap for prototype indexes. Use 0 for all analysed rows.",
    )
    return parser.parse_args()


def clip_key(row: dict) -> str:
    return str(row.get("clip_path") or f"{row.get('recording_id')}:{row.get('clip_index')}")


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Weather asset manifest not found: {args.manifest}")

    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row.get("analysis_status") == "ok" and row.get("clip_path")
        ]

    if args.max_assets > 0:
        rows = rows[: args.max_assets]

    if not rows:
        raise RuntimeError("No analysed weather asset rows available for embedding index.")

    embeddings = np.stack([weather_row_embedding(row) for row in rows]).astype("float32")
    clip_keys = np.asarray([clip_key(row) for row in rows], dtype="U512")
    feature_names = np.asarray(weather_embedding_feature_names(), dtype="U128")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(args.output),
        version=np.asarray(WEATHER_EMBEDDING_VERSION),
        clip_keys=clip_keys,
        embeddings=embeddings,
        feature_names=feature_names,
        source_manifest=np.asarray(str(args.manifest.relative_to(PROJECT_ROOT))),
    )

    print(f"Written {len(rows)} weather embeddings -> {args.output}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print("This .npz is a local artifact and should not be committed.")


if __name__ == "__main__":
    main()
