"""Embed every ambient_segment once and cache to disk.

Output:
- data/embeddings_cache.npy   — (N, D) float32, L2-normed, indexed by row in ambient_index.csv
- data/embeddings_meta.json   — {segment_ids: [...], model_id, sample_rate, n, dim}

Re-run is idempotent; cache is reused by build_anchors.py and eval.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from clap_backbone import CLAPBackbone, MODEL_ID, TARGET_SR  # noqa: E402
from paths import AMBIENT_INDEX, DATA_DIR, SEGMENTS_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-embed even if cache exists.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "embeddings_cache.npy"
    meta_path = DATA_DIR / "embeddings_meta.json"

    if cache_path.exists() and meta_path.exists() and not args.force:
        meta = json.loads(meta_path.read_text())
        print(f"cache present: n={meta['n']} dim={meta['dim']} model={meta['model_id']}")
        print("re-run with --force to recompute.")
        return

    df = pd.read_csv(AMBIENT_INDEX)
    paths = [str(SEGMENTS_DIR / f"{sid}.wav") for sid in df["segment_id"]]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} ambient_segment WAV files missing on disk (first: {missing[0]}). "
            "Run `dvc pull` for ambient_segments.dvc."
        )

    backbone = CLAPBackbone()
    print(f"device={backbone.device} model={MODEL_ID} embedding {len(paths)} segments...")
    t0 = time.time()
    emb = backbone.embed_audio(paths, verbose=True)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s ({elapsed / len(paths) * 1000:.1f} ms/segment)")

    np.save(cache_path, emb)
    meta = {
        "model_id": MODEL_ID,
        "sample_rate": TARGET_SR,
        "n": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "segment_ids": df["segment_id"].tolist(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {cache_path} and {meta_path}")


if __name__ == "__main__":
    main()
