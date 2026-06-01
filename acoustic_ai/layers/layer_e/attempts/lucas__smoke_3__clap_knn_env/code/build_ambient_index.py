"""Build the seed-42 source-clip-disjoint index/query split.

The index is the searchable pool of L2-normed CLAP embeddings (one row per
ambient_segment) aligned with metadata. Query segments are held out: all
segments from a "query" source_clip are excluded from the index, so no
segment retrieves a near-duplicate from the same recording.

Outputs (all in data/):
- index_embeddings.npy   — (N_index, D) float32, L2-normed
- index_meta.csv         — (N_index rows) segment_id, source_clip, diel_bin,
                          season, hour_sin, hour_cos, month_sin, month_cos
- splits/index.csv       — full ambient_index rows assigned to the index
- splits/query.csv       — held-out query rows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from paths import AMBIENT_INDEX, DATA_DIR, SPLITS_DIR  # noqa: E402


META_COLS = [
    "segment_id",
    "source_clip",
    "diel_bin",
    "season",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def make_split(seed: int, query_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(AMBIENT_INDEX)
    df["cell"] = df["season"] + "_" + df["diel_bin"]
    rng = np.random.default_rng(seed)
    index_rows, query_rows = [], []
    for cell, group in df.groupby("cell", sort=False):
        clips = group["source_clip"].unique().tolist()
        rng.shuffle(clips)
        n_q = max(1, int(np.ceil(len(clips) * query_frac)))
        query_clips = set(clips[:n_q])
        index_rows.append(group[~group["source_clip"].isin(query_clips)])
        query_rows.append(group[group["source_clip"].isin(query_clips)])
    index_df = pd.concat(index_rows, ignore_index=True)
    query_df = pd.concat(query_rows, ignore_index=True)
    overlap = set(index_df["source_clip"]) & set(query_df["source_clip"])
    if overlap:
        raise RuntimeError(f"split leak: {len(overlap)} source_clip(s) in both index and query")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(SPLITS_DIR / "index.csv", index=False)
    query_df.to_csv(SPLITS_DIR / "query.csv", index=False)
    return index_df, query_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-frac", type=float, default=0.2)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "embeddings_cache.npy"
    meta_path = DATA_DIR / "embeddings_meta.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"missing {cache_path} — run embed_segments.py first.")
    embeddings = np.load(cache_path)
    meta = json.loads(meta_path.read_text())
    id_to_row = {sid: i for i, sid in enumerate(meta["segment_ids"])}

    print(f"building seed-{args.seed} source-clip-disjoint split (query_frac={args.query_frac})...")
    index_df, query_df = make_split(seed=args.seed, query_frac=args.query_frac)
    print(f"  index segments: {len(index_df)} (clips: {index_df['source_clip'].nunique()})")
    print(f"  query segments: {len(query_df)} (clips: {query_df['source_clip'].nunique()})")

    index_rows = [id_to_row[sid] for sid in index_df["segment_id"]]
    index_emb = embeddings[index_rows]
    np.save(DATA_DIR / "index_embeddings.npy", index_emb)
    index_df[META_COLS].to_csv(DATA_DIR / "index_meta.csv", index=False)

    out_meta = {
        "n_index": int(len(index_df)),
        "n_query": int(len(query_df)),
        "n_index_clips": int(index_df["source_clip"].nunique()),
        "n_query_clips": int(query_df["source_clip"].nunique()),
        "embedding_dim": int(index_emb.shape[1]),
        "model_id": meta["model_id"],
        "seed": args.seed,
        "query_frac": args.query_frac,
    }
    (DATA_DIR / "index_build_meta.json").write_text(json.dumps(out_meta, indent=2))
    print(f"wrote index_embeddings.npy ({index_emb.shape}) + index_meta.csv")


if __name__ == "__main__":
    main()
