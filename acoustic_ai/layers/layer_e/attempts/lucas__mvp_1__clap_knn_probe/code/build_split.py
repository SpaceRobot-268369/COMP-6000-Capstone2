"""Build the single seed-42 source-clip-disjoint split + index + anchors.

mvp_1 uses ONE split (cleaner than the smokes' two):
- train -> both the k-NN index AND the season-probe training set
- val   -> held-out eval (queried against the train index; never trained on)

Outputs (all in data/):
- splits/train.csv, splits/val.csv       — source-clip-disjoint, stratified by cell
- index_embeddings.npy                    — (N_train, D) L2-normed, the searchable pool
- index_meta.csv                          — aligned metadata for the index rows
- anchors_audio.npy                       — (16, D) train-only cell prototypes (agreement head)
- build_meta.json                         — provenance + counts

Requires embeddings_cache.npy from embed_segments.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from paths import AMBIENT_INDEX, CELL_ORDER, DATA_DIR, SPLITS_DIR  # noqa: E402

META_COLS = [
    "segment_id", "source_clip", "diel_bin", "season",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "cell",
]


def make_split(seed: int, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(AMBIENT_INDEX)
    df["cell"] = df["season"] + "_" + df["diel_bin"]
    rng = np.random.default_rng(seed)
    train_rows, val_rows = [], []
    for cell, group in df.groupby("cell", sort=False):
        clips = group["source_clip"].unique().tolist()
        rng.shuffle(clips)
        n_val = max(1, int(np.ceil(len(clips) * val_frac)))
        val_clips = set(clips[:n_val])
        train_rows.append(group[~group["source_clip"].isin(val_clips)])
        val_rows.append(group[group["source_clip"].isin(val_clips)])
    train = pd.concat(train_rows, ignore_index=True)
    val = pd.concat(val_rows, ignore_index=True)
    overlap = set(train["source_clip"]) & set(val["source_clip"])
    if overlap:
        raise RuntimeError(f"split leak: {len(overlap)} source_clip(s) in both train and val")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv", index=False)
    return train, val


def build_audio_prototypes(train: pd.DataFrame, emb: np.ndarray, id_to_row: dict[str, int]) -> np.ndarray:
    out = np.zeros((16, emb.shape[1]), dtype=np.float32)
    for c_idx, cell in enumerate(CELL_ORDER):
        rows = [id_to_row[sid] for sid in train.loc[train["cell"] == cell, "segment_id"]]
        if not rows:
            raise ValueError(f"cell {cell} has zero train segments")
        proto = emb[rows].mean(axis=0)
        proto /= max(float(np.linalg.norm(proto)), 1e-12)
        out[c_idx] = proto
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "embeddings_cache.npy"
    meta_path = DATA_DIR / "embeddings_meta.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"missing {cache_path} — run embed_segments.py first.")
    emb = np.load(cache_path)
    meta = json.loads(meta_path.read_text())
    id_to_row = {sid: i for i, sid in enumerate(meta["segment_ids"])}

    print(f"building seed-{args.seed} source-clip-disjoint split (val_frac={args.val_frac})...")
    train, val = make_split(seed=args.seed, val_frac=args.val_frac)
    print(f"  train: {len(train)} segs ({train['source_clip'].nunique()} clips)")
    print(f"  val:   {len(val)} segs ({val['source_clip'].nunique()} clips)")

    # Index = train embeddings + aligned metadata.
    index_rows = [id_to_row[sid] for sid in train["segment_id"]]
    index_emb = emb[index_rows]
    np.save(DATA_DIR / "index_embeddings.npy", index_emb)
    train[META_COLS].to_csv(DATA_DIR / "index_meta.csv", index=False)

    # Audio prototypes (agreement head) from train.
    anchors = build_audio_prototypes(train, emb, id_to_row)
    np.save(DATA_DIR / "anchors_audio.npy", anchors)

    out_meta = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_train_clips": int(train["source_clip"].nunique()),
        "n_val_clips": int(val["source_clip"].nunique()),
        "embedding_dim": int(emb.shape[1]),
        "model_id": meta["model_id"],
        "cell_order": CELL_ORDER,
    }
    (DATA_DIR / "build_meta.json").write_text(json.dumps(out_meta, indent=2))
    print(f"wrote index_embeddings.npy ({index_emb.shape}), index_meta.csv, anchors_audio.npy")


if __name__ == "__main__":
    main()
