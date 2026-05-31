"""Build the seed-42 source-clip-disjoint split, then the two anchor sets.

Outputs (all in data/):
- splits/train.csv, splits/val.csv          — source-clip-disjoint, stratified by cell
- anchors_text.npy                          — (16, D) — pure zero-shot text anchors
- anchors_audio.npy                         — (16, D) — train-only audio prototypes
- anchors_meta.json                         — provenance + counts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from clap_backbone import CLAPBackbone  # noqa: E402
from paths import (  # noqa: E402
    AMBIENT_INDEX,
    CELL_ORDER,
    DATA_DIR,
    PROD_ATTEMPT_KEY,
    REGISTRY,
    SPLITS_DIR,
)


def load_cell_prompts() -> dict[str, str]:
    reg = yaml.safe_load(REGISTRY.read_text())
    cells = reg["layers"]["layer_a"]["attempts"][PROD_ATTEMPT_KEY]["params"]["cells"]
    prompts = {name: body["prompt"] for name, body in cells.items()}
    missing = set(CELL_ORDER) - set(prompts)
    extra = set(prompts) - set(CELL_ORDER)
    if missing or extra:
        raise ValueError(f"registry cells mismatch: missing={missing}, extra={extra}")
    return prompts


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
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv", index=False)
    return train, val


def build_text_anchors(backbone: CLAPBackbone, prompts: dict[str, str]) -> np.ndarray:
    ordered = [prompts[c] for c in CELL_ORDER]
    return backbone.embed_text(ordered)


def build_audio_prototypes(train: pd.DataFrame, embeddings: np.ndarray, id_to_row: dict[str, int]) -> np.ndarray:
    out = np.zeros((16, embeddings.shape[1]), dtype=np.float32)
    for c_idx, cell in enumerate(CELL_ORDER):
        rows = [id_to_row[sid] for sid in train.loc[train["cell"] == cell, "segment_id"]]
        if not rows:
            raise ValueError(f"cell {cell} has zero train segments")
        proto = embeddings[rows].mean(axis=0)
        proto /= max(float(np.linalg.norm(proto)), 1e-12)
        out[c_idx] = proto
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("loading cell prompts...")
    prompts = load_cell_prompts()

    print(f"building seed-{args.seed} source-clip-disjoint split (val_frac={args.val_frac})...")
    train, val = make_split(seed=args.seed, val_frac=args.val_frac)
    print(f"  train segments: {len(train)} (clips: {train['source_clip'].nunique()})")
    print(f"  val segments:   {len(val)} (clips: {val['source_clip'].nunique()})")
    overlap = set(train["source_clip"]) & set(val["source_clip"])
    if overlap:
        raise RuntimeError(f"split leak: {len(overlap)} source_clip(s) in both train and val")

    cache_path = DATA_DIR / "embeddings_cache.npy"
    meta_path = DATA_DIR / "embeddings_meta.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"missing {cache_path} — run embed_segments.py first."
        )
    embeddings = np.load(cache_path)
    meta = json.loads(meta_path.read_text())
    id_to_row = {sid: i for i, sid in enumerate(meta["segment_ids"])}

    print("loading CLAP backbone for text anchors...")
    backbone = CLAPBackbone()

    print("building text anchors (16 × D)...")
    text_anchors = build_text_anchors(backbone, prompts)
    np.save(DATA_DIR / "anchors_text.npy", text_anchors)

    print("building audio-prototype anchors (16 × D) from train split...")
    audio_anchors = build_audio_prototypes(train, embeddings, id_to_row)
    np.save(DATA_DIR / "anchors_audio.npy", audio_anchors)

    out_meta = {
        "cell_order": CELL_ORDER,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_train_clips": int(train["source_clip"].nunique()),
        "n_val_clips": int(val["source_clip"].nunique()),
        "embedding_dim": int(text_anchors.shape[1]),
        "model_id": meta["model_id"],
        "prompts": prompts,
    }
    (DATA_DIR / "anchors_meta.json").write_text(json.dumps(out_meta, indent=2))
    print("wrote anchors_text.npy, anchors_audio.npy, anchors_meta.json")


if __name__ == "__main__":
    main()
