"""Embed local E-C species clips with frozen CLAP."""

from __future__ import annotations

import csv
import json
import time

import numpy as np

from clap_backbone import CLAPBackbone, MODEL_ID, TARGET_SR
from common import load_config, project_path


def main() -> int:
    cfg = load_config()
    manifest = project_path(cfg["data"]["manifest"])
    out_dir = project_path(cfg["output"]["embedding_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "meta.json"

    rows = list(csv.DictReader(manifest.open("r", newline="", encoding="utf-8")))
    paths = [project_path(row["audio_path"]) for row in rows]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} clips missing; first={missing[0]}")

    backbone = CLAPBackbone()
    print(f"device={backbone.device} model={MODEL_ID} clips={len(paths)}")
    start = time.time()
    embeddings = backbone.embed_audio(paths, verbose=True)
    elapsed = time.time() - start

    np.save(emb_path, embeddings)
    meta = {
        "model_id": MODEL_ID,
        "sample_rate": TARGET_SR,
        "n": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "manifest": str(manifest),
        "clip_ids": [row["clip_id"] for row in rows],
        "labels": [row["label"] for row in rows],
        "class_indices": [int(row["class_index"]) for row in rows],
        "splits": [row["split"] for row in rows],
        "audio_paths": [row["audio_path"] for row in rows],
        "elapsed_s": elapsed,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {emb_path}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
