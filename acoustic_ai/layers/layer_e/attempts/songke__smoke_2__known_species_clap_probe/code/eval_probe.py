"""Evaluate the trained CLAP probe."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common import build_probe, device, load_config, project_path
from metrics import classification_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config()
    labels = list(cfg["data"]["labels"])
    run_device = device()
    emb_dir = project_path(cfg["output"]["embedding_dir"])
    model_dir = project_path(cfg["output"]["model_dir"])
    checkpoint = torch.load(model_dir / "best_probe.pt", map_location=run_device)

    X_np = np.load(emb_dir / "embeddings.npy")
    meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    y_np = np.array(meta["class_indices"], dtype=np.int64)
    splits = np.array(meta["splits"])
    mask = torch.from_numpy(splits == args.split).to(run_device)
    X = torch.from_numpy(X_np).float().to(run_device)
    y = torch.from_numpy(y_np).long().to(run_device)

    probe = build_probe(
        int(checkpoint["in_dim"]),
        len(labels),
        str(checkpoint["arch"]),
        int(checkpoint["hidden"]),
    ).to(run_device)
    probe.load_state_dict(checkpoint["state_dict"])
    probe.eval()

    with torch.no_grad():
        metrics = classification_metrics(probe(X[mask]), y[mask], labels)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
