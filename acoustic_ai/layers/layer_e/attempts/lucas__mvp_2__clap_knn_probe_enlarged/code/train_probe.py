"""Train the season probe on frozen-CLAP embeddings.

A tiny head (linear by default, --mlp for one hidden layer) over the 512-d
L2-normed CLAP vector -> 4-way season. Backbone stays frozen; we only learn
the head on cached embeddings. Inverse-frequency class weights handle the
source-thin cells. Seed 42.

Reads the train/val split + embeddings_cache produced by build_split.py.
Writes the checkpoint to model/candidates/lucas/mvp_2__clap_knn_probe_enlarged/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from paths import (  # noqa: E402
    CANDIDATE_DIR, DATA_DIR, PROBE_PATH, SEASON_ORDER, SPLITS_DIR,
)

SEASON_IDX = {s: i for i, s in enumerate(SEASON_ORDER)}


def build_probe(in_dim: int, mlp: bool, hidden: int) -> nn.Module:
    if mlp:
        return nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, len(SEASON_ORDER)),
        )
    return nn.Linear(in_dim, len(SEASON_ORDER))


def load_xy(split_csv: Path, emb: np.ndarray, id_to_row: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(split_csv)
    rows = [id_to_row[sid] for sid in df["segment_id"]]
    X = torch.from_numpy(emb[rows]).float()
    y = torch.tensor([SEASON_IDX[s] for s in df["season"]], dtype=torch.long)
    return X, y


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp", action="store_true", help="Use a 1-hidden-layer MLP probe.")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    emb = np.load(DATA_DIR / "embeddings_cache.npy")
    meta = json.loads((DATA_DIR / "embeddings_meta.json").read_text())
    id_to_row = {sid: i for i, sid in enumerate(meta["segment_ids"])}

    X_tr, y_tr = load_xy(SPLITS_DIR / "train.csv", emb, id_to_row)
    X_va, y_va = load_xy(SPLITS_DIR / "val.csv", emb, id_to_row)
    print(f"train={len(y_tr)} val={len(y_va)} dim={X_tr.shape[1]}")

    # Inverse-frequency class weights from the train split.
    counts = torch.bincount(y_tr, minlength=len(SEASON_ORDER)).float()
    class_w = (counts.sum() / (counts.clamp(min=1) * len(SEASON_ORDER)))
    print("season counts:", {SEASON_ORDER[i]: int(counts[i]) for i in range(len(SEASON_ORDER))})

    probe = build_probe(X_tr.shape[1], args.mlp, args.hidden)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    best_val = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        probe.train()
        opt.zero_grad()
        loss = loss_fn(probe(X_tr), y_tr)
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == args.epochs:
            probe.eval()
            with torch.no_grad():
                tr_acc = accuracy(probe(X_tr), y_tr)
                va_acc = accuracy(probe(X_va), y_va)
            print(f"epoch {epoch:4d}  loss={float(loss.detach()):.4f}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")
            if va_acc > best_val:
                best_val = va_acc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "arch": "mlp" if args.mlp else "linear",
            "hidden": args.hidden,
            "in_dim": int(X_tr.shape[1]),
            "season_order": SEASON_ORDER,
            "best_val_acc": best_val,
            "seed": args.seed,
        },
        PROBE_PATH,
    )
    (CANDIDATE_DIR / "probe_meta.json").write_text(json.dumps({
        "arch": "mlp" if args.mlp else "linear",
        "best_val_season_acc": best_val,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "season_order": SEASON_ORDER,
    }, indent=2))
    print(f"saved probe -> {PROBE_PATH}  (best val season acc = {best_val:.3f})")


if __name__ == "__main__":
    main()
