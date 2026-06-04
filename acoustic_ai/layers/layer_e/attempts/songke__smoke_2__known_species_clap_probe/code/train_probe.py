"""Train a classifier head on frozen CLAP embeddings."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from common import build_probe, device, load_config, project_path, set_seed
from metrics import classification_metrics


def main() -> int:
    cfg = load_config()
    set_seed(int(cfg["training"]["seed"]))
    labels = list(cfg["data"]["labels"])
    run_device = device()
    emb_dir = project_path(cfg["output"]["embedding_dir"])
    model_dir = project_path(cfg["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    X_np = np.load(emb_dir / "embeddings.npy")
    meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    y_np = np.array(meta["class_indices"], dtype=np.int64)
    splits = np.array(meta["splits"])

    X = torch.from_numpy(X_np).float().to(run_device)
    y = torch.from_numpy(y_np).long().to(run_device)
    masks = {split: torch.from_numpy(splits == split).to(run_device) for split in ("train", "val", "test")}

    probe = build_probe(
        X.shape[1],
        len(labels),
        str(cfg["training"]["arch"]),
        int(cfg["training"]["hidden"]),
    ).to(run_device)

    counts = torch.bincount(y[masks["train"]], minlength=len(labels)).float()
    weights = inverse_frequency_weights(counts).to(run_device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    best_score = -1.0
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        logits = probe(X[masks["train"]])
        loss = loss_fn(logits, y[masks["train"]])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(X[masks["val"]])
                val_metrics = classification_metrics(val_logits, y[masks["val"]], labels)
                train_acc = float((probe(X[masks["train"]]).argmax(dim=1) == y[masks["train"]]).float().mean())
            score = float(val_metrics["macro_f1"])
            history.append({
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "train_accuracy": train_acc,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            })
            print(
                f"epoch {epoch:03d} loss={float(loss.detach().cpu()):.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_metrics['accuracy']:.3f} "
                f"val_macro_f1={val_metrics['macro_f1']:.3f}"
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best state captured")

    checkpoint_path = model_dir / "best_probe.pt"
    torch.save({
        "state_dict": best_state,
        "labels": labels,
        "config": cfg,
        "in_dim": int(X.shape[1]),
        "arch": cfg["training"]["arch"],
        "hidden": int(cfg["training"]["hidden"]),
        "best_epoch": best_epoch,
    }, checkpoint_path)

    probe.load_state_dict(best_state)
    probe.to(run_device)
    probe.eval()
    with torch.no_grad():
        test_metrics = classification_metrics(probe(X[masks["test"]]), y[masks["test"]], labels)

    metrics = {
        "best_epoch": best_epoch,
        "train_counts": dict(zip(labels, [int(v) for v in counts.cpu().tolist()])),
        "history": history,
        "test_metrics": test_metrics,
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved {checkpoint_path}")
    print(json.dumps(test_metrics, indent=2))
    return 0


def inverse_frequency_weights(counts: torch.Tensor) -> torch.Tensor:
    total = counts.sum()
    weights = total / counts.clamp(min=1)
    return weights / weights.mean()


if __name__ == "__main__":
    raise SystemExit(main())
