"""Metrics helpers for CLAP probe training/evaluation."""

from __future__ import annotations

import torch


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor, labels: list[str]) -> dict:
    preds = logits.argmax(dim=1)
    num_classes = len(labels)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, pred in zip(targets.cpu().tolist(), preds.cpu().tolist()):
        confusion[target][pred] += 1

    total = int(targets.numel())
    correct = int((preds == targets).sum().item())
    per_class = {}
    f1_values = []
    for idx, label in enumerate(labels):
        tp = confusion[idx][idx]
        fp = sum(confusion[row][idx] for row in range(num_classes) if row != idx)
        fn = sum(confusion[idx][col] for col in range(num_classes) if col != idx)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[idx]),
        }

    return {
        "accuracy": correct / max(total, 1),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "rows_true_cols_pred": confusion,
        },
    }
