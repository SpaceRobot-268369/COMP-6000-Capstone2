"""Train E-B MVP-3 balanced MLP weather heads on frozen PANNs/DSP features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from layers.layer_e.attempts.liting__mvp_1__panns_weather_baseline.code.weather_detector import (  # noqa: E402
    load_weather_assets_from_index,
)
from layers.layer_e.attempts.liting__mvp_3__balanced_weather_head.code.weather_head import (  # noqa: E402
    FEATURE_NAMES,
    RAIN_CLASSES,
    WIND_CLASSES,
    extract_feature_vector,
    normalise_label,
    predict_with_checkpoint,
)


DEFAULT_INDEX = (
    PROJECT_ROOT
    / "acoustic_ai"
    / "layers"
    / "layer_b"
    / "attempts"
    / "lucas__smoke_1__curated_assets"
    / "data"
    / "weather"
    / "asset_index.csv"
)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model" / "candidates" / "liting" / "mvp_3__balanced_weather_head"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "debug" / "e_b_weather_mvp3"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train E-B MVP-3 balanced weather head.")
    parser.add_argument("--asset-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start_total = time.perf_counter()
    assets = [
        asset
        for asset in load_weather_assets_from_index(args.asset_index)
        if asset.audio_path.exists()
        and (asset.metadata or {}).get("source_type") == "site"
        and asset.labels.get("rain") != "unclear"
        and asset.labels.get("wind") != "unclear"
    ]
    if args.limit > 0:
        assets = assets[: args.limit]
    if len(assets) < 8:
        print(f"FAIL: need at least 8 materialised Site257 assets; found {len(assets)}")
        return 1

    print(f"assets={len(assets)} index={args.asset_index}")
    feature_start = time.perf_counter()
    rows = []
    vectors = []
    for asset in assets:
        vector, evidence = extract_feature_vector(asset.audio_path)
        vectors.append(vector)
        rows.append(
            {
                "asset_id": asset.asset_id,
                "audio_path": str(asset.audio_path.relative_to(PROJECT_ROOT)),
                "rain_label": normalise_label(asset.labels.get("rain"), RAIN_CLASSES),
                "wind_label": normalise_label(asset.labels.get("wind"), WIND_CLASSES),
                "policy_class": classify_policy(asset),
                "panns_available": evidence["panns_available"],
                "panns_status": evidence["panns_status"],
            }
        )
        print(
            f"[feature] {asset.asset_id}: rain={rows[-1]['rain_label']} "
            f"wind={rows[-1]['wind_label']} policy={rows[-1]['policy_class']} "
            f"panns={rows[-1]['panns_available']}"
        )
    feature_seconds = time.perf_counter() - feature_start

    X = np.vstack(vectors).astype(np.float32)
    train_idx, val_idx = make_split(rows, args.val_frac, args.seed)
    if not train_idx or not val_idx:
        print("FAIL: split produced empty train or validation set")
        return 1

    mu = X[train_idx].mean(axis=0)
    sigma = X[train_idx].std(axis=0) + 1e-6
    Xn = (X - mu) / sigma

    train_start = time.perf_counter()
    rain_head, rain_history = train_component(
        Xn,
        rows,
        train_idx,
        val_idx,
        "rain_label",
        RAIN_CLASSES,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    wind_head, wind_history = train_component(
        Xn,
        rows,
        train_idx,
        val_idx,
        "wind_label",
        WIND_CLASSES,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    train_seconds = time.perf_counter() - train_start

    checkpoint = {
        "attempt": "liting__mvp_3__balanced_weather_head",
        "head_architecture": "mlp_v1",
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mu.tolist(),
        "feature_std": sigma.tolist(),
        "rain_classes": list(RAIN_CLASSES),
        "wind_classes": list(WIND_CLASSES),
        "rain_head": head_to_dict(rain_head),
        "wind_head": head_to_dict(wind_head),
        "seed": args.seed,
    }
    eval_rows, metrics = evaluate(X, rows, val_idx, checkpoint)
    timings = {
        "feature_seconds": round(feature_seconds, 3),
        "training_seconds": round(train_seconds, 3),
        "total_seconds": round(time.perf_counter() - start_total, 3),
    }
    metrics.update(
        {
            "attempt": "liting__mvp_3__balanced_weather_head",
            "baseline_attempt": "liting__mvp_2__calibrated_weather_head",
            "head_architecture": "mlp_v1",
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "asset_index": str(args.asset_index),
            "case_count": len(rows),
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "timings": timings,
            "rain_history": rain_history,
            "wind_history": wind_history,
        }
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_dir / "weather_head.pt")
    (args.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(args.debug_dir / "feature_manifest.csv", rows)
    write_csv(args.debug_dir / "validation_predictions.csv", eval_rows)
    (args.debug_dir / "report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print(f"saved checkpoint: {args.model_dir / 'weather_head.pt'}")
    print(f"saved metrics: {args.model_dir / 'metrics.json'}")
    print(f"saved report: {args.debug_dir / 'report.json'}")
    print(
        "Summary: "
        f"rain_val_acc={metrics['rain_val_accuracy']:.3f}, "
        f"wind_val_acc={metrics['wind_val_accuracy']:.3f}, "
        f"training_seconds={timings['training_seconds']:.2f}, "
        f"total_seconds={timings['total_seconds']:.2f}"
    )
    return 0


def classify_policy(asset) -> str:
    rain = asset.labels.get("rain", "none") != "none"
    wind = asset.labels.get("wind", "none") != "none"
    if rain and wind:
        return "mixed_rain_wind"
    if rain:
        return "rain_primary"
    if wind:
        return "wind_primary"
    return "no_weather"


def make_split(rows: list[dict], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        key = row["policy_class"]
        groups.setdefault(key, []).append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for indices in groups.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_frac))) if len(shuffled) > 1 else 0
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])
    return sorted(train_idx), sorted(val_idx)


def train_component(
    X: np.ndarray,
    rows: list[dict],
    train_idx: list[int],
    val_idx: list[int],
    label_key: str,
    classes: tuple[str, ...],
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
) -> tuple[nn.Module, list[dict]]:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    balanced_idx = balanced_indices(rows, train_idx, label_key, classes)
    X_train = torch.from_numpy(X[balanced_idx]).float()
    y_train = torch.tensor([class_to_idx[rows[i][label_key]] for i in balanced_idx], dtype=torch.long)
    X_train_raw = torch.from_numpy(X[train_idx]).float()
    y_train_raw = torch.tensor([class_to_idx[rows[i][label_key]] for i in train_idx], dtype=torch.long)
    X_val = torch.from_numpy(X[val_idx]).float()
    y_val = torch.tensor([class_to_idx[rows[i][label_key]] for i in val_idx], dtype=torch.long)

    counts = torch.bincount(y_train_raw, minlength=len(classes)).float()
    weights = counts.sum() / (counts.clamp(min=1.0) * len(classes))
    model = WeatherMLP(X_train.shape[1], hidden_dim, len(classes), dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    history = []
    best_state = None
    best_val = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimiser.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimiser.step()
        if epoch % 25 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                train_acc = accuracy(model(X_train_raw), y_train_raw)
                val_acc = accuracy(model(X_val), y_val)
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(float(loss.detach()), 6),
                    "train_accuracy": round(train_acc, 3),
                    "val_accuracy": round(val_acc, 3),
                }
            )
            if val_acc > best_val:
                best_val = val_acc
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, history


class WeatherMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.hidden(x)))
        return self.output(hidden)


def balanced_indices(
    rows: list[dict],
    train_idx: list[int],
    label_key: str,
    classes: tuple[str, ...],
) -> list[int]:
    groups: dict[str, list[int]] = {label: [] for label in classes}
    for idx in train_idx:
        groups[rows[idx][label_key]].append(idx)

    non_empty = [indices for indices in groups.values() if indices]
    if not non_empty:
        return train_idx

    target = max(len(indices) for indices in non_empty)
    balanced: list[int] = []
    for indices in non_empty:
        repeats = (target + len(indices) - 1) // len(indices)
        balanced.extend((indices * repeats)[:target])

    return sorted(balanced)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean())


def head_to_dict(model: nn.Module) -> dict:
    if isinstance(model, WeatherMLP):
        return {
            "architecture": "mlp_v1",
            "hidden_weight": model.hidden.weight.detach().cpu().numpy().tolist(),
            "hidden_bias": model.hidden.bias.detach().cpu().numpy().tolist(),
            "output_weight": model.output.weight.detach().cpu().numpy().tolist(),
            "output_bias": model.output.bias.detach().cpu().numpy().tolist(),
        }
    return {
        "architecture": "linear",
        "weight": model.weight.detach().cpu().numpy().tolist(),
        "bias": model.bias.detach().cpu().numpy().tolist(),
    }


def evaluate(X: np.ndarray, rows: list[dict], val_idx: list[int], checkpoint: dict) -> tuple[list[dict], dict]:
    eval_rows = []
    rain_correct = 0
    wind_correct = 0
    rain_confusion = empty_confusion(RAIN_CLASSES)
    wind_confusion = empty_confusion(WIND_CLASSES)
    policy_breakdown: dict[str, dict[str, int]] = {}
    for idx in val_idx:
        pred = predict_with_checkpoint(X[idx], checkpoint)
        row = rows[idx]
        rain_ok = pred["rain_intensity"] == row["rain_label"]
        wind_ok = pred["wind_intensity"] == row["wind_label"]
        rain_correct += int(rain_ok)
        wind_correct += int(wind_ok)
        rain_confusion[row["rain_label"]][pred["rain_intensity"]] += 1
        wind_confusion[row["wind_label"]][pred["wind_intensity"]] += 1
        policy_stats = policy_breakdown.setdefault(
            row["policy_class"],
            {"count": 0, "rain_pass": 0, "wind_pass": 0, "joint_pass": 0},
        )
        policy_stats["count"] += 1
        policy_stats["rain_pass"] += int(rain_ok)
        policy_stats["wind_pass"] += int(wind_ok)
        policy_stats["joint_pass"] += int(rain_ok and wind_ok)
        eval_rows.append(
            {
                "asset_id": row["asset_id"],
                "policy_class": row["policy_class"],
                "expected_rain": row["rain_label"],
                "predicted_rain": pred["rain_intensity"],
                "rain_confidence": pred["component_confidence"]["rain"],
                "rain_status": "pass" if rain_ok else "fail",
                "expected_wind": row["wind_label"],
                "predicted_wind": pred["wind_intensity"],
                "wind_confidence": pred["component_confidence"]["wind"],
                "wind_status": "pass" if wind_ok else "fail",
            }
        )
    count = max(len(val_idx), 1)
    rain_accuracy = rain_correct / count
    wind_accuracy = wind_correct / count
    joint_accuracy = sum(1 for row in eval_rows if row["rain_status"] == "pass" and row["wind_status"] == "pass") / count
    gate = classify_gate(rain_accuracy, wind_accuracy, joint_accuracy, policy_breakdown)
    return eval_rows, {
        "rain_val_accuracy": rain_accuracy,
        "wind_val_accuracy": wind_accuracy,
        "joint_val_accuracy": joint_accuracy,
        "gate": gate,
        "confusion_matrix": {
            "rain": rain_confusion,
            "wind": wind_confusion,
        },
        "policy_breakdown": add_policy_rates(policy_breakdown),
        "demo_summary": {
            "model": "Frozen PANNs CNN14 + DSP features + two class-balanced MLP heads",
            "training_mode": "Server B training over Site257 weather-labelled clips",
            "output_contract": {
                "wind_intensity": list(WIND_CLASSES),
                "rain_intensity": list(RAIN_CLASSES),
                "thunder_intensity": "none until Site257 thunder evidence is validated",
                "confidence": "per-component probability from the calibrated head",
            },
            "duration_note": "Feature extraction reads the available audio clip; current Server B run used materialised Site257 weather clips and reports feature/training/total time separately.",
        },
        "validation_rows": eval_rows,
    }


def empty_confusion(classes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {expected: {predicted: 0 for predicted in classes} for expected in classes}


def add_policy_rates(policy_breakdown: dict[str, dict[str, int]]) -> dict[str, dict]:
    rated = {}
    for policy, stats in policy_breakdown.items():
        count = max(stats["count"], 1)
        rated[policy] = {
            **stats,
            "rain_accuracy": round(stats["rain_pass"] / count, 3),
            "wind_accuracy": round(stats["wind_pass"] / count, 3),
            "joint_accuracy": round(stats["joint_pass"] / count, 3),
        }
    return rated


def classify_gate(
    rain_accuracy: float,
    wind_accuracy: float,
    joint_accuracy: float,
    policy_breakdown: dict[str, dict[str, int]],
) -> dict[str, object]:
    single_component = {
        key: value for key, value in add_policy_rates(policy_breakdown).items() if key in {"rain_primary", "wind_primary", "no_weather"}
    }
    single_joint = [
        value["joint_accuracy"]
        for value in single_component.values()
        if isinstance(value.get("joint_accuracy"), float) and value.get("count", 0) > 0
    ]
    single_component_mean = sum(single_joint) / len(single_joint) if single_joint else 0.0
    passed = rain_accuracy >= 0.70 and wind_accuracy >= 0.70 and single_component_mean >= 0.65
    return {
        "status": "pass" if passed else "needs_iteration",
        "rain_accuracy_min": 0.70,
        "wind_accuracy_min": 0.70,
        "single_component_joint_accuracy_min": 0.65,
        "single_component_joint_accuracy": round(single_component_mean, 3),
        "interpretation": (
            "MVP3 is usable for demo and further calibration."
            if passed
            else "MVP3 should be treated as an iteration checkpoint, not a final detector."
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
