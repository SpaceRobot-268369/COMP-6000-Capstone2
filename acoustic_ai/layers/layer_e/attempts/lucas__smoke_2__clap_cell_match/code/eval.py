"""Score text-anchor vs audio-prototype heads on the held-out val split.

Outputs:
- metrics.json                                       (attempt root)
- data/confusion_text.png, data/confusion_audio.png  (16x16 per variant)
- report.md                                          (attempt root)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from paths import (  # noqa: E402
    ATTEMPT_DIR,
    CELL_ORDER,
    DATA_DIR,
    SPLITS_DIR,
    diel_of,
    season_of,
)


def softmax(x: np.ndarray, axis: int, tau: float) -> np.ndarray:
    s = x / tau
    s = s - s.max(axis=axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=axis, keepdims=True)


def score(anchors: np.ndarray, val_emb: np.ndarray, true_cells: list[str], tau: float) -> tuple[dict, np.ndarray]:
    sims = val_emb @ anchors.T  # (N_val, 16)
    pred_idx = sims.argmax(axis=1)
    preds = [CELL_ORDER[int(i)] for i in pred_idx]
    top3 = np.argsort(-sims, axis=1)[:, :3]
    top3_cells = [[CELL_ORDER[int(j)] for j in row] for row in top3]

    cell_top1 = float(np.mean([p == t for p, t in zip(preds, true_cells)]))
    cell_top3 = float(np.mean([t in row for row, t in zip(top3_cells, true_cells)]))
    season_acc = float(
        np.mean([season_of(p) == season_of(t) for p, t in zip(preds, true_cells)])
    )
    diel_acc = float(
        np.mean([diel_of(p) == diel_of(t) for p, t in zip(preds, true_cells)])
    )

    probs = softmax(sims, axis=1, tau=tau)
    confs = probs.max(axis=1)

    cm = np.zeros((16, 16), dtype=np.int64)
    for t, p in zip(true_cells, preds):
        cm[CELL_ORDER.index(t), CELL_ORDER.index(p)] += 1

    per_cell = {}
    for i, cell in enumerate(CELL_ORDER):
        n = int(cm[i].sum())
        per_cell[cell] = {
            "n": n,
            "acc": float(cm[i, i] / n) if n else 0.0,
        }

    return {
        "cell_top1": cell_top1,
        "cell_top3": cell_top3,
        "season_acc": season_acc,
        "diel_acc": diel_acc,
        "mean_confidence": float(confs.mean()),
        "per_cell": per_cell,
        "confusion_matrix": cm.tolist(),
    }, cm


def plot_confusion(cm: np.ndarray, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))
    ax.set_xticklabels(CELL_ORDER, rotation=90, fontsize=7)
    ax.set_yticklabels(CELL_ORDER, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"smoke_2 cell_match — {label} anchors")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(out: dict, report_path: Path) -> None:
    bar = out["smoke_bar"]
    lines = [
        "# smoke_2 cell_match — eval report",
        "",
        f"n_val = {out['n_val']} segments (source-clip-disjoint from train)",
        f"softmax tau = {out['tau']}",
        "",
        "## Smoke bar",
        f"- season_acc {bar['season_acc']}",
        f"- diel_acc {bar['diel_acc']}",
        f"- cell_top3 {bar['cell_top3']}",
        "",
        "## Results",
        "",
        "| metric | text anchors | audio prototypes |",
        "|---|---|---|",
    ]
    t = out["metrics"]["text"]
    a = out["metrics"]["audio"]
    for key, name in [
        ("cell_top1", "cell top-1"),
        ("cell_top3", "cell top-3"),
        ("season_acc", "season acc"),
        ("diel_acc", "diel acc"),
        ("mean_confidence", "mean confidence"),
    ]:
        lines.append(f"| {name} | {t[key]:.3f} | {a[key]:.3f} |")
    lines += [
        "",
        "Per-cell accuracy (audio prototypes):",
        "",
        "| cell | n | acc |",
        "|---|---|---|",
    ]
    for cell in CELL_ORDER:
        pc = a["per_cell"][cell]
        lines.append(f"| {cell} | {pc['n']} | {pc['acc']:.3f} |")
    lines += [
        "",
        "Confusion matrices: `data/confusion_text.png`, `data/confusion_audio.png`.",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()

    text_anchors = np.load(DATA_DIR / "anchors_text.npy")
    audio_anchors = np.load(DATA_DIR / "anchors_audio.npy")
    embeddings = np.load(DATA_DIR / "embeddings_cache.npy")
    meta = json.loads((DATA_DIR / "embeddings_meta.json").read_text())
    id_to_row = {sid: i for i, sid in enumerate(meta["segment_ids"])}

    val = pd.read_csv(SPLITS_DIR / "val.csv")
    val["cell"] = val["season"] + "_" + val["diel_bin"]
    val_rows = [id_to_row[sid] for sid in val["segment_id"]]
    val_emb = embeddings[val_rows]
    true_cells = val["cell"].tolist()

    metrics: dict[str, dict] = {}
    for anchors, label in [(text_anchors, "text"), (audio_anchors, "audio")]:
        m, cm = score(anchors, val_emb, true_cells, tau=args.tau)
        plot_confusion(cm, label, DATA_DIR / f"confusion_{label}.png")
        metrics[label] = m
        print(
            f"[{label}] top1={m['cell_top1']:.3f} top3={m['cell_top3']:.3f} "
            f"season={m['season_acc']:.3f} diel={m['diel_acc']:.3f}"
        )

    out = {
        "smoke_bar": {
            "season_acc": ">= 0.70",
            "diel_acc": ">= 0.55",
            "cell_top3": ">= 0.50",
        },
        "n_val": int(len(val)),
        "tau": args.tau,
        "metrics": metrics,
    }
    (ATTEMPT_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    write_report(out, ATTEMPT_DIR / "report.md")
    print(f"wrote {ATTEMPT_DIR / 'metrics.json'} and report.md")


if __name__ == "__main__":
    main()
