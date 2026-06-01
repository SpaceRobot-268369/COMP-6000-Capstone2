"""Score Head A alone, Head B alone, and Fused on one shared val/query split.

Tests the two PLAN.md §1 hypotheses:
1. Fusion ≥ best single head on cell accuracy.
2. Head agreement correlates with correctness (cheap confidence / OOD signal).

Outputs:
- metrics.json
- report.md
- data/confusion_{a,b,fused}.png
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


def softmax_rows(scores: np.ndarray, tau: float) -> np.ndarray:
    s = scores / tau
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def head_a_posteriors(val_emb: np.ndarray, anchors: np.ndarray, tau: float) -> np.ndarray:
    sims = val_emb @ anchors.T
    return softmax_rows(sims, tau=tau)


def head_b_posteriors(
    val_emb: np.ndarray, index_emb: np.ndarray, index_meta: pd.DataFrame, k: int, tau: float
) -> np.ndarray:
    sims = val_emb @ index_emb.T  # (N_q, N_i)
    top_idx = np.argsort(-sims, axis=1)[:, :k]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)
    weights = softmax_rows(top_sims, tau=tau)

    nb_cells = (index_meta["season"] + "_" + index_meta["diel_bin"]).to_numpy()
    p_b = np.zeros((val_emb.shape[0], 16), dtype=np.float64)
    for i in range(val_emb.shape[0]):
        for w, cell_idx in zip(weights[i], top_idx[i]):
            cell = str(nb_cells[cell_idx])
            p_b[i, CELL_ORDER.index(cell)] += float(w)
        s = p_b[i].sum()
        if s > 0:
            p_b[i] /= s
    return p_b


def metrics_from_posteriors(p: np.ndarray, true_cells: list[str]) -> tuple[dict, np.ndarray]:
    pred_idx = p.argmax(axis=1)
    preds = [CELL_ORDER[int(i)] for i in pred_idx]
    top3 = np.argsort(-p, axis=1)[:, :3]
    top3_cells = [[CELL_ORDER[int(j)] for j in row] for row in top3]

    cell_top1 = float(np.mean([a == b for a, b in zip(preds, true_cells)]))
    cell_top3 = float(np.mean([t in row for row, t in zip(top3_cells, true_cells)]))
    season_acc = float(np.mean([season_of(a) == season_of(b) for a, b in zip(preds, true_cells)]))
    diel_acc = float(np.mean([diel_of(a) == diel_of(b) for a, b in zip(preds, true_cells)]))
    mean_conf = float(p.max(axis=1).mean())

    cm = np.zeros((16, 16), dtype=np.int64)
    for t, pcell in zip(true_cells, preds):
        cm[CELL_ORDER.index(t), CELL_ORDER.index(pcell)] += 1

    return {
        "cell_top1": cell_top1,
        "cell_top3": cell_top3,
        "season_acc": season_acc,
        "diel_acc": diel_acc,
        "mean_confidence": mean_conf,
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
    ax.set_title(f"smoke_4 fused — {label}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(out: dict, report_path: Path) -> None:
    lines = [
        "# smoke_4 cell+knn fused — eval report",
        "",
        f"n_val = {out['n_val']} segments (source-clip-disjoint).",
        f"params: head_a_variant={out['params']['head_a_variant']}, "
        f"k={out['params']['k']}, tau_a={out['params']['tau_a']}, "
        f"tau_b={out['params']['tau_b']}, w_a={out['params']['w_a']}, "
        f"w_b={out['params']['w_b']}.",
        "",
        "## Hypothesis 1 — fusion ≥ best single head",
        "",
        "| metric | head A (cell) | head B (knn) | fused |",
        "|---|---|---|---|",
    ]
    a = out["metrics"]["a"]
    b = out["metrics"]["b"]
    f = out["metrics"]["fused"]
    for key, name in [
        ("cell_top1", "cell top-1"),
        ("cell_top3", "cell top-3"),
        ("season_acc", "season acc"),
        ("diel_acc", "diel acc"),
        ("mean_confidence", "mean confidence"),
    ]:
        lines.append(f"| {name} | {a[key]:.3f} | {b[key]:.3f} | {f[key]:.3f} |")

    h2 = out["agreement"]
    lines += [
        "",
        "## Hypothesis 2 — head agreement signals correctness",
        "",
        f"- agreement rate: **{h2['agreement_rate']:.3f}**",
        f"- accuracy | agree:    {h2['acc_given_agree']:.3f} (n={h2['n_agree']})",
        f"- accuracy | disagree: {h2['acc_given_disagree']:.3f} (n={h2['n_disagree']})",
        f"- error-correlation between A and B: {h2['error_correlation']:.3f}",
        "",
        "Confusion matrices: `data/confusion_{a,b,fused}.png`.",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-a-variant", choices=["audio", "text"], default="audio")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--tau-a", type=float, default=0.1)
    parser.add_argument("--tau-b", type=float, default=0.1)
    parser.add_argument("--w-a", type=float, default=0.5)
    parser.add_argument("--w-b", type=float, default=0.5)
    args = parser.parse_args()

    embeddings = np.load(DATA_DIR / "embeddings_cache.npy")
    cache_meta = json.loads((DATA_DIR / "embeddings_meta.json").read_text())
    id_to_row = {sid: i for i, sid in enumerate(cache_meta["segment_ids"])}

    # The val split equals the query split for this attempt: same source-clip
    # disjoint partition feeds both heads on identical rows.
    val = pd.read_csv(SPLITS_DIR / "val.csv")
    val["cell"] = val["season"] + "_" + val["diel_bin"]
    val_rows = [id_to_row[sid] for sid in val["segment_id"]]
    val_emb = embeddings[val_rows]
    true_cells = val["cell"].tolist()

    anchors = np.load(DATA_DIR / f"anchors_{args.head_a_variant}.npy")
    index_emb = np.load(DATA_DIR / "index_embeddings.npy")
    index_meta = pd.read_csv(DATA_DIR / "index_meta.csv")

    p_a = head_a_posteriors(val_emb, anchors, tau=args.tau_a)
    p_b = head_b_posteriors(val_emb, index_emb, index_meta, k=args.k, tau=args.tau_b)
    p_fused = args.w_a * p_a + args.w_b * p_b
    p_fused = p_fused / p_fused.sum(axis=1, keepdims=True)

    m_a, cm_a = metrics_from_posteriors(p_a, true_cells)
    m_b, cm_b = metrics_from_posteriors(p_b, true_cells)
    m_f, cm_f = metrics_from_posteriors(p_fused, true_cells)
    plot_confusion(cm_a, "head A (cell)", DATA_DIR / "confusion_a.png")
    plot_confusion(cm_b, "head B (knn)", DATA_DIR / "confusion_b.png")
    plot_confusion(cm_f, "fused", DATA_DIR / "confusion_fused.png")

    pred_a = [CELL_ORDER[int(i)] for i in p_a.argmax(axis=1)]
    pred_b = [CELL_ORDER[int(i)] for i in p_b.argmax(axis=1)]
    agree_mask = np.array([a == b for a, b in zip(pred_a, pred_b)])
    correct_fused = np.array([CELL_ORDER[int(p_fused[i].argmax())] == true_cells[i] for i in range(len(val))])
    n_agree = int(agree_mask.sum())
    n_disagree = int((~agree_mask).sum())
    acc_agree = float(correct_fused[agree_mask].mean()) if n_agree else 0.0
    acc_disagree = float(correct_fused[~agree_mask].mean()) if n_disagree else 0.0

    err_a = np.array([a != t for a, t in zip(pred_a, true_cells)], dtype=int)
    err_b = np.array([a != t for a, t in zip(pred_b, true_cells)], dtype=int)
    if err_a.std() > 0 and err_b.std() > 0:
        err_corr = float(np.corrcoef(err_a, err_b)[0, 1])
    else:
        err_corr = float("nan")

    out = {
        "params": {
            "head_a_variant": args.head_a_variant,
            "k": args.k,
            "tau_a": args.tau_a,
            "tau_b": args.tau_b,
            "w_a": args.w_a,
            "w_b": args.w_b,
        },
        "n_val": int(len(val)),
        "metrics": {"a": m_a, "b": m_b, "fused": m_f},
        "agreement": {
            "agreement_rate": float(agree_mask.mean()),
            "n_agree": n_agree,
            "n_disagree": n_disagree,
            "acc_given_agree": acc_agree,
            "acc_given_disagree": acc_disagree,
            "error_correlation": err_corr,
        },
        "smoke_bar": {
            "fused_top1": ">= max(head_a.top1, head_b.top1)",
            "acc_given_agree": "> acc_given_disagree",
        },
    }
    (ATTEMPT_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    write_report(out, ATTEMPT_DIR / "report.md")
    print(
        f"A.top1={m_a['cell_top1']:.3f}  B.top1={m_b['cell_top1']:.3f}  "
        f"Fused.top1={m_f['cell_top1']:.3f}  agree_rate={float(agree_mask.mean()):.3f}  "
        f"acc|agree={acc_agree:.3f}  acc|disagree={acc_disagree:.3f}"
    )
    print(f"wrote {ATTEMPT_DIR / 'metrics.json'} and report.md")


if __name__ == "__main__":
    main()
