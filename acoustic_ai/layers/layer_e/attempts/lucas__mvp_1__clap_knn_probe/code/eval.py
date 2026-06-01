"""Evaluate mvp_1 on the held-out val split (cached embeddings, no re-embed).

Reports the pass/fail gate:
- probe season acc vs k-NN season acc (the baseline the probe must beat)
- diel acc, hour/month circular MAE (k-NN head — must not regress vs smoke_3)
- agreement gate: season acc | agree vs | disagree

Writes metrics.json + report.md.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from handler import AmbientAnalyzer, softmax  # noqa: E402
from paths import DATA_DIR, SEASON_ORDER, SPLITS_DIR  # noqa: E402


def true_hour_month(row) -> tuple[float, float]:
    h = (math.atan2(row.hour_sin, row.hour_cos) % (2 * math.pi)) / (2 * math.pi) * 24
    m = (math.atan2(row.month_sin, row.month_cos) % (2 * math.pi)) / (2 * math.pi) * 12 + 1
    return h, m


def circ_err(est: float, true: float, period: float) -> float:
    return abs(((est - true + period / 2) % period) - period / 2)


def knn_season_vote(analyzer: AmbientAnalyzer, q: np.ndarray) -> str:
    sims = analyzer.index_emb @ q
    top_idx = np.argsort(-sims)[: analyzer.k]
    w = softmax(sims[top_idx], tau=analyzer.tau)
    nb = analyzer.index_meta.iloc[top_idx]
    vote = Counter()
    for wi, s in zip(w, nb["season"]):
        vote[s] += float(wi)
    return max(vote, key=vote.get)


def main() -> None:
    analyzer = AmbientAnalyzer()
    emb = np.load(DATA_DIR / "embeddings_cache.npy")
    meta = json.loads((DATA_DIR / "embeddings_meta.json").read_text())
    id_to_row = {sid: i for i, sid in enumerate(meta["segment_ids"])}
    val = pd.read_csv(SPLITS_DIR / "val.csv")

    n = len(val)
    probe_ok = knn_ok = diel_ok = 0
    hour_errs, month_errs = [], []
    agree_n = agree_correct = disagree_n = disagree_correct = 0
    per_season = {s: {"n": 0, "probe_ok": 0} for s in SEASON_ORDER}

    for _, row in val.iterrows():
        q = emb[id_to_row[row.segment_id]]
        rep = analyzer.analyze_embedding(q)
        ec = rep["estimated_conditions"]
        th, tm = true_hour_month(row)

        ps_ok = ec["season"] == row.season
        ks = knn_season_vote(analyzer, q)
        probe_ok += int(ps_ok)
        knn_ok += int(ks == row.season)
        diel_ok += int(ec["diel_bin"] == row.diel_bin)
        hour_errs.append(circ_err(ec["hour"], th, 24.0))
        month_errs.append(circ_err(ec["month"], tm, 12.0))

        per_season[row.season]["n"] += 1
        per_season[row.season]["probe_ok"] += int(ps_ok)

        if rep["head_agreement"]:
            agree_n += 1
            agree_correct += int(ps_ok)
        else:
            disagree_n += 1
            disagree_correct += int(ps_ok)

    metrics = {
        "n_val": n,
        "probe_season_acc": probe_ok / n,
        "knn_season_acc": knn_ok / n,
        "diel_acc": diel_ok / n,
        "hour_mae": float(np.mean(hour_errs)),
        "month_mae": float(np.mean(month_errs)),
        "agreement_rate": (agree_n) / n,
        "season_acc_given_agree": (agree_correct / agree_n) if agree_n else None,
        "season_acc_given_disagree": (disagree_correct / disagree_n) if disagree_n else None,
        "per_season_probe": {
            s: {"n": d["n"], "acc": (d["probe_ok"] / d["n"] if d["n"] else None)}
            for s, d in per_season.items()
        },
        "bar": {
            "probe_season_acc": "> 0.60 (stretch 0.70)",
            "diel_acc": ">= 0.683 (no regression vs smoke_3)",
            "hour_mae": "< 2.5",
            "month_mae": "< 2.0",
        },
    }
    (Path(__file__).resolve().parents[1] / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [
        "# mvp_1 clap_knn_probe - eval report",
        "",
        f"n_val = {n} (held-out, source-clip-disjoint from the train index).",
        "",
        "| metric | result | baseline / bar |",
        "|---|---|---|",
        f"| probe season acc | {metrics['probe_season_acc']:.3f} | k-NN {metrics['knn_season_acc']:.3f}; bar > 0.60 |",
        f"| diel acc | {metrics['diel_acc']:.3f} | smoke_3 0.683 |",
        f"| hour MAE (h) | {metrics['hour_mae']:.2f} | bar < 2.5 |",
        f"| month MAE (mo) | {metrics['month_mae']:.2f} | bar < 2.0 |",
        f"| agreement rate | {metrics['agreement_rate']:.3f} | - |",
        f"| season acc \\| agree | {metrics['season_acc_given_agree']} | vs disagree {metrics['season_acc_given_disagree']} |",
        "",
        "Per-season probe accuracy:",
        "",
        "| season | n | acc |",
        "|---|---|---|",
    ]
    for s in SEASON_ORDER:
        d = metrics["per_season_probe"][s]
        lines.append(f"| {s} | {d['n']} | {d['acc']:.3f} |" if d["acc"] is not None else f"| {s} | {d['n']} | - |")
    delta = metrics["probe_season_acc"] - metrics["knn_season_acc"]
    lines += ["", f"**Probe vs k-NN season: {delta:+.3f}.** "
              + ("Probe improves season." if delta > 0 else "Probe does NOT beat k-NN — keep k-NN season vote."), ""]
    (Path(__file__).resolve().parents[1] / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
