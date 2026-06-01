"""Score k-NN retrieval on the held-out query split.

Metrics (PLAN.md §5):
- Hour circular MAE (h), Month circular MAE (months)
- Season acc (weighted top-k vote), Diel acc
- Precision@k (fraction of top-k that share the true cell)
- Sweep over k ∈ {1, 3, 5, 10} and tau ∈ {0.05, 0.1, 0.2}
- Baselines: global-mean for hour/month, majority-class for season/diel

Outputs:
- metrics.json
- report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from paths import ATTEMPT_DIR, DATA_DIR, SPLITS_DIR  # noqa: E402


def softmax_rows(scores: np.ndarray, tau: float) -> np.ndarray:
    s = scores / tau
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def circular_mae_hours(pred_h: np.ndarray, true_h: np.ndarray) -> float:
    diff = (pred_h - true_h + 12.0) % 24.0 - 12.0
    return float(np.mean(np.abs(diff)))


def circular_mae_months(pred_m: np.ndarray, true_m: np.ndarray) -> float:
    diff = (pred_m - true_m + 6.0) % 12.0 - 6.0
    return float(np.mean(np.abs(diff)))


def decode_circular(weights: np.ndarray, sin: np.ndarray, cos: np.ndarray, period_units: float, offset: float = 0.0) -> np.ndarray:
    """Decode (sin, cos) → unit value via atan2. weights/sin/cos are (N, K)."""
    s = (weights * sin).sum(axis=1)
    c = (weights * cos).sum(axis=1)
    angle = np.arctan2(s, c)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    return angle / (2 * np.pi) * period_units + offset


def evaluate(
    query_emb: np.ndarray,
    query_df: pd.DataFrame,
    index_emb: np.ndarray,
    index_meta: pd.DataFrame,
    k: int,
    tau: float,
) -> dict:
    sims = query_emb @ index_emb.T  # (N_q, N_i)
    top_idx = np.argsort(-sims, axis=1)[:, :k]  # (N_q, k)
    top_sims = np.take_along_axis(sims, top_idx, axis=1)  # (N_q, k)
    weights = softmax_rows(top_sims, tau=tau)  # (N_q, k)

    season_arr = index_meta["season"].to_numpy()
    diel_arr = index_meta["diel_bin"].to_numpy()
    hour_sin_arr = index_meta["hour_sin"].to_numpy()
    hour_cos_arr = index_meta["hour_cos"].to_numpy()
    month_sin_arr = index_meta["month_sin"].to_numpy()
    month_cos_arr = index_meta["month_cos"].to_numpy()

    # Categorical: weighted majority vote per query
    season_preds = []
    diel_preds = []
    for i in range(len(query_df)):
        nb_season = season_arr[top_idx[i]]
        nb_diel = diel_arr[top_idx[i]]
        s = Counter()
        d = Counter()
        for w, ss, dd in zip(weights[i], nb_season, nb_diel):
            s[ss] += float(w)
            d[dd] += float(w)
        season_preds.append(max(s, key=s.get))
        diel_preds.append(max(d, key=d.get))

    season_acc = float(np.mean([p == t for p, t in zip(season_preds, query_df["season"])]))
    diel_acc = float(np.mean([p == t for p, t in zip(diel_preds, query_df["diel_bin"])]))

    # Continuous: circular decode
    nb_hour_sin = hour_sin_arr[top_idx]
    nb_hour_cos = hour_cos_arr[top_idx]
    pred_hour = decode_circular(weights, nb_hour_sin, nb_hour_cos, period_units=24.0)
    nb_month_sin = month_sin_arr[top_idx]
    nb_month_cos = month_cos_arr[top_idx]
    pred_month = decode_circular(weights, nb_month_sin, nb_month_cos, period_units=12.0, offset=1.0)

    # True hour/month from sin/cos in query rows (they're already stored that way)
    true_hour = (
        (np.arctan2(query_df["hour_sin"].to_numpy(), query_df["hour_cos"].to_numpy()) % (2 * np.pi))
        / (2 * np.pi)
        * 24.0
    )
    true_month = (
        (np.arctan2(query_df["month_sin"].to_numpy(), query_df["month_cos"].to_numpy()) % (2 * np.pi))
        / (2 * np.pi)
        * 12.0
        + 1.0
    )

    hour_mae = circular_mae_hours(pred_hour, true_hour)
    month_mae = circular_mae_months(pred_month, true_month)

    # Precision@k: fraction of top-k that share the true cell
    true_cells = (query_df["season"] + "_" + query_df["diel_bin"]).to_numpy()
    nb_cells = (np.array([season_arr[idx] for idx in top_idx]) + "_" + np.array([diel_arr[idx] for idx in top_idx]))
    p_at_k = float(np.mean([np.mean(nb_cells[i] == true_cells[i]) for i in range(len(query_df))]))

    return {
        "k": int(k),
        "tau": float(tau),
        "season_acc": season_acc,
        "diel_acc": diel_acc,
        "hour_mae_h": hour_mae,
        "month_mae_mo": month_mae,
        "precision_at_k": p_at_k,
        "mean_top_sim": float(top_sims.mean()),
    }


def compute_baselines(query_df: pd.DataFrame, index_meta: pd.DataFrame) -> dict:
    # Majority class
    season_majority = index_meta["season"].mode().iloc[0]
    diel_majority = index_meta["diel_bin"].mode().iloc[0]
    season_acc = float(np.mean(query_df["season"] == season_majority))
    diel_acc = float(np.mean(query_df["diel_bin"] == diel_majority))

    # Global-mean predictor for hour/month (atan2 of mean sin/cos over index)
    def _decode_global(sin_arr, cos_arr, period, offset=0.0):
        s = float(sin_arr.mean())
        c = float(cos_arr.mean())
        a = np.arctan2(s, c)
        if a < 0:
            a += 2 * np.pi
        return a / (2 * np.pi) * period + offset

    pred_hour = _decode_global(index_meta["hour_sin"], index_meta["hour_cos"], 24.0)
    pred_month = _decode_global(index_meta["month_sin"], index_meta["month_cos"], 12.0, offset=1.0)
    true_hour = (
        (np.arctan2(query_df["hour_sin"], query_df["hour_cos"]) % (2 * np.pi))
        / (2 * np.pi)
        * 24.0
    )
    true_month = (
        (np.arctan2(query_df["month_sin"], query_df["month_cos"]) % (2 * np.pi))
        / (2 * np.pi)
        * 12.0
        + 1.0
    )
    hour_mae = circular_mae_hours(np.full(len(query_df), pred_hour), true_hour.to_numpy())
    month_mae = circular_mae_months(np.full(len(query_df), pred_month), true_month.to_numpy())

    return {
        "season_majority": season_majority,
        "diel_majority": diel_majority,
        "season_acc": season_acc,
        "diel_acc": diel_acc,
        "hour_mae_h": hour_mae,
        "month_mae_mo": month_mae,
    }


def write_report(out: dict, report_path: Path) -> None:
    bar = out["smoke_bar"]
    base = out["baselines"]
    lines = [
        "# smoke_3 clap_knn_env — eval report",
        "",
        f"n_query = {out['n_query']} segments (source-clip-disjoint from index of size {out['n_index']})",
        "",
        "## Baselines",
        f"- season majority `{base['season_majority']}` → acc {base['season_acc']:.3f}",
        f"- diel majority `{base['diel_majority']}` → acc {base['diel_acc']:.3f}",
        f"- hour global-mean → MAE {base['hour_mae_h']:.2f} h",
        f"- month global-mean → MAE {base['month_mae_mo']:.2f} months",
        "",
        "## Smoke bar",
        f"- season_acc {bar['season_acc']}",
        f"- diel_acc {bar['diel_acc']}",
        f"- hour MAE {bar['hour_mae_h']}",
        f"- month MAE {bar['month_mae_mo']}",
        "",
        "## Sweep results (k × tau)",
        "",
        "| k | tau | season | diel | hour MAE (h) | month MAE (mo) | P@k |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in out["sweep"]:
        lines.append(
            f"| {row['k']} | {row['tau']} | {row['season_acc']:.3f} | {row['diel_acc']:.3f} "
            f"| {row['hour_mae_h']:.2f} | {row['month_mae_mo']:.2f} | {row['precision_at_k']:.3f} |"
        )
    best = out["best"]
    lines += [
        "",
        f"**Best (by hour MAE, ties broken by season acc):** k={best['k']}, tau={best['tau']}",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = parser.parse_args()

    embeddings = np.load(DATA_DIR / "embeddings_cache.npy")
    cache_meta = json.loads((DATA_DIR / "embeddings_meta.json").read_text())
    id_to_row = {sid: i for i, sid in enumerate(cache_meta["segment_ids"])}

    query_df = pd.read_csv(SPLITS_DIR / "query.csv")
    query_rows = [id_to_row[sid] for sid in query_df["segment_id"]]
    query_emb = embeddings[query_rows]
    index_emb = np.load(DATA_DIR / "index_embeddings.npy")
    index_meta = pd.read_csv(DATA_DIR / "index_meta.csv")

    sweep = []
    for k in args.ks:
        for tau in args.taus:
            m = evaluate(query_emb, query_df, index_emb, index_meta, k=k, tau=tau)
            sweep.append(m)
            print(
                f"[k={k:2d} tau={tau:.2f}] season={m['season_acc']:.3f} diel={m['diel_acc']:.3f} "
                f"hourMAE={m['hour_mae_h']:.2f} monthMAE={m['month_mae_mo']:.2f} P@k={m['precision_at_k']:.3f}"
            )

    best = min(sweep, key=lambda r: (r["hour_mae_h"], -r["season_acc"]))
    baselines = compute_baselines(query_df, index_meta)

    out = {
        "smoke_bar": {
            "season_acc": ">= 0.70",
            "diel_acc": ">= 0.55",
            "hour_mae_h": "< 3.0",
            "month_mae_mo": "< 2.0",
        },
        "n_query": int(len(query_df)),
        "n_index": int(len(index_meta)),
        "baselines": baselines,
        "sweep": sweep,
        "best": best,
    }
    (ATTEMPT_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    write_report(out, ATTEMPT_DIR / "report.md")
    print(f"\nwrote {ATTEMPT_DIR / 'metrics.json'} and report.md")


if __name__ == "__main__":
    main()
