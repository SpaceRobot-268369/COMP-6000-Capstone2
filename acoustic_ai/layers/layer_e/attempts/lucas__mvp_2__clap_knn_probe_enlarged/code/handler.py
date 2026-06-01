"""E-A entry point for mvp_2 — k-NN retrieval + season probe + agreement gate.

`AmbientAnalyzer.analyze(audio_path)` returns the report from PLAN.md §3:
season comes from the trained probe; diel / hour / month + similar_clips from
the k-NN head; confidence / OOD from cell-match-vs-knn agreement plus the
probe margin. Long uploads are windowed and mean-pooled by the CLAP backbone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from clap_backbone import CLAPBackbone  # noqa: E402
from paths import (  # noqa: E402
    CELL_ORDER, DATA_DIR, PROBE_PATH, SEASON_ORDER, diel_of, season_of,
)


def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    s = (x / tau) - (x / tau).max()
    e = np.exp(s)
    return e / e.sum()


def _circular_decode(weights: np.ndarray, sin: np.ndarray, cos: np.ndarray, period: float, offset: float) -> float:
    s = float((weights * sin).sum())
    c = float((weights * cos).sum())
    angle = np.arctan2(s, c)
    if angle < 0:
        angle += 2 * np.pi
    return float(angle / (2 * np.pi) * period) + offset


def _load_probe() -> tuple[nn.Module, list[str]]:
    ckpt = torch.load(PROBE_PATH, map_location="cpu")
    if ckpt["arch"] == "mlp":
        probe: nn.Module = nn.Sequential(
            nn.Linear(ckpt["in_dim"], ckpt["hidden"]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(ckpt["hidden"], len(ckpt["season_order"])),
        )
    else:
        probe = nn.Linear(ckpt["in_dim"], len(ckpt["season_order"]))
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()
    return probe, ckpt["season_order"]


class AmbientAnalyzer:
    def __init__(self, k: int = 5, tau: float = 0.1, ood_conf_threshold: float = 0.25) -> None:
        self.k = k
        self.tau = tau
        self.ood_conf_threshold = ood_conf_threshold
        self.backbone = CLAPBackbone()
        self.index_emb = np.load(DATA_DIR / "index_embeddings.npy")
        self.index_meta = pd.read_csv(DATA_DIR / "index_meta.csv")
        self.anchors = np.load(DATA_DIR / "anchors_audio.npy")  # (16, D)
        self.probe, self.season_order = _load_probe()

    def analyze_embedding(self, q: np.ndarray) -> dict:
        # --- k-NN head: diel / hour / month / neighbours ---
        sims = self.index_emb @ q
        top_idx = np.argsort(-sims)[: self.k]
        top_sims = sims[top_idx]
        w = softmax(top_sims, tau=self.tau)
        nb = self.index_meta.iloc[top_idx]

        diel = Counter()
        for wi, d in zip(w, nb["diel_bin"]):
            diel[d] += float(wi)
        diel_hat = max(diel, key=diel.get)
        hour = _circular_decode(w, nb["hour_sin"].to_numpy(), nb["hour_cos"].to_numpy(), 24.0, 0.0)
        month = _circular_decode(w, nb["month_sin"].to_numpy(), nb["month_cos"].to_numpy(), 12.0, 1.0)

        # k-NN cell vote (for the agreement check)
        knn_cell_vote = Counter()
        for wi, (_, row) in zip(w, nb.iterrows()):
            knn_cell_vote[f"{row['season']}_{row['diel_bin']}"] += float(wi)
        knn_cell = max(knn_cell_vote, key=knn_cell_vote.get)

        # --- season probe head ---
        with torch.no_grad():
            logits = self.probe(torch.from_numpy(q).float().unsqueeze(0))
            p_season = torch.softmax(logits, dim=1).squeeze(0).numpy()
        season_idx = int(p_season.argmax())
        season_hat = self.season_order[season_idx]
        season_conf = float(p_season[season_idx])

        # --- agreement gate (cell-match anchor head vs k-NN cell vote) ---
        anchor_cell = CELL_ORDER[int((self.anchors @ q).argmax())]
        agreement = bool(anchor_cell == knn_cell)
        ood = bool(not agreement and season_conf < self.ood_conf_threshold)

        return {
            "estimated_conditions": {
                "season": season_hat,
                "diel_bin": diel_hat,
                "hour": round(hour, 2),
                "month": round(month, 2),
            },
            "season_source": "probe",
            "similar_clips": [
                {
                    "segment_id": str(row["segment_id"]),
                    "source_clip": str(row["source_clip"]),
                    "similarity": float(sim),
                }
                for sim, (_, row) in zip(top_sims, nb.iterrows())
            ],
            "confidence": round(float(top_sims.mean()), 4),
            "season_confidence": round(season_conf, 4),
            "head_agreement": agreement,
            "ood_flag": ood,
            "k": int(self.k),
            "tau": float(self.tau),
        }

    def analyze(self, audio_path: str | Path) -> dict:
        q = self.backbone.embed_audio([str(audio_path)])[0]
        return self.analyze_embedding(q)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    a = AmbientAnalyzer(k=args.k, tau=args.tau)
    print(json.dumps(a.analyze(args.audio_path), indent=2))


if __name__ == "__main__":
    main()
