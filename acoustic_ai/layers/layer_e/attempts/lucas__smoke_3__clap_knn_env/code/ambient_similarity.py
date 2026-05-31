"""E-A entry point — locate a query clip in soundscape space and read env off neighbours.

`query(audio_path)` returns the report shape from PLAN.md §3:
    {
      "estimated_conditions": {"season": ..., "diel_bin": ..., "hour": ..., "month": ...},
      "similar_clips": [{"segment_id": ..., "source_clip": ..., "similarity": ...}],
      "confidence": ...
    }
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
from clap_backbone import CLAPBackbone  # noqa: E402
from paths import DATA_DIR  # noqa: E402


def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    s = x / tau
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def circular_decode_hour(weights: np.ndarray, sin: np.ndarray, cos: np.ndarray) -> float:
    """Returns hour ∈ [0, 24) from weighted (sin, cos) blends."""
    s = float((weights * sin).sum())
    c = float((weights * cos).sum())
    angle = np.arctan2(s, c)  # in [-π, π]
    if angle < 0:
        angle += 2 * np.pi
    return float(angle / (2 * np.pi) * 24.0)


def circular_decode_month(weights: np.ndarray, sin: np.ndarray, cos: np.ndarray) -> float:
    """Returns month ∈ [1, 13) from weighted (sin, cos) blends."""
    s = float((weights * sin).sum())
    c = float((weights * cos).sum())
    angle = np.arctan2(s, c)
    if angle < 0:
        angle += 2 * np.pi
    return float(angle / (2 * np.pi) * 12.0) + 1.0


class AmbientRetriever:
    def __init__(self, k: int = 5, tau: float = 0.1) -> None:
        self.k = k
        self.tau = tau
        self.backbone = CLAPBackbone()
        self.index_emb = np.load(DATA_DIR / "index_embeddings.npy")
        self.index_meta = pd.read_csv(DATA_DIR / "index_meta.csv")
        for col in ("hour_sin", "hour_cos", "month_sin", "month_cos"):
            self.index_meta[col] = self.index_meta[col].astype(np.float32)

    def query(self, audio_path: str | Path) -> dict:
        emb = self.backbone.embed_audio([str(audio_path)])[0]  # (D,)
        sims = self.index_emb @ emb  # (N,)
        top_idx = np.argsort(-sims)[: self.k]
        top_sims = sims[top_idx]
        w = softmax(top_sims, tau=self.tau)
        nb = self.index_meta.iloc[top_idx]

        season = Counter()
        diel = Counter()
        for wi, (_, row) in zip(w, nb.iterrows()):
            season[row["season"]] += float(wi)
            diel[row["diel_bin"]] += float(wi)

        hour = circular_decode_hour(
            w,
            nb["hour_sin"].to_numpy(),
            nb["hour_cos"].to_numpy(),
        )
        month = circular_decode_month(
            w,
            nb["month_sin"].to_numpy(),
            nb["month_cos"].to_numpy(),
        )
        return {
            "estimated_conditions": {
                "season": max(season, key=season.get),
                "diel_bin": max(diel, key=diel.get),
                "hour": round(hour, 2),
                "month": round(month, 2),
            },
            "similar_clips": [
                {
                    "segment_id": str(row["segment_id"]),
                    "source_clip": str(row["source_clip"]),
                    "similarity": float(sim),
                }
                for sim, (_, row) in zip(top_sims, nb.iterrows())
            ],
            "confidence": float(top_sims.mean()),
            "k": int(self.k),
            "tau": float(self.tau),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    r = AmbientRetriever(k=args.k, tau=args.tau)
    print(json.dumps(r.query(args.audio_path), indent=2))


if __name__ == "__main__":
    main()
