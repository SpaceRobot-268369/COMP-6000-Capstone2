"""E-A entry point — fuse cell-anchor (smoke_2) + k-NN (smoke_3) heads on one shared query embedding.

Returns the superset report from PLAN.md §3 of smoke_4: discrete cell label
+ caption, continuous env estimate + similar clips, agreement and a simple
OOD flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from clap_backbone import CLAPBackbone  # noqa: E402
from paths import CELL_ORDER, DATA_DIR, PROD_ATTEMPT_KEY, REGISTRY, diel_of, season_of  # noqa: E402


def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    s = x / tau
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def normalize_probs(p: np.ndarray) -> np.ndarray:
    return p / max(float(p.sum()), 1e-12)


class AmbientFusedAnalyzer:
    def __init__(
        self,
        head_a_variant: str = "audio",
        k: int = 5,
        tau_a: float = 0.1,
        tau_b: float = 0.1,
        w_a: float = 0.5,
        w_b: float = 0.5,
        ood_conf_threshold: float = 0.25,
    ) -> None:
        if head_a_variant not in ("audio", "text"):
            raise ValueError(f"head_a_variant must be 'audio' or 'text', got {head_a_variant!r}")
        self.head_a_variant = head_a_variant
        self.k = k
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.w_a = w_a
        self.w_b = w_b
        self.ood_conf_threshold = ood_conf_threshold

        self.backbone = CLAPBackbone()
        self.anchors = np.load(DATA_DIR / f"anchors_{head_a_variant}.npy")  # (16, D)
        self.index_emb = np.load(DATA_DIR / "index_embeddings.npy")
        self.index_meta = pd.read_csv(DATA_DIR / "index_meta.csv")
        reg = yaml.safe_load(REGISTRY.read_text())
        cells = reg["layers"]["layer_a"]["attempts"][PROD_ATTEMPT_KEY]["params"]["cells"]
        self.prompts = {name: body["prompt"] for name, body in cells.items()}

    def analyze(self, audio_path: str | Path) -> dict:
        q = self.backbone.embed_audio([str(audio_path)])[0]

        # Head A — cell anchors
        sims_a = self.anchors @ q
        p_a = softmax(sims_a, tau=self.tau_a)

        # Head B — k-NN over index, project to per-cell posterior
        sims_b = self.index_emb @ q
        top_idx = np.argsort(-sims_b)[: self.k]
        top_sims = sims_b[top_idx]
        nb_w = softmax(top_sims, tau=self.tau_b)
        nb = self.index_meta.iloc[top_idx]
        nb_cells = (nb["season"] + "_" + nb["diel_bin"]).to_numpy()

        p_b = np.zeros(16, dtype=np.float64)
        for w, cell in zip(nb_w, nb_cells):
            p_b[CELL_ORDER.index(str(cell))] += float(w)
        p_b = normalize_probs(p_b)

        # Fusion
        p_fused = normalize_probs(self.w_a * p_a + self.w_b * p_b)
        cell_hat_idx = int(np.argmax(p_fused))
        cell_hat = CELL_ORDER[cell_hat_idx]

        cell_a = CELL_ORDER[int(np.argmax(p_a))]
        cell_b = CELL_ORDER[int(np.argmax(p_b))]
        agreement = bool(cell_a == cell_b)
        confidence = float(p_fused[cell_hat_idx])
        ood = bool(not agreement and confidence < self.ood_conf_threshold)

        # Continuous env from neighbour blend (head B)
        s = float((nb_w * nb["hour_sin"].to_numpy()).sum())
        c = float((nb_w * nb["hour_cos"].to_numpy()).sum())
        h_angle = np.arctan2(s, c)
        if h_angle < 0:
            h_angle += 2 * np.pi
        hour = float(h_angle / (2 * np.pi) * 24.0)
        s = float((nb_w * nb["month_sin"].to_numpy()).sum())
        c = float((nb_w * nb["month_cos"].to_numpy()).sum())
        m_angle = np.arctan2(s, c)
        if m_angle < 0:
            m_angle += 2 * np.pi
        month = float(m_angle / (2 * np.pi) * 12.0 + 1.0)

        # Continuous env from blend; season/diel from fused argmax
        return {
            "predicted_cell": cell_hat,
            "season": season_of(cell_hat),
            "diel": diel_of(cell_hat),
            "caption": self.prompts[cell_hat],
            "estimated_conditions": {
                "season": season_of(cell_hat),
                "diel_bin": diel_of(cell_hat),
                "hour": round(hour, 2),
                "month": round(month, 2),
            },
            "similar_clips": [
                {
                    "segment_id": str(nb.iloc[i]["segment_id"]),
                    "source_clip": str(nb.iloc[i]["source_clip"]),
                    "similarity": float(top_sims[i]),
                }
                for i in range(len(nb))
            ],
            "head_a_cell": cell_a,
            "head_b_cell": cell_b,
            "head_agreement": agreement,
            "confidence": confidence,
            "ood_flag": ood,
            "topk": [
                {"cell": CELL_ORDER[i], "score": float(p_fused[i])}
                for i in np.argsort(-p_fused)[:3]
            ],
            "params": {
                "head_a_variant": self.head_a_variant,
                "k": self.k,
                "tau_a": self.tau_a,
                "tau_b": self.tau_b,
                "w_a": self.w_a,
                "w_b": self.w_b,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str)
    parser.add_argument("--variant", choices=["audio", "text"], default="audio")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    a = AmbientFusedAnalyzer(head_a_variant=args.variant, k=args.k)
    print(json.dumps(a.analyze(args.audio_path), indent=2))


if __name__ == "__main__":
    main()
