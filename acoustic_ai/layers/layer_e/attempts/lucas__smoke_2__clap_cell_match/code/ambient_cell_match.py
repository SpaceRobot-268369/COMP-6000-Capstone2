"""E-A entry point — classify a single audio clip into one of the 16 cells.

Returns the "reverse prompt" report described in the PLAN: the cell label,
the locked generation caption for that cell, a confidence score, and the
top-3 candidates. Two anchor variants are exposed: text (pure zero-shot)
and audio (train-set prototypes); they share the same query embedding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from clap_backbone import CLAPBackbone  # noqa: E402
from paths import CELL_ORDER, DATA_DIR, PROD_ATTEMPT_KEY, REGISTRY, diel_of, season_of  # noqa: E402


def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    s = x / tau
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


class CellMatcher:
    def __init__(self, variant: str = "audio", tau: float = 0.1) -> None:
        if variant not in ("text", "audio"):
            raise ValueError(f"variant must be 'text' or 'audio', got {variant!r}")
        self.variant = variant
        self.tau = tau
        self.backbone = CLAPBackbone()
        self.anchors = np.load(DATA_DIR / f"anchors_{variant}.npy")
        reg = yaml.safe_load(REGISTRY.read_text())
        cells = reg["layers"]["layer_a"]["attempts"][PROD_ATTEMPT_KEY]["params"]["cells"]
        self.prompts = {name: body["prompt"] for name, body in cells.items()}

    def classify(self, audio_path: str | Path) -> dict:
        emb = self.backbone.embed_audio([str(audio_path)])[0]
        sims = self.anchors @ emb
        probs = softmax(sims, tau=self.tau)
        order = np.argsort(-probs)[:3]
        pred = CELL_ORDER[int(order[0])]
        return {
            "predicted_cell": pred,
            "season": season_of(pred),
            "diel": diel_of(pred),
            "caption": self.prompts[pred],
            "confidence": float(probs[int(order[0])]),
            "topk": [
                {"cell": CELL_ORDER[int(i)], "score": float(probs[int(i)])}
                for i in order
            ],
            "variant": self.variant,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str)
    parser.add_argument("--variant", choices=["audio", "text"], default="audio")
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    matcher = CellMatcher(variant=args.variant, tau=args.tau)
    print(json.dumps(matcher.classify(args.audio_path), indent=2))


if __name__ == "__main__":
    main()
