"""Handler stub for the tiny Layer E-C classifier attempt.

This smoke attempt is intentionally offline-first. Use
``tiny_event_classifier.py train/predict/detect`` for the smoke workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib


def load(checkpoint_dir: Optional[Path], params: dict, extra: dict | None = None):
    if checkpoint_dir is None:
        return None
    model_path = checkpoint_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return {"model": joblib.load(model_path), "params": params}


def generate(state, seed: Optional[int] = None, **_ignored) -> dict:
    raise NotImplementedError(
        "Tiny Layer E-C classifier is an offline analysis model, not a generate endpoint. "
        "Run code/tiny_event_classifier.py predict or detect."
    )
