"""Placeholder handler for: VAE baseline (encoder-only; use /analysis instead)

This attempt is registered for the dropdown but does not yet implement a
generate() flow. See attempt README for status.
"""

from pathlib import Path
from typing import Optional


def load(checkpoint_dir: Optional[Path], params: dict, extra: dict | None = None):
    return None


def generate(state, seed: Optional[int] = None, **_ignored) -> dict:
    raise NotImplementedError(
        "This attempt has no generate() implementation. "
        "See layer_a/attempts/lucas__smoke_4__vae_baseline/README.md for status."
    )
