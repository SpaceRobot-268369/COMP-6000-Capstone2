"""Placeholder handler for: Layer B curated wind/rain assets

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
        "See layer_b/attempts/murphy__smoke_1__curated_assets/README.md for status."
    )
