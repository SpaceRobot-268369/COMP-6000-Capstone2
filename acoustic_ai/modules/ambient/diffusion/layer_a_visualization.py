"""Shared Layer A mel-spectrogram rendering utilities.

The dev API and AudioLDM2 CLI both use this module so the same waveform is
visualized with identical mel parameters and PNG rendering settings.
"""

from __future__ import annotations

import io

import numpy as np

from modules.ambient.preprocess import SPEC_CFG, waveform_to_melspec

LAYER_A_SPEC_TITLE = "Layer A - Ambient Bed Spectrogram"
LAYER_A_SPEC_FIGSIZE = (12, 4)
LAYER_A_SPEC_DPI = 110
LAYER_A_SPEC_CMAP = "magma"


def waveform_to_layer_a_mel_db(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert an AudioLDM2 waveform to the project Layer A log-mel format."""
    cfg = {
        **SPEC_CFG,
        "sample_rate": int(sample_rate),
        "fmax": min(float(SPEC_CFG["fmax"]), float(sample_rate) / 2.0),
    }
    return waveform_to_melspec(waveform, cfg=cfg)


def render_layer_a_mel_png_bytes(mel_db: np.ndarray, duration_s: float) -> bytes:
    """Render a Layer A mel-spectrogram PNG with shared visual settings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=LAYER_A_SPEC_FIGSIZE)
    img = ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        extent=[0, duration_s, 0, SPEC_CFG["n_mels"]],
        cmap=LAYER_A_SPEC_CMAP,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_title(LAYER_A_SPEC_TITLE)
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=LAYER_A_SPEC_DPI)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

