"""Technical audio metrics used by Layer D format and mixing experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def audio_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Return format and amplitude metrics without modifying the audio."""

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2:
        raise ValueError("audio must have shape (frames,) or (frames, channels)")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    finite = np.isfinite(samples)
    finite_audio = samples[finite]
    if finite_audio.size:
        rms = float(np.sqrt(np.mean(np.square(finite_audio, dtype=np.float64))))
        peak = float(np.max(np.abs(finite_audio)))
        dc_offset = float(np.mean(finite_audio, dtype=np.float64))
        clipping_ratio = float(np.mean(np.abs(finite_audio) >= 1.0))
    else:
        rms = peak = dc_offset = clipping_ratio = 0.0

    return {
        "sample_rate": int(sample_rate),
        "channels": int(samples.shape[1]),
        "frames": int(samples.shape[0]),
        "duration_s": float(samples.shape[0] / sample_rate),
        "rms": rms,
        "rms_dbfs": _amplitude_to_dbfs(rms),
        "peak": peak,
        "peak_dbfs": _amplitude_to_dbfs(peak),
        "clipping_ratio": clipping_ratio,
        "dc_offset": dc_offset,
        "nan_count": int(np.isnan(samples).sum()),
        "inf_count": int(np.isinf(samples).sum()),
    }


def _amplitude_to_dbfs(value: float) -> float | None:
    if value <= 0:
        return None
    return float(20.0 * math.log10(value))
