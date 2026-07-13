"""Balanced weather head for Layer E-B MVP-3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from layers.layer_e.attempts.liting__mvp_1__panns_weather_baseline.code import weather_detector as mvp1


RAIN_CLASSES = ("none", "light", "moderate", "heavy")
WIND_CLASSES = ("none", "light", "moderate", "strong")

PANNS_LABEL_KEYS = (
    ("rain", "Rain"),
    ("rain", "Raindrop"),
    ("rain", "Rain on surface"),
    ("wind", "Wind"),
    ("wind", "Rustling leaves"),
    ("wind", "Wind noise"),
    ("thunder", "Thunderstorm"),
    ("thunder", "Thunder"),
)

DSP_FEATURE_NAMES = (
    "duration_s",
    "rms_dbfs",
    "peak_dbfs",
    "low_band_ratio_20_250",
    "low_mid_band_ratio_250_1000",
    "rain_band_ratio_2000_8000",
    "high_band_ratio_4000_10000",
    "spectral_centroid_hz",
    "spectral_flatness",
    "onset_strength_mean",
    "onset_strength_p95",
    "zero_crossing_rate",
    "rms_modulation",
)

FEATURE_NAMES = (
    "panns_rain_score",
    "panns_wind_score",
    "panns_thunder_score",
    *(f"panns_label_{component}_{label.lower().replace(' ', '_')}" for component, label in PANNS_LABEL_KEYS),
    *DSP_FEATURE_NAMES,
)


def extract_feature_vector(audio_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract frozen PANNs/DSP features for one audio file."""
    path = Path(audio_path)
    dsp = mvp1.smoke.extract_weather_features(path)
    panns = mvp1.score_with_panns(path)

    values: dict[str, float] = {
        "panns_rain_score": float(panns.scores.get("rain", 0.0)),
        "panns_wind_score": float(panns.scores.get("wind", 0.0)),
        "panns_thunder_score": float(panns.scores.get("thunder", 0.0)),
    }
    for component, label in PANNS_LABEL_KEYS:
        key = f"panns_label_{component}_{label.lower().replace(' ', '_')}"
        values[key] = float(panns.labels.get(component, {}).get(label, 0.0))
    for name in DSP_FEATURE_NAMES:
        values[name] = float(dsp.get(name, 0.0))

    vector = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    evidence = {
        "panns_available": panns.available,
        "panns_status": panns.reason,
        "panns_scores": panns.scores,
        "panns_labels": panns.labels,
        "dsp_features": dsp,
    }
    return vector, evidence


def normalise_label(label: str | None, classes: tuple[str, ...]) -> str:
    value = (label or "none").strip().lower()
    if value == "medium":
        value = "moderate"
    if classes == WIND_CLASSES and value == "heavy":
        value = "strong"
    return value if value in classes else "none"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(), 1e-12)


def predict_with_checkpoint(features: np.ndarray, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Run saved linear/MLP heads without needing PyTorch at inference time."""
    mu = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    sigma = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    x = (features.astype(np.float32) - mu) / np.maximum(sigma, 1e-6)

    rain = _predict_component(x, checkpoint["rain_head"], RAIN_CLASSES)
    wind = _predict_component(x, checkpoint["wind_head"], WIND_CLASSES)
    confidence = round(float((rain["confidence"] + wind["confidence"]) / 2.0), 3)
    return {
        "rain_intensity": rain["label"],
        "wind_intensity": wind["label"],
        "thunder_intensity": "none",
        "confidence": confidence,
        "component_confidence": {
            "rain": rain["confidence"],
            "wind": wind["confidence"],
            "thunder": 0.0,
        },
        "probabilities": {
            "rain": rain["probabilities"],
            "wind": wind["probabilities"],
            "thunder": {"none": 1.0},
        },
    }


def _predict_component(x: np.ndarray, head: dict[str, Any], classes: tuple[str, ...]) -> dict[str, Any]:
    logits = _head_logits(x, head)
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return {
        "label": classes[idx],
        "confidence": round(float(probs[idx]), 3),
        "probabilities": {label: round(float(probs[i]), 4) for i, label in enumerate(classes)},
    }


def _head_logits(x: np.ndarray, head: dict[str, Any]) -> np.ndarray:
    architecture = head.get("architecture", "linear")
    if architecture == "mlp_v1":
        hidden_weight = np.asarray(head["hidden_weight"], dtype=np.float32)
        hidden_bias = np.asarray(head["hidden_bias"], dtype=np.float32)
        output_weight = np.asarray(head["output_weight"], dtype=np.float32)
        output_bias = np.asarray(head["output_bias"], dtype=np.float32)
        hidden = np.maximum(hidden_weight @ x + hidden_bias, 0.0)
        return output_weight @ hidden + output_bias

    weight = np.asarray(head["weight"], dtype=np.float32)
    bias = np.asarray(head["bias"], dtype=np.float32)
    return weight @ x + bias
