"""Registry handler for Liting's E-B MVP-3 balanced weather head."""

from __future__ import annotations

from pathlib import Path

import torch

from .weather_head import extract_feature_vector, predict_with_checkpoint
from layers.layer_e.attempts.liting__mvp_1__panns_weather_baseline.code.handler import analyze as fallback_analyze
from layers.layer_e.attempts.liting__mvp_1__panns_weather_baseline.code.handler import load as fallback_load


PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model" / "candidates" / "liting" / "mvp_3__balanced_weather_head" / "weather_head.pt"
INTENSITY_VALUE = {
    "none": 0.0,
    "light": 0.25,
    "moderate": 0.6,
    "strong": 0.85,
    "heavy": 0.85,
}


def _label_value(label: str | None) -> float:
    return INTENSITY_VALUE.get(str(label or "none").lower(), 0.0)


def _weather_summary(weather: dict) -> dict:
    confidence = weather.get("component_confidence", {})
    return {
        "wind": {
            "intensity": weather.get("wind_intensity", "none"),
            "confidence": float(confidence.get("wind", weather.get("confidence", 0.0))),
        },
        "rain": {
            "intensity": weather.get("rain_intensity", "none"),
            "confidence": float(confidence.get("rain", weather.get("confidence", 0.0))),
        },
        "thunder": {
            "intensity": "none",
            "confidence": 0.9,
        },
    }


def _weather_observations(weather: dict) -> dict:
    summary = _weather_summary(weather)
    wind_label = summary["wind"]["intensity"]
    rain_label = summary["rain"]["intensity"]
    return {
        "weather": {
            "wind": {
                "summary": {
                    "label": wind_label,
                    "intensity": _label_value(wind_label),
                    "variability": 0.0,
                    "coverage": summary["wind"]["confidence"],
                    "confidence": summary["wind"]["confidence"],
                }
            },
            "rain": {
                "summary": {
                    "label": rain_label,
                    "intensity": _label_value(rain_label),
                    "variability": 0.0,
                    "coverage": summary["rain"]["confidence"],
                    "confidence": summary["rain"]["confidence"],
                }
            },
            "thunder": {
                "label": "none",
                "intensity": 0.0,
                "event_count": 0,
                "events": [],
                "mean_interval_s": None,
                "confidence": 0.9,
                "status": "suppressed_until_site257_thunder_evidence_is_validated",
            },
        }
    }


def load(checkpoint_dir: Path | None, params: dict, extra: dict | None = None) -> dict:
    checkpoint_path = DEFAULT_CHECKPOINT
    if checkpoint_dir:
        candidate = Path(checkpoint_dir) / "weather_head.pt"
        if candidate.exists():
            checkpoint_path = candidate

    checkpoint = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    return {
        "params": dict(params or {}),
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "fallback": fallback_load(None, params or {}, extra),
    }


def analyze(state: dict, audio_path: str | Path) -> dict:
    checkpoint = state.get("checkpoint")
    if not checkpoint:
        report = fallback_analyze(state["fallback"], audio_path)
        weather = report.get("weather", {})
        if "observations" not in report:
            report["observations"] = _weather_observations(weather)
        return report

    features, evidence = extract_feature_vector(audio_path)
    prediction = predict_with_checkpoint(features, checkpoint)
    weather = {
        "component": "E-B",
        "method": "panns_dsp_frozen_features__balanced_mlp_head",
        "primary_model": "PANNs CNN14 frozen + balanced E-B weather MLP head",
        **prediction,
        "panns_available": evidence["panns_available"],
        "panns_status": evidence["panns_status"],
        "panns_evidence": evidence["panns_scores"],
        "supporting_detector": "DSP features included in calibrated head",
        "limitations": [
            "MVP-3 uses class-balanced small MLP heads over frozen features.",
            "The PANNs backbone is not fine-tuned.",
            "Thunder is suppressed until Site257 thunder evidence is validated.",
        ],
    }
    return {
        "head": "weather",
        "component": "E-B",
        "weather": weather,
        "summary": _weather_summary(weather),
        "observations": _weather_observations(weather),
        "model": {
            "primary": weather["primary_model"],
            "method": weather["method"],
            "checkpoint_path": state.get("checkpoint_path"),
            "panns_available": evidence["panns_available"],
            "panns_status": evidence["panns_status"],
        },
        "limitations": weather["limitations"],
    }


def generate(state: dict, seed: int | None = None, **runtime_params) -> dict:
    raise NotImplementedError("Layer E weather analysis is upload-based; use analyze().")
