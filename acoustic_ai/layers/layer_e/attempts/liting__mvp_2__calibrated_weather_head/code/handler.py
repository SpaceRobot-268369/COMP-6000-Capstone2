"""Registry handler for Liting's E-B MVP-2 calibrated weather head."""

from __future__ import annotations

from pathlib import Path

import torch

from .weather_head import extract_feature_vector, predict_with_checkpoint
from layers.layer_e.attempts.liting__mvp_1__panns_weather_baseline.code.handler import analyze as fallback_analyze
from layers.layer_e.attempts.liting__mvp_1__panns_weather_baseline.code.handler import load as fallback_load


PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model" / "candidates" / "liting" / "mvp_2__calibrated_weather_head" / "weather_head.pt"


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
        return fallback_analyze(state["fallback"], audio_path)

    features, evidence = extract_feature_vector(audio_path)
    prediction = predict_with_checkpoint(features, checkpoint)
    weather = {
        "component": "E-B",
        "method": "panns_dsp_frozen_features__calibrated_linear_head",
        "primary_model": "PANNs CNN14 frozen + calibrated E-B weather head",
        **prediction,
        "panns_available": evidence["panns_available"],
        "panns_status": evidence["panns_status"],
        "panns_evidence": evidence["panns_scores"],
        "supporting_detector": "DSP features included in calibrated head",
        "limitations": [
            "MVP-2 uses a small calibrated head over frozen features.",
            "The PANNs backbone is not fine-tuned.",
            "Thunder is suppressed until Site257 thunder evidence is validated.",
        ],
    }
    return {
        "head": "weather",
        "component": "E-B",
        "weather": weather,
        "summary": {
            "wind": {
                "intensity": weather["wind_intensity"],
                "confidence": weather["component_confidence"]["wind"],
            },
            "rain": {
                "intensity": weather["rain_intensity"],
                "confidence": weather["component_confidence"]["rain"],
            },
            "thunder": {
                "intensity": "none",
                "confidence": 0.0,
            },
        },
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

