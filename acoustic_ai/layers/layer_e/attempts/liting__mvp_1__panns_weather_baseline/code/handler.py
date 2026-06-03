"""Registry handler for Liting's E-B PANNs weather-analysis MVP attempt."""

from __future__ import annotations

from pathlib import Path

from .weather_detector import (
    analyse_weather,
    discover_legacy_weather_assets,
    load_site_promoted_weather_assets,
    load_weather_assets_from_index,
)


ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = Path(__file__).resolve().parents[5]
SMOKE_ATTEMPT_ROOT = (
    AI_ROOT
    / "layers"
    / "layer_e"
    / "attempts"
    / "liting__smoke_1__e_b_weather_analysis"
)


def load(checkpoint_dir: Path | None, params: dict, extra: dict | None = None) -> dict:
    return {
        "params": dict(params or {}),
        "calibration_assets": _weather_calibration_assets(),
    }


def analyze(state: dict, audio_path: str | Path) -> dict:
    weather = analyse_weather(
        audio_path,
        calibration_assets=state.get("calibration_assets") or [],
    )
    wind = _continuous_weather_summary(
        label=weather["wind_intensity"],
        confidence=weather["component_confidence"]["wind"],
        component="wind",
    )
    rain = _continuous_weather_summary(
        label=weather["rain_intensity"],
        confidence=weather["component_confidence"]["rain"],
        component="rain",
    )
    thunder = _thunder_summary(
        label=weather["thunder_intensity"],
        confidence=weather["component_confidence"]["thunder"],
    )
    return {
        "head": "weather",
        "component": "E-B",
        "observations": {
            "weather": {
                "wind": {"summary": wind},
                "rain": {"summary": rain},
                "thunder": thunder,
            }
        },
        "weather": weather,
        "summary": {
            "wind": wind,
            "rain": rain,
            "thunder": thunder,
        },
        "model": {
            "primary": weather.get("primary_model"),
            "method": weather.get("method"),
            "panns_available": weather.get("panns_available"),
            "panns_status": weather.get("panns_status"),
        },
        "limitations": weather.get("limitations", []),
    }


def generate(state: dict, seed: int | None = None, **runtime_params) -> dict:
    raise NotImplementedError("Layer E weather analysis is upload-based; use analyze().")


def _continuous_weather_summary(label: str, confidence: float, component: str) -> dict:
    intensity = _label_to_intensity(label, component)
    return {
        "intensity": intensity,
        "variability": 0.0,
        "coverage": 0.0 if label == "none" else 1.0,
        "label": label,
        "confidence": round(float(confidence), 3),
    }


def _thunder_summary(label: str, confidence: float) -> dict:
    # Site257 thunder does not yet have enough audited examples for a stable
    # MVP claim. Keep the schema but suppress provisional PANNs thunder hits.
    return {
        "intensity": 0.0,
        "event_count": 0,
        "events": [],
        "mean_interval_s": None,
        "label": "none",
        "confidence": round(float(confidence), 3),
        "status": "insufficient_site_data",
    }


def _label_to_intensity(label: str, component: str) -> float:
    if label == "none":
        return 0.0
    if component == "rain":
        scale = {"light": 0.30, "moderate": 0.62, "heavy": 0.90, "strong": 0.90}
    elif component == "wind":
        scale = {"light": 0.30, "moderate": 0.62, "strong": 0.90, "heavy": 0.90}
    else:
        scale = {"light": 0.30, "moderate": 0.62, "strong": 0.90, "heavy": 0.90}
    return scale.get(label, 0.0)


def _weather_calibration_assets() -> list:
    """Load weather labels for calibration and fallback support."""
    site_manifest = (
        SMOKE_ATTEMPT_ROOT
        / "data"
        / "analysis"
        / "site257_clap_promoted"
        / "layer_d_ready_manifest.csv"
    )
    weather_index = (
        AI_ROOT
        / "layers"
        / "layer_b"
        / "attempts"
        / "lucas__smoke_1__curated_assets"
        / "data"
        / "weather"
        / "asset_index.csv"
    )

    assets = []
    if site_manifest.exists():
        try:
            assets = [
                asset
                for asset in load_site_promoted_weather_assets(site_manifest)
                if asset.audio_path.exists()
            ]
        except Exception:
            assets = []
    if assets:
        return assets

    if weather_index.exists():
        try:
            assets = [
                asset
                for asset in load_weather_assets_from_index(weather_index)
                if asset.audio_path.exists()
            ]
        except Exception:
            assets = []
    return assets or discover_legacy_weather_assets()
