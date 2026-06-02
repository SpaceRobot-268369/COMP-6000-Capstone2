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
                "intensity": weather["thunder_intensity"],
                "confidence": weather["component_confidence"]["thunder"],
            },
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
