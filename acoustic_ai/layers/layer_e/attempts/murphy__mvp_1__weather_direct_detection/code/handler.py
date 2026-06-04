"""Registry handler for E-B weather direct-detection analysis MVP.

This keeps the dev Analysis page and the offline CLI on the same frozen gate:
CLAP + PANNs + AST, with gate v1.1 as documented in the attempt README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .run_weather_analysis import analyze as run_weather_analysis
    from .run_weather_analysis import load_params
except ImportError:  # pragma: no cover - direct script/import fallback.
    from run_weather_analysis import analyze as run_weather_analysis
    from run_weather_analysis import load_params


class WeatherAnalyzer:
    def __init__(
        self,
        params: dict[str, Any],
        model_backend: str = "clap",
        audioset_backend: str = "panns",
        guard_backend: str = "ast",
    ) -> None:
        self.params = params
        self.model_backend = model_backend
        self.audioset_backend = audioset_backend
        self.guard_backend = guard_backend

    def analyze(self, audio_path: str | Path) -> dict[str, Any]:
        return run_weather_analysis(
            Path(audio_path),
            self.params,
            model_backend=self.model_backend,
            audioset_backend=self.audioset_backend,
            guard_backend=self.guard_backend,
        )


def load(
    checkpoint_dir: Path | None,
    params: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> WeatherAnalyzer:
    del checkpoint_dir, extra
    merged_params = load_params()
    merged_params.update(params or {})
    return WeatherAnalyzer(
        merged_params,
        model_backend=str((params or {}).get("model_backend", "clap")),
        audioset_backend=str((params or {}).get("audioset_backend", "panns")),
        guard_backend=str((params or {}).get("guard_backend", "ast")),
    )


def generate(state: WeatherAnalyzer, seed: int | None = None, **_ignored) -> dict[str, Any]:
    del state, seed
    raise NotImplementedError(
        "Layer E-B weather analysis is upload-based. Use analyze(state, audio_path) "
        "through the registry /analyze endpoint instead of /generate."
    )


def analyze(state: WeatherAnalyzer, audio_path: str | Path) -> dict[str, Any]:
    """Registry analyze() entry point for the E-B weather head."""
    return state.analyze(audio_path)
