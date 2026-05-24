"""Layer B weather segment selection for Layer D handoff.

Layer B owns weather asset retrieval and segment selection. It does not place
segments on the final timeline or render the final mix; Layer D does that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import soundfile as sf

WeatherType = Literal["wind", "rain", "thunder"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_OVERLAP_SECONDS = 2.0
DEFAULT_TOP_ASSETS = 3
DEFAULT_SEGMENTS_PER_TYPE = 2


def select_weather_segments(
    *,
    query: Optional[str] = None,
    weather_types: Optional[Iterable[WeatherType]] = None,
    wind_speed_ms: Optional[float] = None,
    precipitation_mm: Optional[float] = None,
    include_thunder: bool = False,
    target_duration: float = 30.0,
    top_assets: int = DEFAULT_TOP_ASSETS,
    segments_per_type: int = DEFAULT_SEGMENTS_PER_TYPE,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> dict:
    """Select weather segments and return Layer D handoff metadata."""
    if target_duration <= 0:
        raise ValueError("target_duration must be greater than 0.")
    if top_assets <= 0:
        raise ValueError("top_assets must be greater than 0.")
    if segments_per_type <= 0:
        raise ValueError("segments_per_type must be greater than 0.")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be greater than 0.")
    if overlap_seconds < 0 or overlap_seconds >= window_seconds:
        raise ValueError("overlap_seconds must be >= 0 and smaller than window_seconds.")

    resolved_types = list(weather_types or _infer_weather_types(
        wind_speed_ms=wind_speed_ms,
        precipitation_mm=precipitation_mm,
        include_thunder=include_thunder,
    ))

    requests = [
        {
            "weather_type": weather_type,
            "query": query or _build_weather_query(
                weather_type,
                wind_speed_ms=wind_speed_ms,
                precipitation_mm=precipitation_mm,
            ),
        }
        for weather_type in resolved_types
    ]

    results = []
    warnings = []
    for request in requests:
        try:
            candidates = _retrieve_assets(
                query_text=request["query"],
                weather_type=request["weather_type"],
                top_k=top_assets,
            )
        except Exception as exc:
            warnings.append(
                f"{request['weather_type']} retrieval failed: {exc}"
            )
            continue

        segments = _rank_asset_segments(
            candidates,
            weather_type=request["weather_type"],
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
        )
        results.extend(segments[:segments_per_type])

    return {
        "ok": True,
        "query": query,
        "target_duration": target_duration,
        "weather_types": resolved_types,
        "window_seconds": window_seconds,
        "overlap_seconds": overlap_seconds,
        "results": results,
        "warnings": warnings,
        "layer_d_contract": {
            "layer_b_selects": "weather asset files and candidate segment metadata",
            "layer_d_owns": "timeline placement, crossfades, gain staging, and final mix",
        },
    }


def _retrieve_assets(query_text: str, weather_type: WeatherType, top_k: int) -> list[dict]:
    try:
        from modules.weather.retriever import retrieve_weather_asset
    except ModuleNotFoundError:
        from acoustic_ai.modules.weather.retriever import retrieve_weather_asset

    return retrieve_weather_asset(
        query_text=query_text,
        weather_type=weather_type,
        top_k=top_k,
    )


def _infer_weather_types(
    *,
    wind_speed_ms: Optional[float],
    precipitation_mm: Optional[float],
    include_thunder: bool,
) -> tuple[WeatherType, ...]:
    weather_types: list[WeatherType] = []
    if wind_speed_ms is not None and wind_speed_ms >= 2:
        weather_types.append("wind")
    if precipitation_mm is not None and precipitation_mm > 0:
        weather_types.append("rain")
    if include_thunder:
        weather_types.append("thunder")
    if not weather_types:
        weather_types.append("wind")
    return tuple(weather_types)


def _build_weather_query(
    weather_type: WeatherType,
    *,
    wind_speed_ms: Optional[float],
    precipitation_mm: Optional[float],
) -> str:
    if weather_type == "wind":
        intensity = _wind_intensity(wind_speed_ms or 0.0)
        return f"{intensity} natural forest wind ambience"
    if weather_type == "rain":
        intensity = _rain_intensity(precipitation_mm or 0.0)
        return f"{intensity} forest rain ambience"
    return "distant natural rolling thunderstorm ambience"


def _rank_asset_segments(
    assets: list[dict],
    *,
    weather_type: WeatherType,
    window_seconds: float,
    overlap_seconds: float,
) -> list[dict]:
    ranked = []
    for asset in assets:
        path = _resolve_asset_path(asset["file"])
        if not path.exists():
            ranked.append(_missing_asset_segment(asset, weather_type))
            continue

        try:
            info = sf.info(path)
        except Exception as exc:
            ranked.append(_missing_asset_segment(asset, weather_type, warning=str(exc)))
            continue

        for start_time, duration in _segment_windows(
            duration_seconds=float(info.frames) / float(info.samplerate),
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
        ):
            quality = _score_segment_quality(
                path,
                start_time=start_time,
                duration=duration,
                sample_rate=info.samplerate,
                weather_type=weather_type,
            )
            score = float(asset["score"]) + quality["quality_score"]
            ranked.append({
                "weather_type": weather_type,
                "file": asset["file"],
                "score": score,
                "retrieval_score": float(asset["score"]),
                "segment": {
                    "start_time": start_time,
                    "duration": duration,
                    "fade_in": 0.5 if weather_type == "thunder" else 1.0,
                    "fade_out": 2.0 if weather_type == "thunder" else 1.0,
                    "role": _segment_role(weather_type),
                },
                "validation": quality,
                "reason": _segment_reason(weather_type, quality),
            })

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def _segment_windows(
    *,
    duration_seconds: float,
    window_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    if duration_seconds <= window_seconds:
        return [(0.0, max(0.0, duration_seconds))]

    step = window_seconds - overlap_seconds
    starts = np.arange(0.0, duration_seconds - window_seconds + 1e-6, step)
    return [(round(float(start), 3), window_seconds) for start in starts]


def _score_segment_quality(
    path: Path,
    *,
    start_time: float,
    duration: float,
    sample_rate: int,
    weather_type: WeatherType,
) -> dict:
    start_frame = int(round(start_time * sample_rate))
    frame_count = max(1, int(round(duration * sample_rate)))
    audio, _ = sf.read(path, start=start_frame, frames=frame_count, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)

    if mono.size == 0:
        return {
            "quality_score": -1.0,
            "rms": 0.0,
            "peak": 0.0,
            "silence_ratio": 1.0,
            "clipping_ratio": 0.0,
            "stability": 0.0,
        }

    abs_audio = np.abs(mono)
    rms = float(np.sqrt(np.mean(np.square(mono))))
    peak = float(np.max(abs_audio))
    silence_ratio = float(np.mean(abs_audio < 0.005))
    clipping_ratio = float(np.mean(abs_audio > 0.98))
    stability = _rms_stability(mono)

    if weather_type in {"wind", "rain"}:
        quality_score = 0.25 * stability - 0.45 * silence_ratio - 0.35 * clipping_ratio
    else:
        transient_strength = min(peak / max(rms, 1e-6), 12.0) / 12.0
        quality_score = 0.30 * transient_strength - 0.35 * silence_ratio - 0.35 * clipping_ratio

    return {
        "quality_score": round(float(quality_score), 4),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "silence_ratio": round(silence_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 4),
        "stability": round(stability, 4),
    }


def _rms_stability(audio: np.ndarray, frame_size: int = 2048) -> float:
    if len(audio) < frame_size:
        return 1.0
    frame_count = len(audio) // frame_size
    frames = audio[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    mean_rms = float(np.mean(rms))
    if mean_rms <= 1e-8:
        return 0.0
    return float(1.0 / (1.0 + np.std(rms) / mean_rms))


def _missing_asset_segment(
    asset: dict,
    weather_type: WeatherType,
    warning: Optional[str] = None,
) -> dict:
    validation = {
        "quality_score": -1.0,
        "asset_available": False,
    }
    if warning:
        validation["warning"] = warning

    return {
        "weather_type": weather_type,
        "file": asset["file"],
        "score": float(asset["score"]) - 1.0,
        "retrieval_score": float(asset["score"]),
        "segment": {
            "start_time": 0.0,
            "duration": DEFAULT_WINDOW_SECONDS,
            "fade_in": 0.5 if weather_type == "thunder" else 1.0,
            "fade_out": 2.0 if weather_type == "thunder" else 1.0,
            "role": _segment_role(weather_type),
        },
        "validation": validation,
        "reason": "Asset was retrieved by CLAP, but the local WAV file is not available for segment validation.",
    }


def _resolve_asset_path(file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _segment_role(weather_type: WeatherType) -> str:
    return {
        "wind": "wind_texture",
        "rain": "base_rain",
        "thunder": "thunder_accent",
    }[weather_type]


def _segment_reason(weather_type: WeatherType, quality: dict) -> str:
    if weather_type == "thunder":
        return "CLAP-matched thunder asset with transient-oriented segment validation."
    if quality["silence_ratio"] > 0.35:
        return "CLAP-matched asset, but this segment contains notable quiet sections."
    return f"CLAP-matched {weather_type} asset with stable texture suitable for Layer D."


def _wind_intensity(wind_speed_ms: float) -> str:
    if wind_speed_ms < 2:
        return "no"
    if wind_speed_ms < 6:
        return "light"
    if wind_speed_ms < 10:
        return "moderate"
    return "strong"


def _rain_intensity(precipitation_mm: float) -> str:
    if precipitation_mm <= 0:
        return "no"
    if precipitation_mm < 2:
        return "light"
    if precipitation_mm < 5:
        return "moderate"
    return "heavy"
