"""Layer B weather segment selection for Layer D handoff.

Layer B owns weather asset retrieval and segment selection. It does not place
segments on the final timeline or render the final mix; Layer D does that.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import soundfile as sf

WeatherType = Literal["wind", "rain", "thunder"]

ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_ASSET_INDEX = ATTEMPT_ROOT / "data" / "weather" / "asset_index.csv"
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
    intensity_index = _load_asset_intensity_index()
    for request in requests:
        try:
            candidates = _retrieve_assets(
                query_text=request["query"],
                weather_type=request["weather_type"],
                top_k=max(top_assets, 10),
            )
        except Exception as exc:
            warnings.append(
                f"{request['weather_type']} retrieval failed: {exc}"
            )
            continue

        target_intensity = _target_intensity(
            request["weather_type"],
            wind_speed_ms=wind_speed_ms,
            precipitation_mm=precipitation_mm,
            include_thunder=include_thunder,
        )
        candidates = _apply_intensity_preference(
            candidates,
            weather_type=request["weather_type"],
            target_intensity=target_intensity,
            intensity_index=intensity_index,
        )[:top_assets]

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
        from .retriever import retrieve_weather_asset
    except ModuleNotFoundError:
        try:
            from retriever import retrieve_weather_asset
        except ModuleNotFoundError:
            from acoustic_ai.layers.layer_b.attempts.lucas__smoke_1__curated_assets.code.retriever import (
                retrieve_weather_asset,
            )

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
            target_intensity = asset.get("target_intensity", "unknown")
            score = _segment_rank_score(
                retrieval_score=float(asset["score"]),
                quality=quality,
                weather_type=weather_type,
                target_intensity=target_intensity,
            )
            ranked.append({
                "weather_type": weather_type,
                "file": asset["file"],
                "score": score,
                "retrieval_score": float(asset["score"]),
                "asset_intensity": asset.get("asset_intensity", "unknown"),
                "target_intensity": asset.get("target_intensity", "unknown"),
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

    usable = [
        item
        for item in ranked
        if _segment_is_usable(item["weather_type"], item.get("validation", {}))
    ]
    return sorted(usable or ranked, key=lambda item: item["score"], reverse=True)


def _segment_rank_score(
    *,
    retrieval_score: float,
    quality: dict,
    weather_type: WeatherType,
    target_intensity: str,
) -> float:
    score = retrieval_score + quality["quality_score"]
    if weather_type == "wind" and target_intensity == "light":
        rms = float(quality.get("rms", 0.0))
        peak = float(quality.get("peak", 0.0))
        lightness_bonus = max(0.0, 0.04 - rms) * 4.0
        gust_penalty = max(0.0, peak - 0.25) * 0.35
        energy_penalty = max(0.0, rms - 0.04) * 3.0
        score += lightness_bonus - gust_penalty - energy_penalty
    return round(float(score), 4)


def _segment_is_usable(weather_type: WeatherType, validation: dict) -> bool:
    if validation.get("asset_available") is False:
        return False

    silence_ratio = float(validation.get("silence_ratio", 1.0))
    clipping_ratio = float(validation.get("clipping_ratio", 1.0))
    stability = float(validation.get("stability", 0.0))

    if clipping_ratio > 0.02:
        return False
    if weather_type in {"wind", "rain"}:
        return silence_ratio <= 0.40 and stability >= 0.20
    if weather_type == "thunder":
        return silence_ratio <= 0.80
    return True


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
        "asset_intensity": asset.get("asset_intensity", "unknown"),
        "target_intensity": asset.get("target_intensity", "unknown"),
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


def _load_asset_intensity_index(index_path: Path = DEFAULT_ASSET_INDEX) -> dict[str, dict[str, str]]:
    if not index_path.exists():
        return {}

    intensity_index = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clip_path = row.get("clip_path", "")
            if not clip_path:
                continue
            intensities = {
                "wind": _normalize_intensity(row.get("wind_intensity", "")),
                "rain": _normalize_intensity(row.get("rain_intensity", "")),
                "thunder": _normalize_intensity(row.get("thunder_intensity", "")),
            }
            resolved = str(_resolve_asset_path(clip_path))
            intensity_index[resolved] = intensities
            intensity_index[Path(clip_path).name] = intensities

    return intensity_index


def _apply_intensity_preference(
    candidates: list[dict],
    *,
    weather_type: WeatherType,
    target_intensity: str,
    intensity_index: dict[str, str],
) -> list[dict]:
    annotated = []
    for candidate in candidates:
        resolved = str(_resolve_asset_path(candidate["file"]))
        asset_intensities = (
            intensity_index.get(resolved)
            or intensity_index.get(Path(candidate["file"]).name)
            or {}
        )
        asset_intensity = asset_intensities.get(weather_type, "")
        item = {
            **candidate,
            "asset_intensity": asset_intensity or "unknown",
            "target_intensity": target_intensity or "unknown",
        }
        annotated.append(item)

    matches = [
        candidate
        for candidate in annotated
        if candidate["asset_intensity"] == target_intensity
    ]
    if matches:
        return matches

    return sorted(
        annotated,
        key=lambda candidate: _intensity_distance(
            candidate["asset_intensity"],
            target_intensity,
            weather_type,
        ),
    )


def _target_intensity(
    weather_type: WeatherType,
    *,
    wind_speed_ms: Optional[float],
    precipitation_mm: Optional[float],
    include_thunder: bool,
) -> str:
    if weather_type == "wind":
        return _normalize_intensity(_wind_intensity(wind_speed_ms or 0.0))
    if weather_type == "rain":
        return _normalize_intensity(_rain_intensity(precipitation_mm or 0.0))
    if include_thunder:
        return "strong"
    return "medium"


def _normalize_intensity(intensity: str) -> str:
    normalized = (intensity or "").strip().lower()
    if normalized in {"moderate", "medium"}:
        return "medium"
    if normalized in {"heavy", "strong"}:
        return "strong"
    return normalized


def _intensity_distance(asset_intensity: str, target_intensity: str, weather_type: WeatherType) -> int:
    if not asset_intensity or asset_intensity == "unknown":
        return 99
    order = {
        "wind": {"light": 0, "medium": 1, "strong": 2},
        "rain": {"light": 0, "medium": 1, "strong": 2},
        "thunder": {"medium": 1, "strong": 2},
    }[weather_type]
    return abs(order.get(asset_intensity, 99) - order.get(target_intensity, 99))


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
