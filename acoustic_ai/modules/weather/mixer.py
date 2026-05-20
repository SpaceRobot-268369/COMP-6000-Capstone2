"""Module B retrieval-based weather layer mixer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from modules.weather.asset_index import (
    DEFAULT_ASSET_INDEX,
    DEFAULT_ASSET_ROOT,
    WeatherAsset,
    select_asset,
)

DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_DURATION_SECONDS = 10.0
FADE_SECONDS = 0.25

WIND_GAIN_DB = {
    "none": -120.0,
    "light": -24.0,
    "moderate": -18.0,
    "strong": -12.0,
}
RAIN_GAIN_DB = {
    "none": -120.0,
    "light": -26.0,
    "moderate": -20.0,
    "heavy": -14.0,
}


def wind_intensity(wind_speed_ms: float) -> str:
    """Map wind speed to the Layer B wind bucket."""

    if wind_speed_ms < 2.0:
        return "none"
    if wind_speed_ms < 6.0:
        return "light"
    if wind_speed_ms < 10.0:
        return "moderate"
    return "strong"


def rain_intensity(precipitation_mm: float) -> str:
    """Map precipitation rate to the Layer B rain bucket."""

    if precipitation_mm <= 0.0:
        return "none"
    if precipitation_mm < 2.0:
        return "light"
    if precipitation_mm < 5.0:
        return "moderate"
    return "heavy"


def generate_weather_layer(
    *,
    wind_speed_ms: float,
    precipitation_mm: float,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: Optional[int] = None,
    index_path: Path | str = DEFAULT_ASSET_INDEX,
    asset_root: Path | str = DEFAULT_ASSET_ROOT,
) -> tuple[np.ndarray, dict]:
    """Generate the weather-only layer and metadata.

    Missing assets produce silence for that sub-layer and explicit metadata.
    This lets Layer D run before the curated weather library is complete.
    """

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    target_samples = int(round(duration_seconds * sample_rate))
    weather = np.zeros(target_samples, dtype=np.float32)

    wind_bucket = wind_intensity(float(wind_speed_ms))
    rain_bucket = rain_intensity(float(precipitation_mm))
    layer_meta = []

    for layer, bucket, gain_map in (
        ("wind", wind_bucket, WIND_GAIN_DB),
        ("rain", rain_bucket, RAIN_GAIN_DB),
    ):
        if bucket == "none":
            layer_meta.append(_silent_meta(layer, bucket, "not_requested"))
            continue

        asset = select_asset(
            layer,
            bucket,
            seed=seed,
            index_path=index_path,
            asset_root=asset_root,
        )
        if asset is None:
            layer_meta.append(_silent_meta(layer, bucket, "missing_asset"))
            continue

        audio = _load_asset_audio(asset.path, sample_rate)
        audio = _fit_duration(audio, target_samples)
        audio = _apply_fade(audio, sample_rate)
        audio = audio * _db_to_amp(gain_map[bucket])
        weather += audio.astype(np.float32)
        layer_meta.append(_asset_meta(asset, bucket, gain_map[bucket]))

    peak = float(np.max(np.abs(weather))) if weather.size else 0.0
    if peak > 1.0:
        weather = weather / peak * 0.98

    metadata = {
        "sample_rate": sample_rate,
        "duration_seconds": duration_seconds,
        "wind_speed_ms": float(wind_speed_ms),
        "precipitation_mm": float(precipitation_mm),
        "wind_intensity": wind_bucket,
        "rain_intensity": rain_bucket,
        "layers": layer_meta,
        "peak_normalized": peak > 1.0,
    }
    return weather.astype(np.float32), metadata


def _load_asset_audio(path: Path, sample_rate: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Weather asset not found: {path}")

    audio, source_sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if source_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sample_rate)

    if audio.size == 0:
        raise ValueError(f"Weather asset is empty: {path}")

    return audio.astype(np.float32)


def _fit_duration(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if audio.size >= target_samples:
        return audio[:target_samples]

    repeats = int(np.ceil(target_samples / audio.size))
    return np.tile(audio, repeats)[:target_samples]


def _apply_fade(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    fade_samples = min(int(round(FADE_SECONDS * sample_rate)), audio.size // 2)
    if fade_samples <= 0:
        return audio

    faded = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    faded[:fade_samples] *= fade_in
    faded[-fade_samples:] *= fade_in[::-1]
    return faded


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _asset_meta(asset: WeatherAsset, bucket: str, gain_db: float) -> dict:
    return {
        "layer": asset.layer,
        "intensity": bucket,
        "status": "selected",
        "asset_id": asset.asset_id,
        "clip_path": str(asset.path),
        "gain_db": gain_db,
        "source": asset.source,
        "license": asset.license,
        "attribution": asset.attribution,
        "notes": asset.notes,
    }


def _silent_meta(layer: str, bucket: str, reason: str) -> dict:
    return {
        "layer": layer,
        "intensity": bucket,
        "status": "silent",
        "reason": reason,
    }
