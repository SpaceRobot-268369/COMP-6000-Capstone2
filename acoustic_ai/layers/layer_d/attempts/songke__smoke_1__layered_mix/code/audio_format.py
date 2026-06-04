"""Audio format normalization for Layer D.

This module only converts technical format: dtype, channel count, sample rate,
and exact frame count. It deliberately does not change loudness or apply EQ,
denoising, compression, limiting, fades, or other mix processing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audio_metrics import audio_metrics


@dataclass(frozen=True)
class NormalizedAudio:
    audio: np.ndarray
    sample_rate: int
    operations: tuple[str, ...]
    source_metrics: dict[str, Any]
    output_metrics: dict[str, Any]


def load_audio(path: Path | str) -> tuple[np.ndarray, int]:
    """Load an audio file as float32 with shape ``(frames, channels)``."""

    audio, sample_rate = sf.read(Path(path), dtype="float32", always_2d=True)
    _validate_audio(audio, sample_rate)
    return audio, int(sample_rate)


def normalize_audio_format(
    audio: np.ndarray,
    source_sample_rate: int,
    *,
    target_sample_rate: int,
    target_channels: int,
) -> NormalizedAudio:
    """Convert audio to the requested format while preserving duration."""

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    _validate_audio(samples, source_sample_rate)
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")
    if target_channels not in (1, 2):
        raise ValueError("target_channels must be 1 or 2")

    source = audio_metrics(samples, source_sample_rate)
    operations: list[str] = []

    converted = _convert_channels(samples, target_channels, operations)
    if source_sample_rate != target_sample_rate:
        converted = _resample(
            converted,
            source_sample_rate=source_sample_rate,
            target_sample_rate=target_sample_rate,
        )
        operations.append(f"resample_{source_sample_rate}_to_{target_sample_rate}")

    target_frames = int(round(source["duration_s"] * target_sample_rate))
    converted = _fit_frame_count(converted, target_frames)
    converted = np.ascontiguousarray(converted, dtype=np.float32)
    _validate_audio(converted, target_sample_rate)

    return NormalizedAudio(
        audio=converted,
        sample_rate=target_sample_rate,
        operations=tuple(operations or ["none"]),
        source_metrics=source,
        output_metrics=audio_metrics(converted, target_sample_rate),
    )


def normalize_audio_file(
    source_path: Path | str,
    output_path: Path | str,
    *,
    target_sample_rate: int,
    target_channels: int,
    subtype: str,
) -> NormalizedAudio:
    """Normalize one file and write it without overwriting its source."""

    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if source == output:
        raise ValueError("output_path must not overwrite source_path")

    audio, sample_rate = load_audio(source)
    result = normalize_audio_format(
        audio,
        sample_rate,
        target_sample_rate=target_sample_rate,
        target_channels=target_channels,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result.audio, result.sample_rate, format="WAV", subtype=subtype)
    return result


def _convert_channels(
    audio: np.ndarray,
    target_channels: int,
    operations: list[str],
) -> np.ndarray:
    source_channels = audio.shape[1]
    if source_channels == target_channels:
        return audio
    if target_channels == 1:
        operations.append(f"downmix_{source_channels}_to_mono")
        return np.mean(audio, axis=1, keepdims=True, dtype=np.float32)
    if source_channels == 1 and target_channels == 2:
        operations.append("duplicate_mono_to_stereo")
        return np.repeat(audio, 2, axis=1)
    raise ValueError(f"unsupported channel conversion: {source_channels} to {target_channels}")


def _resample(
    audio: np.ndarray,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    divisor = math.gcd(source_sample_rate, target_sample_rate)
    up = target_sample_rate // divisor
    down = source_sample_rate // divisor
    channels = [
        resample_poly(audio[:, channel], up, down).astype(np.float32, copy=False)
        for channel in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1)


def _fit_frame_count(audio: np.ndarray, target_frames: int) -> np.ndarray:
    if audio.shape[0] > target_frames:
        return audio[:target_frames]
    if audio.shape[0] < target_frames:
        padding = np.zeros((target_frames - audio.shape[0], audio.shape[1]), dtype=np.float32)
        return np.concatenate((audio, padding), axis=0)
    return audio


def _validate_audio(audio: np.ndarray, sample_rate: int) -> None:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if audio.ndim != 2 or audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError("audio must be a non-empty array shaped (frames, channels)")
    if not np.isfinite(audio).all():
        raise ValueError("audio contains NaN or infinite values")
