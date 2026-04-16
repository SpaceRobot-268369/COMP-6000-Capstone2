from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from app.schemas.audio import AudioMetadata, AudioProcessingResult


def load_audio(file_path: str | Path, target_sr: int, mono: bool = True):
    """Load audio into a normalized float32 waveform array."""
    waveform, sample_rate = librosa.load(
        path=str(file_path),
        sr=target_sr,
        mono=mono,
    )
    return waveform.astype(np.float32), sample_rate


def validate_audio(waveform, sample_rate: int, min_duration_seconds: float, silence_threshold: float) -> tuple[bool, str | None]:
    """Validate basic audio quality constraints."""
    if waveform is None:
        return False, "Waveform is missing."
    waveform_array = np.asarray(waveform, dtype=np.float32)
    if waveform_array.size == 0:
        return False, "Waveform is empty."
    if not np.isfinite(waveform_array).all():
        return False, "Waveform contains non-finite values."
    duration_seconds = waveform_array.shape[-1] / float(sample_rate)
    if duration_seconds < min_duration_seconds:
        return False, f"Audio is too short: {duration_seconds:.3f}s."
    peak_amplitude = float(np.max(np.abs(waveform_array)))
    if peak_amplitude <= silence_threshold:
        return False, "Audio appears to be silent."
    return True, None


def normalize_audio(waveform):
    """Normalize waveform amplitudes before feature extraction."""
    waveform_array = np.asarray(waveform, dtype=np.float32)
    peak_amplitude = float(np.max(np.abs(waveform_array))) if waveform_array.size else 0.0
    if peak_amplitude == 0.0:
        return waveform_array
    return waveform_array / peak_amplitude


def preprocess_audio(audio_metadata: AudioMetadata, settings) -> AudioProcessingResult:
    """Standardize an audio file and return a common processing result."""
    try:
        waveform, sample_rate = load_audio(
            file_path=audio_metadata.file_path,
            target_sr=settings.sample_rate,
            mono=settings.mono,
        )
        waveform = normalize_audio(waveform)
        is_valid, error_message = validate_audio(
            waveform=waveform,
            sample_rate=sample_rate,
            min_duration_seconds=settings.min_duration_seconds,
            silence_threshold=settings.silence_threshold,
        )
        return AudioProcessingResult(
            audio_id=audio_metadata.audio_id,
            waveform=waveform,
            sample_rate=sample_rate,
            duration=float(waveform.shape[-1] / sample_rate),
            num_channels=1 if settings.mono else None,
            is_valid=is_valid,
            error_message=error_message,
            metadata={
                "source_path": str(audio_metadata.file_path),
                "target_sample_rate": settings.sample_rate,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive bootstrap path
        return AudioProcessingResult(
            audio_id=audio_metadata.audio_id,
            waveform=None,
            sample_rate=None,
            duration=None,
            num_channels=None,
            is_valid=False,
            error_message=str(exc),
        )
