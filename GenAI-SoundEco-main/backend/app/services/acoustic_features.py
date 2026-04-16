from __future__ import annotations

from collections.abc import Iterable

import librosa
import numpy as np


def summarize_feature(feature_values: Iterable[float], prefix: str) -> dict[str, float | None]:
    values = np.asarray(list(feature_values), dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }

    mean_value = float(np.mean(values))
    variance = float(np.var(values))
    return {
        f"{prefix}_mean": mean_value,
        f"{prefix}_std": variance ** 0.5,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def extract_acoustic_features(waveform, sample_rate: int, settings) -> dict[str, float | None]:
    """Extract structured acoustic features from a waveform."""
    waveform_array = np.asarray(waveform, dtype=np.float32)
    if waveform_array.size == 0:
        return {}

    rms = librosa.feature.rms(
        y=waveform_array,
        frame_length=settings.n_fft,
        hop_length=settings.hop_length,
    ).flatten()
    zcr = librosa.feature.zero_crossing_rate(
        y=waveform_array,
        frame_length=settings.n_fft,
        hop_length=settings.hop_length,
    ).flatten()
    spectral_centroid = librosa.feature.spectral_centroid(
        y=waveform_array,
        sr=sample_rate,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
    ).flatten()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=waveform_array,
        sr=sample_rate,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
    ).flatten()
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=waveform_array,
        sr=sample_rate,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
    ).flatten()
    mfcc = librosa.feature.mfcc(
        y=waveform_array,
        sr=sample_rate,
        n_mfcc=settings.n_mfcc,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        n_mels=settings.n_mels,
    )

    features: dict[str, float | None] = {}
    features.update(summarize_feature(rms, "rms"))
    features.update(summarize_feature(zcr, "zcr"))
    features.update(summarize_feature(spectral_centroid, "spectral_centroid"))
    features.update(summarize_feature(spectral_bandwidth, "spectral_bandwidth"))
    features.update(summarize_feature(spectral_rolloff, "spectral_rolloff"))

    for coefficient_index in range(mfcc.shape[0]):
        features.update(
            summarize_feature(
                mfcc[coefficient_index, :],
                f"mfcc_{coefficient_index + 1}",
            )
        )

    features["duration_seconds"] = float(waveform_array.shape[-1] / sample_rate)
    features["sample_rate"] = float(sample_rate)
    return features
