from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def compute_mel_spectrogram(waveform, sample_rate: int, settings) -> np.ndarray:
    """Compute a log-scaled Mel spectrogram for visualization."""
    waveform_array = np.asarray(waveform, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=waveform_array,
        sr=sample_rate,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        win_length=settings.win_length,
        n_mels=settings.n_mels,
        fmin=settings.fmin,
        fmax=settings.fmax,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)


def save_mel_spectrogram(
    waveform,
    sample_rate: int,
    output_path: str | Path,
    settings,
    title: str | None = None,
) -> Path:
    """Render and save a Mel spectrogram PNG."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    mel_db = compute_mel_spectrogram(waveform, sample_rate, settings)

    fig, ax = plt.subplots(figsize=(10, 4))
    image = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=settings.hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=settings.fmin,
        fmax=settings.fmax,
        ax=ax,
    )
    ax.set(title=title or output.stem)
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
