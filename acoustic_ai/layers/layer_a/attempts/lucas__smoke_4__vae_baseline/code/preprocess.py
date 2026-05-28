"""Audio preprocessing — raw audio to mel-spectrogram.

All spectrogram parameters are defined here as a single source of truth.
Every other module imports SPEC_CFG rather than hardcoding values.

Spectrogram config rationale for ecoacoustic recordings at 22 050 Hz:
  - n_fft=1024     → 46 ms window; resolves individual bird notes
  - hop_length=512 → 50% overlap; 23 ms time step → ~43 frames/sec
  - n_mels=128     → standard for audio ML; covers 0–11 025 Hz
  - fmin=50        → removes sub-bass rumble below most wildlife sounds
  - fmax=11000     → Nyquist for 22 050 Hz source
  - top_db=80      → dynamic range cap; suppresses extreme silence floors
"""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Spectrogram config — single source of truth
# ---------------------------------------------------------------------------
SPEC_CFG = {
    "sample_rate": 22_050,
    "n_fft":       1024,
    "hop_length":  512,
    "n_mels":      128,
    "fmin":        50,
    "fmax":        11_000,
    "top_db":      80,
}

# Expected frames for a 300 s clip at hop_length=512, sr=22050
# 300 * 22050 / 512 ≈ 12 921 frames
FRAMES_PER_CLIP = int(300 * SPEC_CFG["sample_rate"] / SPEC_CFG["hop_length"])


def load_audio(path: str, target_sr: int = SPEC_CFG["sample_rate"]) -> np.ndarray:
    """Load an audio file and return a mono float32 waveform at target_sr."""
    import librosa
    waveform, sr = librosa.load(path, sr=target_sr, mono=True)
    return waveform.astype(np.float32)


def waveform_to_melspec(
    waveform: np.ndarray,
    cfg: dict = SPEC_CFG,
) -> np.ndarray:
    """Convert a mono waveform to a log-mel spectrogram.

    Returns:
        ndarray of shape (n_mels, time_frames), float32, in dB scale.
    """
    import librosa
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"],
        fmax=cfg["fmax"],
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=cfg["top_db"])
    return log_mel.astype(np.float32)


def melspec_to_tensor(log_mel: np.ndarray) -> torch.Tensor:
    """Convert (n_mels, time) ndarray to (1, n_mels, time) tensor."""
    return torch.from_numpy(log_mel).unsqueeze(0)


def pad_or_crop(tensor: torch.Tensor, target_frames: int = FRAMES_PER_CLIP) -> torch.Tensor:
    """Pad (repeat) or crop a (1, n_mels, time) tensor to target_frames."""
    _, n_mels, t = tensor.shape
    if t >= target_frames:
        return tensor[:, :, :target_frames]
    # Tile then crop to avoid discontinuities from zero-padding
    repeats = (target_frames // t) + 1
    tensor = tensor.repeat(1, 1, repeats)
    return tensor[:, :, :target_frames]


def audio_to_tensor(path: str) -> torch.Tensor:
    """Full pipeline: file path → (1, n_mels, target_frames) tensor."""
    waveform = load_audio(path)
    log_mel  = waveform_to_melspec(waveform)
    tensor   = melspec_to_tensor(log_mel)
    return pad_or_crop(tensor)
