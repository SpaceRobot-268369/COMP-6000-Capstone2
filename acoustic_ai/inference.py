"""Inference helpers for the trained SoundscapeModel.

Provides two high-level functions used by the backend API:

  encode_clip(wav_path, env_dict, checkpoint)
    → latent vector (256-dim numpy array)
    → used by POST /api/analysis

  generate_spectrogram(env_dict, checkpoint)
    → mel-spectrogram (128 × T numpy array, dB scale)
    → used by POST /api/generation

Both functions accept env_dict as a plain Python dict with the same keys
as the training manifest (temperature_c, humidity_pct, season, etc.).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CKPT   = CHECKPOINT_DIR / "best.pt"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(checkpoint: Path, device: torch.device):
    """Load SoundscapeModel from checkpoint. Returns model in eval mode."""
    from model import SoundscapeModel
    from dataset import N_ENV_FEATURES

    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args  = ckpt.get("args", {})

    model = SoundscapeModel(
        env_dim=N_ENV_FEATURES,
        embed_dim=args.get("embed_dim", 512),
        latent_dim=args.get("latent_dim", 256),
        target_frames=args.get("crop_frames") or _default_target_frames(args),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _default_target_frames(args: dict) -> int:
    from preprocess import SPEC_CFG
    crop_seconds = args.get("crop_seconds", 30.0)
    if crop_seconds and crop_seconds > 0:
        return int(crop_seconds * SPEC_CFG["sample_rate"] / SPEC_CFG["hop_length"])
    from preprocess import FRAMES_PER_CLIP
    return FRAMES_PER_CLIP


def _build_env_tensor(env_dict: dict) -> torch.Tensor:
    """Convert a plain env dict → (1, N_ENV_FEATURES) float32 tensor.

    Uses the same feature schema as dataset.py but without normalisation stats
    (caller should pass already-normalised numeric values, or use the dataset
    stats dict if available).  Categorical fields (season, sample_bin) are
    handled by one-hot / circular encoding regardless.
    """
    from dataset import NUMERIC_COLS, CIRCULAR_COLS, ONEHOT_COLS

    parts: list[float] = []

    for col in NUMERIC_COLS:
        parts.append(float(env_dict.get(col, 0.0)))

    for col, period in CIRCULAR_COLS:
        val = float(env_dict.get(col, 0.0))
        parts.append(math.sin(2 * math.pi * val / period))
        parts.append(math.cos(2 * math.pi * val / period))

    for col, categories in ONEHOT_COLS.items():
        val = str(env_dict.get(col, "")).strip().lower()
        parts.extend([1.0 if val == c else 0.0 for c in categories])

    return torch.tensor(parts, dtype=torch.float32).unsqueeze(0)  # (1, N)


MEL_MIN_DB = -80.0
MEL_MAX_DB  =  0.0


def _denormalise(mel_norm: np.ndarray) -> np.ndarray:
    """[0,1] → dB scale [-80, 0]."""
    return mel_norm * (MEL_MAX_DB - MEL_MIN_DB) + MEL_MIN_DB


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_clip(
    wav_path: str,
    env_dict: dict,
    checkpoint: Path = DEFAULT_CKPT,
) -> np.ndarray:
    """Encode an audio clip into a latent vector.

    Args:
        wav_path   : path to .wav file
        env_dict   : environmental feature dict (same keys as training manifest)
        checkpoint : path to .pt checkpoint (defaults to best.pt)

    Returns:
        numpy array of shape (latent_dim,) — typically (256,)
    """
    from preprocess import audio_to_tensor

    device = _get_device()
    model  = _load_model(checkpoint, device)

    mel = audio_to_tensor(wav_path)                   # (1, 128, T)
    mel = (mel - MEL_MIN_DB) / (MEL_MAX_DB - MEL_MIN_DB)  # normalise
    mel = mel.unsqueeze(0).to(device)                 # (1, 1, 128, T)

    env = _build_env_tensor(env_dict).to(device)      # (1, N)

    with torch.no_grad():
        z = model.encode(mel, env)                    # (1, latent_dim)

    return z.squeeze(0).cpu().numpy()


def generate_spectrogram(
    env_dict: dict,
    checkpoint: Path = DEFAULT_CKPT,
    noise_std: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a mel-spectrogram from environmental conditions.

    The encoder is bypassed — a random latent vector is sampled and decoded.
    This is a simple baseline for speculative generation; replace with a
    proper generative prior (VAE / diffusion) in Stage 3.

    Args:
        env_dict   : environmental feature dict
        checkpoint : path to .pt checkpoint
        noise_std  : std of the random latent noise (controls variety)
        seed       : optional random seed for reproducibility

    Returns:
        numpy array of shape (128, T) in dB scale [-80, 0]
    """
    device = _get_device()
    model  = _load_model(checkpoint, device)

    env = _build_env_tensor(env_dict).to(device)      # (1, N)

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    latent_dim = model.fusion.net[-2].out_features    # read from model
    z = torch.randn(1, latent_dim, generator=rng).to(device) * noise_std

    with torch.no_grad():
        mel_norm = model.decoder(z)                   # (1, 1, 128, T)

    mel_norm = mel_norm.squeeze().cpu().numpy()       # (128, T)
    return _denormalise(mel_norm)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Checkpoint :", DEFAULT_CKPT)
    print("Device     :", _get_device())

    if not DEFAULT_CKPT.exists():
        print("ERROR: best.pt not found — run training first.")
        sys.exit(1)

    # Test generation (no audio needed)
    dummy_env = {
        "temperature_c": 22.0, "humidity_pct": 60.0, "wind_speed_ms": 3.0,
        "precipitation_mm": 0.0, "solar_radiation_wm2": 400.0,
        "cloud_clearness_index": 0.6, "surface_pressure_kpa": 101.3,
        "temp_max_c": 28.0, "temp_min_c": 15.0, "precipitation_daily_mm": 0.0,
        "wind_max_ms": 7.0, "days_since_rain": 5.0, "daylight_hours": 11.5,
        "hour_utc": 6.0, "hour_local": 16.0,
        "wind_direction_deg": 180.0, "month": 9.0, "day_of_year": 260.0,
        "season": "spring", "sample_bin": "afternoon",
    }

    mel = generate_spectrogram(dummy_env, seed=42)
    print(f"Generated  : shape={mel.shape}  min={mel.min():.1f}  max={mel.max():.1f} dB")
    print("OK — inference module working.")
