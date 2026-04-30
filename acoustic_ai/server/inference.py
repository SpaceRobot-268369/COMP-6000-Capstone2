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
_AI_ROOT          = Path(__file__).resolve().parent.parent
PROJECT_ROOT      = _AI_ROOT.parent
CHECKPOINT_DIR    = _AI_ROOT / "checkpoints" / "ambient"
DEFAULT_CKPT      = CHECKPOINT_DIR / "best.pt"
TEMPLATES_PATH    = _AI_ROOT / "data" / "ambient" / "latents" / "latent_templates.npy"
CLIPS_PATH        = _AI_ROOT / "data" / "ambient" / "latents" / "latent_clips.npy"
VOCODER_CKPT      = _AI_ROOT / "checkpoints" / "vocoder" / "best.pt"


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
    from modules.ambient.model import SoundscapeModel
    from modules.ambient.dataset import N_ENV_FEATURES

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
    from modules.ambient.preprocess import SPEC_CFG
    crop_seconds = args.get("crop_seconds", 30.0)
    if crop_seconds and crop_seconds > 0:
        return int(crop_seconds * SPEC_CFG["sample_rate"] / SPEC_CFG["hop_length"])
    from modules.ambient.preprocess import FRAMES_PER_CLIP
    return FRAMES_PER_CLIP


def _build_env_tensor(env_dict: dict) -> torch.Tensor:
    """Convert a plain env dict → (1, N_ENV_FEATURES) float32 tensor.

    Uses the same feature schema as dataset.py but without normalisation stats
    (caller should pass already-normalised numeric values, or use the dataset
    stats dict if available).  Categorical fields (season, sample_bin) are
    handled by one-hot / circular encoding regardless.
    """
    from modules.ambient.dataset import NUMERIC_COLS, CIRCULAR_COLS, ONEHOT_COLS

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


def mel_db_to_wav_ecoacoustic(mel_db: np.ndarray, sample_rate: int = 22_050) -> bytes:
    """Convert a (128, T) dB-scale mel-spectrogram to WAV using the fine-tuned
    ecoacoustic HiFi-GAN vocoder trained on site 257 audio.

    Requires acoustic_ai/vocoder_checkpoints/best.pt to exist (produced by
    running train_vocoder.py).  Raises FileNotFoundError if not available so
    the caller can fall back gracefully.

    Args:
        mel_db      : (128, T) array in dB scale [-80, 0]
        sample_rate : output sample rate (22 050 Hz — native, no resampling)

    Returns:
        WAV file as raw bytes
    """
    if not VOCODER_CKPT.exists():
        raise FileNotFoundError(f"Ecoacoustic vocoder not found at {VOCODER_CKPT}. "
                                 "Run train_vocoder.py first.")

    import io
    import soundfile as sf
    from modules.ambient.train_vocoder import HiFiGANGenerator, TOP_DB

    # Load generator
    ckpt  = torch.load(VOCODER_CKPT, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    model = HiFiGANGenerator(base_channels=saved.get("base_channels", 128))
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Normalise dB [-80, 0] → [0, 1]  (same as training)
    mel_norm = (mel_db + TOP_DB) / TOP_DB                    # (128, T)
    mel_tensor = torch.FloatTensor(mel_norm).unsqueeze(0)    # (1, 128, T)

    with torch.no_grad():
        waveform = model(mel_tensor).squeeze().cpu().numpy()  # (T_samples,)

    # Peak normalise
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak * 0.9

    buf = io.BytesIO()
    sf.write(buf, waveform.astype(np.float32), sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def mel_db_to_wav_hifigan(mel_db: np.ndarray, sample_rate: int = 22_050) -> bytes:
    """Convert a (128, T) dB-scale mel-spectrogram to WAV using SpeechT5 HiFi-GAN.

    Uses microsoft/speecht5_hifigan (public, no auth needed, ~50 MB).
    That model expects 80 mel bins at 16 kHz. We interpolate 128→80 and
    resample the output from 16 kHz → 22050 Hz.

    Args:
        mel_db      : (128, T) array in dB scale [-80, 0]
        sample_rate : output sample rate (resampled to this from 16 kHz)

    Returns:
        WAV file as raw bytes
    """
    import io
    import librosa
    import soundfile as sf
    import torch
    from scipy.ndimage import zoom
    from transformers import SpeechT5HifiGan

    HIFIGAN_SR = 16_000  # SpeechT5 HiFi-GAN native sample rate

    # 1. dB → power → interpolate 128 → 80 mel bins
    mel_power = librosa.db_to_power(mel_db)               # (128, T)
    mel_80    = zoom(mel_power, (80 / 128, 1), order=1)   # (80, T)

    # 2. log mel normalised to roughly [-11, 2] (matches SpeechT5 training scale)
    mel_log = np.log(np.maximum(mel_80, 1e-9))

    # 3. Load HiFi-GAN (cached by transformers after first download)
    model = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    model.eval()

    # 4. Vocode — SpeechT5HifiGan expects (batch, time, mel_bins)
    mel_tensor = torch.FloatTensor(mel_log).T.unsqueeze(0)  # (1, T, 80)
    with torch.no_grad():
        waveform = model(mel_tensor)                         # (1, T_wav) at 16 kHz

    waveform = waveform.squeeze().cpu().numpy()

    # 5. Resample 16 kHz → target sample_rate
    if HIFIGAN_SR != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=HIFIGAN_SR, target_sr=sample_rate)

    # 6. Normalise
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak * 0.9

    buf = io.BytesIO()
    sf.write(buf, waveform.astype(np.float32), sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def mel_db_to_wav(mel_db: np.ndarray, sample_rate: int = 22_050) -> bytes:
    """Convert a (128, T) dB-scale mel-spectrogram to a WAV file (bytes).

    Uses Griffin-Lim phase reconstruction — no extra model needed.
    Quality is robotic but functional for MVP; swap HiFi-GAN in Stage 3.

    Args:
        mel_db      : (n_mels, T) array in dB scale, typically [-80, 0]
        sample_rate : target sample rate (must match training config)

    Returns:
        WAV file as raw bytes (suitable for HTTP response or file write)
    """
    import io
    import librosa
    import soundfile as sf
    from modules.ambient.preprocess import SPEC_CFG

    # dB → power
    mel_power = librosa.db_to_power(mel_db)

    # mel power → linear STFT magnitude via mel filterbank pseudo-inverse
    mel_basis   = librosa.filters.mel(
        sr=sample_rate,
        n_fft=SPEC_CFG["n_fft"],
        n_mels=SPEC_CFG["n_mels"],
        fmin=SPEC_CFG["fmin"],
        fmax=SPEC_CFG["fmax"],
    )
    mel_pinv    = np.linalg.pinv(mel_basis)
    stft_mag    = np.maximum(mel_pinv @ mel_power, 0.0) ** 0.5  # amplitude

    # Griffin-Lim phase reconstruction
    waveform = librosa.griffinlim(
        stft_mag,
        n_iter=32,
        hop_length=SPEC_CFG["hop_length"],
        win_length=SPEC_CFG["n_fft"],
    )

    # Normalise to [-1, 1] to avoid clipping
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak * 0.9

    buf = io.BytesIO()
    sf.write(buf, waveform.astype(np.float32), sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


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
    from modules.ambient.preprocess import audio_to_tensor

    device = _get_device()
    model  = _load_model(checkpoint, device)

    mel = audio_to_tensor(wav_path)                   # (1, 128, T)
    mel = (mel - MEL_MIN_DB) / (MEL_MAX_DB - MEL_MIN_DB)  # normalise
    mel = mel.unsqueeze(0).to(device)                 # (1, 1, 128, T)

    env = _build_env_tensor(env_dict).to(device)      # (1, N)

    with torch.no_grad():
        mu, _ = model.encode(mel, env)                # use mu (deterministic)

    return mu.squeeze(0).cpu().numpy()


def _load_templates() -> Optional[dict]:
    """Load pre-computed mean latent templates if available."""
    if not TEMPLATES_PATH.exists():
        return None
    return np.load(str(TEMPLATES_PATH), allow_pickle=True).item()


def _load_clips() -> Optional[dict]:
    """Load per-clip latent database for nearest-neighbour generation."""
    if not CLIPS_PATH.exists():
        return None
    return np.load(str(CLIPS_PATH), allow_pickle=True).item()


def estimate_env_conditions(
    z_query: np.ndarray,
    clips: dict,
    top_k: int = 5,
) -> dict:
    """Estimate environmental conditions for a latent vector.

    Finds the top-k most similar training clips by cosine similarity in
    latent space, then averages their raw environmental values.  Returns a
    dict of human-readable env estimates plus a confidence score.

    Args:
        z_query : (latent_dim,) latent vector from encode_clip()
        clips   : dict loaded from latent_clips.npy (must include env_raw)
        top_k   : number of nearest neighbours to average

    Returns:
        dict with keys:
          temperature_c, humidity_pct, wind_speed_ms, precipitation_mm,
          solar_radiation_wm2, surface_pressure_kpa, temp_max_c, temp_min_c,
          wind_max_ms, days_since_rain, daylight_hours, hour_local,
          season, sample_bin, confidence (0–1)
    """
    latents = clips["latents"]   # (N, latent_dim)
    env_raw = clips.get("env_raw")
    if env_raw is None:
        return {}

    # Cosine similarity in latent space
    q_norm = z_query / (np.linalg.norm(z_query) + 1e-8)
    d_norm = latents / (np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8)
    sims   = np.nan_to_num(d_norm @ q_norm, nan=0.0)  # (N,)

    top_idx  = np.argsort(sims)[-top_k:]
    top_sims = sims[top_idx]
    confidence = float(np.mean(top_sims).clip(0, 1))

    # Average numeric values of top-k neighbours
    NUMERIC = [
        "temperature_c", "humidity_pct", "wind_speed_ms", "precipitation_mm",
        "solar_radiation_wm2", "surface_pressure_kpa", "temp_max_c", "temp_min_c",
        "wind_max_ms", "days_since_rain", "daylight_hours", "hour_local",
    ]
    estimates: dict = {}
    for col in NUMERIC:
        vals = [float(env_raw[i].get(col, 0.0)) for i in top_idx]
        estimates[col] = round(float(np.mean(vals)), 2)

    # Most common season / sample_bin among top-k
    from collections import Counter
    estimates["season"]     = Counter(env_raw[i]["season"]     for i in top_idx).most_common(1)[0][0]
    estimates["sample_bin"] = Counter(env_raw[i]["sample_bin"] for i in top_idx).most_common(1)[0][0]
    estimates["confidence"] = round(confidence, 3)

    return estimates


def _nearest_neighbour_latent(
    env_vec: np.ndarray,
    clips: dict,
    top_k: int = 10,
) -> np.ndarray:
    """Return the mean latent of the top-k clips most similar to env_vec.

    Uses cosine similarity on the encoded env feature vectors stored at
    precompute time — each dimension was encoded identically to training,
    so the similarity is meaningful across all env features.
    """
    latents  = clips["latents"]   # (N, latent_dim)
    env_vecs = clips["env_vecs"]  # (N, env_dim)

    # Cosine similarity: dot product of L2-normalised vectors
    q_norm = env_vec / (np.linalg.norm(env_vec) + 1e-8)
    d_norm = env_vecs / (np.linalg.norm(env_vecs, axis=1, keepdims=True) + 1e-8)
    sims   = np.nan_to_num(d_norm @ q_norm, nan=0.0, posinf=0.0, neginf=0.0)  # (N,)

    top_idx = np.argsort(sims)[-top_k:]             # indices of top-k
    return latents[top_idx].mean(axis=0)            # (latent_dim,)


def generate_spectrogram(
    env_dict: dict,
    checkpoint: Path = DEFAULT_CKPT,
    noise_std: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a mel-spectrogram from environmental conditions.

    Priority:
      1. Per-clip nearest-neighbour (latent_clips.npy) — uses all env features
      2. Group template (latent_templates.npy) — uses only season × sample_bin
      3. Pure N(0,1) sample — no grounding

    Args:
        env_dict   : environmental feature dict
        checkpoint : path to .pt checkpoint
        noise_std  : std of noise added to the anchor latent (controls variety)
        seed       : optional random seed for reproducibility

    Returns:
        numpy array of shape (128, T) in dB scale [-80, 0]
    """
    device = _get_device()
    model  = _load_model(checkpoint, device)
    clips  = _load_clips()

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    latent_dim = model.fusion.latent_dim

    if clips is not None:
        # Build the same env feature vector the dataset uses at training time
        env_tensor = _build_env_tensor(env_dict)          # (1, env_dim)
        env_vec    = env_tensor.squeeze(0).numpy()        # (env_dim,)
        mean_z     = _nearest_neighbour_latent(env_vec, clips)
    elif TEMPLATES_PATH.exists():
        templates  = _load_templates()
        season     = str(env_dict.get("season", "")).strip().lower()
        sample_bin = str(env_dict.get("sample_bin", "")).strip().lower()
        key        = f"{season}|{sample_bin}"
        if key in templates:
            mean_z = templates[key]
        else:
            matches = [v for k, v in templates.items() if k.endswith(f"|{sample_bin}")]
            mean_z  = np.mean(matches, axis=0) if matches else np.mean(list(templates.values()), axis=0)
    else:
        mean_z = None

    if mean_z is not None:
        mean_tensor = torch.FloatTensor(mean_z).unsqueeze(0).to(device)
        noise       = torch.randn(1, latent_dim, generator=rng).to(device) * noise_std
        z           = mean_tensor + noise
    else:
        z = torch.randn(1, latent_dim, generator=rng).to(device)

    with torch.no_grad():
        mel_norm = model.decoder(z)               # (1, 1, 128, T)

    mel_norm = mel_norm.squeeze().cpu().numpy()   # (128, T)
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
