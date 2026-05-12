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
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_AI_ROOT          = Path(__file__).resolve().parent.parent
PROJECT_ROOT      = _AI_ROOT.parent
CHECKPOINT_DIR    = PROJECT_ROOT / "model" / "production" / "ambient-vae"
DEFAULT_CKPT      = CHECKPOINT_DIR / "best.pt"
TEMPLATES_PATH    = _AI_ROOT / "data" / "ambient" / "latents" / "latent_templates.npy"
CLIPS_PATH        = _AI_ROOT / "data" / "ambient" / "latents" / "latent_clips.npy"
VOCODER_CKPT      = PROJECT_ROOT / "model" / "production" / "vocoder" / "best.pt"
AUDIOLDM2_BASE_MODEL = "cvssp/audioldm2"
AUDIOLDM2_LAYER_A_LORA_DIR = PROJECT_ROOT / "model" / "candidates" / "lucas" / "layer-a-audioldm2-raw-smoke"
LAYER_A_FIXED_PROMPT = (
    "quiet spring night ambient soundscape, Bowra dry woodland, Australia, "
    "distant environmental bed, no foreground events, no music, no machinery"
)
LAYER_A_OUTPUT_RMS = 0.0015
LAYER_A_HIGHPASS_HZ = 80.0
LAYER_A_GUIDANCE_SCALE = 2.0
LAYER_A_INFERENCE_STEPS = 100
LAYER_A_AUDIO_LENGTH_S = 10.0
LAYER_A_SMOKE_TESTS = {
    "smoke_test_1": {
        "label": "Dev - Layer A - Smoking Test 1 (spring night)",
        "model_status": "smoke_test_1_success",
        "prompt": LAYER_A_FIXED_PROMPT,
        "lora_dir": AUDIOLDM2_LAYER_A_LORA_DIR,
        "dataset": "resources/site_257_bowra-dry-a/smoking_test_dataset",
        "notes": [
            "Layer A dev endpoint is locked to the spring-night smoke-test prompt.",
            "No user-specified prompts are accepted while the model is trained on the small smoke dataset.",
            "Deprecated checkpoint audioldm2-lora-rms005-smoke should not be used for quality testing.",
        ],
    },
    "smoke_test_2": {
        "label": "Dev - Layer A - Smoking Test 2",
        "model_status": "smoke_test_2_insects_success",
        "prompt": (
            "summer afternoon insect-rich ambient soundscape, cicada and insect texture, "
            "Bowra dry woodland, Australia, dry hot air, distant environmental bed, "
            "no birds, no foreground events, no music, no machinery, no strong wind"
        ),
        "lora_dir": PROJECT_ROOT / "model" / "candidates" / "lucas" / "layer-a-audioldm2-insects-smoke",
        "dataset": "resources/site_257_bowra-dry-a/smoking_test2_insects_dataset",
        "notes": [
            "Layer A dev endpoint is locked to the insect/cicada smoke-test prompt.",
            "No user-specified prompts are accepted while the model is trained on the small smoke dataset.",
            "Training data excludes annotated event overlaps and strong-wind rows.",
        ],
    },
}


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

    Requires model/production/vocoder/best.pt to exist (produced by running
    train_vocoder.py). Raises FileNotFoundError if not available so the caller
    can fall back gracefully.

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


# ---------------------------------------------------------------------------
# AudioLDM2 Pipeline Cache
# ---------------------------------------------------------------------------
_audioldm2_pipelines = {}

def _get_audioldm2_pipeline(lora_dir: Path = AUDIOLDM2_LAYER_A_LORA_DIR):
    cache_key = str(lora_dir.resolve())
    if cache_key in _audioldm2_pipelines:
        return _audioldm2_pipelines[cache_key]

    from diffusers import AudioLDM2Pipeline
    from peft import PeftModel
    from transformers import GPT2LMHeadModel
    
    device = _get_device()
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pretrained_model_name = AUDIOLDM2_BASE_MODEL
    print(f"[INFO] Loading AudioLDM2 pipeline from {pretrained_model_name}...")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        pretrained_model_name, 
        torch_dtype=torch_dtype
    )
    
    pipeline.language_model = GPT2LMHeadModel.from_pretrained(
        pretrained_model_name,
        subfolder="language_model",
        torch_dtype=torch_dtype,
    )
    
    pipeline = pipeline.to(device)
    
    if lora_dir.exists():
        print(f"[INFO] Injecting LoRA weights from {lora_dir}...")
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, str(lora_dir))
    else:
        print(f"[WARN] LoRA weights not found at {lora_dir}. Using base model.")
        
    _audioldm2_pipelines[cache_key] = pipeline
    return pipeline


def _env_dict_to_prompt(env_dict: dict) -> str:
    """Format the environmental dictionary into a text prompt for AudioLDM2."""
    season = str(env_dict.get("season", "")).strip().lower()
    time_of_day = str(env_dict.get("sample_bin", "")).strip().lower()
    temp_c = env_dict.get("temperature_c", 20.0)
    humidity = env_dict.get("humidity_pct", 50.0)
    wind_speed = env_dict.get("wind_speed_ms", 0.0)
    
    # Map time of day
    if time_of_day == "dawn":
        time_desc = "early morning dawn"
    elif time_of_day == "morning":
        time_desc = "morning"
    elif time_of_day == "afternoon":
        time_desc = "afternoon"
    elif time_of_day == "night":
        time_desc = "night"
    else:
        time_desc = "daytime"
        
    # Map temperature
    if temp_c < 10:
        temp_desc = f"cold ({temp_c:.1f}°C)"
    elif temp_c < 20:
        temp_desc = f"cool ({temp_c:.1f}°C)"
    elif temp_c < 30:
        temp_desc = f"warm ({temp_c:.1f}°C)"
    else:
        temp_desc = f"hot ({temp_c:.1f}°C)"
        
    # Map humidity
    if humidity < 40:
        hum_desc = "dry air"
    elif humidity < 70:
        hum_desc = "moderate humidity"
    else:
        hum_desc = "humid"
        
    # Map wind
    if wind_speed < 1:
        wind_desc = "calm"
    elif wind_speed < 4:
        wind_desc = "light breeze"
    elif wind_speed < 8:
        wind_desc = "breezy"
    else:
        wind_desc = "windy"

    parts = [
        f"{season} {time_desc}",
        "ambient soundscape",
        "Bowra dry woodland, Australia",
        temp_desc,
        hum_desc,
        wind_desc
    ]
    
    return ", ".join(parts)


def _audio_stats(audio: np.ndarray, sample_rate: int) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    return {
        "sample_rate": sample_rate,
        "duration_s": float(audio.shape[0] / sample_rate),
        "min": float(audio.min()),
        "max": float(audio.max()),
        "mean": float(audio.mean()),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
        "clip_pct": float(np.mean(np.abs(audio) >= 0.999) * 100.0),
    }


def _highpass_audio(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return audio.astype(np.float32, copy=False)

    from scipy import signal

    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def _match_audio_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    if target_rms <= 0:
        return audio.astype(np.float32, copy=False)

    audio = audio.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if not np.isfinite(rms) or rms <= 1e-8:
        return audio

    audio = audio * (target_rms / rms)
    peak = float(np.max(np.abs(audio)))
    if np.isfinite(peak) and peak > 0.95:
        audio = audio * (0.95 / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _postprocess_layer_a_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict]:
    before = _audio_stats(audio, sample_rate)
    processed = _highpass_audio(audio, sample_rate, LAYER_A_HIGHPASS_HZ)
    processed = _match_audio_rms(processed, LAYER_A_OUTPUT_RMS)
    return processed, {
        "highpass_hz": LAYER_A_HIGHPASS_HZ,
        "output_target_rms": LAYER_A_OUTPUT_RMS,
        "before": before,
        "after": _audio_stats(processed, sample_rate),
    }


def _waveform_to_melspec_at_sr(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    from modules.ambient.diffusion.layer_a_visualization import waveform_to_layer_a_mel_db

    return waveform_to_layer_a_mel_db(waveform, sample_rate)


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return buf.read()


def generate_layer_a_smoke_test_audio(
    smoke_test_id: str = "smoke_test_1",
    seed: Optional[int] = None,
) -> tuple[np.ndarray, bytes, dict]:
    """Generate the dev Layer A ambient bed with a fixed smoke-test prompt.

    The prompt is intentionally locked while the Layer A model is trained only on
    the tiny smoke dataset. User-specified prompts should not be accepted here.
    """
    if smoke_test_id not in LAYER_A_SMOKE_TESTS:
        raise ValueError(f"Unknown Layer A smoke test: {smoke_test_id}")

    config = LAYER_A_SMOKE_TESTS[smoke_test_id]
    prompt = config["prompt"]
    lora_dir = config["lora_dir"]

    device = _get_device()
    pipeline = _get_audioldm2_pipeline(lora_dir)

    rng = torch.Generator(device)
    if seed is not None:
        rng.manual_seed(seed)
    else:
        rng.seed()

    print(f"[INFO] Generating Layer A {smoke_test_id} with fixed prompt: '{prompt}'")
    raw_audio = pipeline(
        prompt,
        num_inference_steps=LAYER_A_INFERENCE_STEPS,
        audio_length_in_s=LAYER_A_AUDIO_LENGTH_S,
        guidance_scale=LAYER_A_GUIDANCE_SCALE,
        generator=rng,
    ).audios[0]

    sample_rate = int(pipeline.vocoder.config.sampling_rate)
    audio, postprocess = _postprocess_layer_a_audio(raw_audio, sample_rate)
    mel_db = _waveform_to_melspec_at_sr(audio, sample_rate)
    wav_bytes = _wav_bytes(audio, sample_rate)

    metadata = {
        "generator": "audioldm2_lora",
        "smoke_test_id": smoke_test_id,
        "label": config["label"],
        "model_status": config["model_status"],
        "prompt_locked": True,
        "fixed_prompt": prompt,
        "pretrained_model_name": AUDIOLDM2_BASE_MODEL,
        "lora_dir": str(lora_dir),
        "checkpoint_exists": lora_dir.exists(),
        "dataset": config["dataset"],
        "seed": seed,
        "num_inference_steps": LAYER_A_INFERENCE_STEPS,
        "audio_length_in_s": LAYER_A_AUDIO_LENGTH_S,
        "guidance_scale": LAYER_A_GUIDANCE_SCALE,
        "spectrogram_renderer": "modules.ambient.diffusion.layer_a_visualization",
        "spectrogram_type": "log_mel",
        "audio": _audio_stats(audio, sample_rate),
        "postprocess": postprocess,
        "notes": config["notes"],
    }

    return mel_db, wav_bytes, metadata


def generate_layer_a_ambient_audio(seed: Optional[int] = None) -> tuple[np.ndarray, bytes, dict]:
    return generate_layer_a_smoke_test_audio("smoke_test_1", seed=seed)


def generate_ambient_audio(
    env_dict: dict,
    noise_std: float = 0.3,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, bytes]:
    """Generate ambient audio from environmental conditions using AudioLDM2.

    Args:
        env_dict   : environmental feature dict
        noise_std  : unused for AudioLDM2 (kept for API compatibility)
        seed       : optional random seed for reproducibility

    Returns:
        tuple containing:
          - mel_db: numpy array of shape (128, T) in dB scale
          - wav_bytes: generated WAV file as raw bytes
    """
    import io
    import soundfile as sf
    from modules.ambient.preprocess import waveform_to_melspec
    
    device = _get_device()
    pipeline = _get_audioldm2_pipeline(AUDIOLDM2_LAYER_A_LORA_DIR)

    prompt = _env_dict_to_prompt(env_dict)
    print(f"[INFO] Generating audio for prompt: '{prompt}'")
    
    rng = torch.Generator(device)
    if seed is not None:
        rng.manual_seed(seed)
    else:
        rng.seed()

    # Generate the audio
    # audio output is a numpy array of shape (time,) at 16000 Hz
    audio = pipeline(
        prompt,
        num_inference_steps=100,  # reduced from 200 for faster inference in API
        audio_length_in_s=10.0,
        guidance_scale=3.5,
        generator=rng,
    ).audios[0]

    # AudioLDM2 generates at 16kHz. 
    audioldm_sr = 16000
    target_sr = 22050
    
    # Resample to 22050 Hz to match project defaults
    import librosa
    audio_resampled = librosa.resample(audio, orig_sr=audioldm_sr, target_sr=target_sr)
    
    # Save to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio_resampled, target_sr, format="WAV")
    buf.seek(0)
    wav_bytes = buf.read()
    
    # Compute mel spectrogram using project defaults
    mel_db = waveform_to_melspec(audio_resampled)

    return mel_db, wav_bytes


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

    mel, wav = generate_ambient_audio(dummy_env, seed=42)
    print(f"Generated  : shape={mel.shape}  min={mel.min():.1f}  max={mel.max():.1f} dB")
    print(f"Generated audio bytes: {len(wav)} bytes")
    print("OK — inference module working.")
