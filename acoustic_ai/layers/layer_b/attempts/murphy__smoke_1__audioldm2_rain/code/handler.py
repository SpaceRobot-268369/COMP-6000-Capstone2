"""Layer B attempt handler — AudioLDM2 LoRA, rain-only smoke test.

Registry-facing interface (see ``server/registry.py``):
    load(checkpoint_dir, params, extra) -> state
    generate(state, seed, **runtime_params)   -> dict

All training-time and inference-time hyperparameters come from the matching
``registry.yaml`` entry — never from constants in this file. The dev API
contract is enforced by accepting only ``seed`` as a runtime param.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .layer_a_visualization import waveform_to_layer_a_mel_db


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class _State:
    pipeline: object
    sample_rate: int
    params: dict
    checkpoint: Optional[Path]


def load(checkpoint_dir: Optional[Path], params: dict, extra: dict | None = None) -> _State:
    """Build the AudioLDM2 pipeline + inject LoRA. Cached by the registry."""
    from diffusers import AudioLDM2Pipeline
    from peft import PeftModel
    from transformers import GPT2LMHeadModel

    base_model = params.get("base_model", "cvssp/audioldm2")
    device = _get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[INFO] Loading AudioLDM2 pipeline from {base_model}...")
    pipeline = AudioLDM2Pipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipeline.language_model = GPT2LMHeadModel.from_pretrained(
        base_model, subfolder="language_model", torch_dtype=dtype,
    )
    pipeline = pipeline.to(device)

    if checkpoint_dir and Path(checkpoint_dir).exists():
        adapter = Path(checkpoint_dir) / "adapter_model.safetensors"
        if not adapter.is_file():
            raise FileNotFoundError(
                f"LoRA weights missing at {adapter}. The checkpoint directory "
                "exists but adapter_model.safetensors is not on disk — it is "
                "DVC-tracked. Run `dvc pull "
                f"{checkpoint_dir}/adapter_model.safetensors` to fetch it."
            )
        print(f"[INFO] Injecting LoRA weights from {checkpoint_dir}...")
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, str(checkpoint_dir))
    else:
        print(f"[WARN] LoRA weights not found at {checkpoint_dir}. Using base model.")

    sample_rate = int(pipeline.vocoder.config.sampling_rate)
    return _State(pipeline=pipeline, sample_rate=sample_rate, params=params,
                  checkpoint=checkpoint_dir)


def _audio_stats(audio: np.ndarray, sample_rate: int) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    return {
        "sample_rate": sample_rate,
        "duration_s": float(audio.shape[0] / sample_rate),
        "min":  float(audio.min()),
        "max":  float(audio.max()),
        "mean": float(audio.mean()),
        "rms":  float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
        "clip_pct": float(np.mean(np.abs(audio) >= 0.999) * 100.0),
    }


def _highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return audio.astype(np.float32, copy=False)
    from scipy import signal
    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def _spectral_denoise(
    audio: np.ndarray,
    sample_rate: int,
    *,
    strength: float = 0.25,
    noise_quantile: float = 0.2,
    floor_ratio: float = 0.15,
    nperseg: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """Gentle STFT-domain denoise to lower stationary hiss/noise floor."""
    from scipy import signal

    if strength <= 0:
        return audio.astype(np.float32, copy=False)

    audio = audio.astype(np.float32, copy=False)
    if audio.size < nperseg:
        return audio

    noverlap = max(0, nperseg - hop_length)
    window = "hann"
    _, _, stft = signal.stft(
        audio,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(stft)
    phase = np.exp(1j * np.angle(stft))

    frame_energy = np.sqrt(np.mean(mag * mag, axis=0))
    if frame_energy.size == 0:
        return audio
    threshold = np.quantile(frame_energy, np.clip(noise_quantile, 0.05, 0.5))
    noise_mask = frame_energy <= threshold
    if np.any(noise_mask):
        noise_profile = np.median(mag[:, noise_mask], axis=1, keepdims=True)
    else:
        noise_profile = np.median(mag, axis=1, keepdims=True)

    cleaned_mag = mag - (float(strength) * noise_profile)
    cleaned_mag = np.maximum(cleaned_mag, float(floor_ratio) * noise_profile)
    cleaned_stft = cleaned_mag * phase

    _, denoised = signal.istft(
        cleaned_stft,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        input_onesided=True,
        boundary=True,
    )

    if denoised.shape[0] < audio.shape[0]:
        denoised = np.pad(denoised, (0, audio.shape[0] - denoised.shape[0]))
    elif denoised.shape[0] > audio.shape[0]:
        denoised = denoised[: audio.shape[0]]
    return denoised.astype(np.float32, copy=False)


def _apply_fade(audio: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    if fade_ms <= 0:
        return audio.astype(np.float32, copy=False)
    audio = audio.astype(np.float32, copy=False)
    n = int(sample_rate * (fade_ms / 1000.0))
    n = max(1, min(n, audio.shape[0] // 2))
    if n <= 1:
        return audio
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = audio.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def _match_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
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


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return buf.read()


def generate(state: _State, seed: Optional[int] = None, **_ignored) -> dict:
    """Generate one weather rain stem.

    Only ``seed`` is accepted from the runtime — every other parameter
    comes from the attempt's registry entry. This is the Layer B generate
    smoke contract for the rain-only LoRA.
    """
    p = state.params
    prompt              = p["prompt"]
    negative_prompt     = p.get("negative_prompt")
    guidance_scale      = float(p.get("guidance_scale", 2.0))
    num_inference_steps = int(p.get("num_inference_steps", 100))
    audio_length_in_s   = float(p.get("audio_length_in_s", 10.0))
    output_target_rms   = float(p.get("output_target_rms", 0.0))
    highpass_hz         = float(p.get("highpass_hz", 0.0))
    denoise_enabled     = bool(p.get("denoise_enabled", False))
    denoise_strength    = float(p.get("denoise_strength", 0.25))
    denoise_quantile    = float(p.get("denoise_noise_quantile", 0.2))
    denoise_floor_ratio = float(p.get("denoise_floor_ratio", 0.15))
    denoise_hop_length  = int(p.get("denoise_hop_length", 512))
    fade_ms             = float(p.get("fade_ms", 80.0))

    device = _get_device()
    rng = torch.Generator(device)
    if seed is not None:
        rng.manual_seed(int(seed))
    else:
        rng.seed()

    print(f"[INFO] Generating: '{prompt[:80]}…'  (seed={seed})")
    raw = state.pipeline(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        audio_length_in_s=audio_length_in_s,
        guidance_scale=guidance_scale,
        generator=rng,
    ).audios[0]

    sr = state.sample_rate
    before = _audio_stats(raw, sr)
    audio = _highpass(raw, sr, highpass_hz)
    after_highpass = _audio_stats(audio, sr)
    if denoise_enabled:
        audio = _spectral_denoise(
            audio,
            sr,
            strength=denoise_strength,
            noise_quantile=denoise_quantile,
            floor_ratio=denoise_floor_ratio,
            hop_length=denoise_hop_length,
        )
    after_denoise = _audio_stats(audio, sr)
    audio = _apply_fade(audio, sr, fade_ms=fade_ms)
    after_fade = _audio_stats(audio, sr)
    audio = _match_rms(audio, output_target_rms)
    mel_db = waveform_to_layer_a_mel_db(audio, sr)

    metadata = {
        "generator":             "audioldm2_lora_layer_b_rain",
        "prompt_locked":         True,
        "prompt":                prompt,
        "negative_prompt":       negative_prompt,
        "base_model":            p.get("base_model"),
        "seed":                  seed,
        "num_inference_steps":   num_inference_steps,
        "audio_length_in_s":     audio_length_in_s,
        "guidance_scale":        guidance_scale,
        "audio": _audio_stats(audio, sr),
        "postprocess": {
            "highpass_hz":       highpass_hz,
            "output_target_rms": output_target_rms,
            "before":            before,
            "after_highpass":    after_highpass,
            "denoise": {
                "enabled":      denoise_enabled,
                "strength":     denoise_strength if denoise_enabled else 0.0,
                "noise_quantile": denoise_quantile if denoise_enabled else None,
                "floor_ratio":  denoise_floor_ratio if denoise_enabled else None,
                "hop_length": denoise_hop_length if denoise_enabled else None,
                "after_denoise": after_denoise,
            },
            "fade_ms": fade_ms,
            "after_fade": after_fade,
        },
        "layer_b": {
            "weather_type": "rain",
            "mode": "generate",
            "attempt": "murphy__smoke_1__audioldm2_rain",
        },
    }
    return {
        "wav_bytes": _wav_bytes(audio, sr),
        "mel_db":    mel_db,
        "metadata":  metadata,
    }
