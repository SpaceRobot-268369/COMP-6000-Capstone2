"""Layer B wind-intensity bank handler (mvp_1).

Routes generation by wind intensity:
  - medium: learned LoRA adapter
  - heavy:  learned LoRA adapter
  - light:  derived from medium adapter via gentler inference/postprocess params
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .layer_a_visualization import waveform_to_layer_a_mel_db

INTENSITIES = ("light", "medium", "heavy")


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
    bank_root: Path
    profiles: dict
    loaded_adapters: list[str] = field(default_factory=list)
    default_intensity: str = "medium"


def load(checkpoint_dir: Optional[Path], params: dict, extra: dict | None = None) -> _State:
    from diffusers import AudioLDM2Pipeline
    from peft import PeftModel
    from transformers import GPT2LMHeadModel

    base_model = params.get("base_model", "cvssp/audioldm2")
    profiles = dict(params.get("intensity_profiles") or {})
    if not profiles:
        raise ValueError("wind_intensity_bank requires `intensity_profiles` in registry params.")
    if checkpoint_dir is None:
        raise ValueError("wind_intensity_bank requires a checkpoint dir (bank root).")
    bank_root = Path(checkpoint_dir)

    device = _get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[INFO] Loading AudioLDM2 pipeline from {base_model}...")
    pipeline = AudioLDM2Pipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipeline.language_model = GPT2LMHeadModel.from_pretrained(
        base_model, subfolder="language_model", torch_dtype=dtype
    )
    pipeline = pipeline.to(device)

    # Load only adapters that are actually trained models (declared by adapter key).
    loaded: list[str] = []
    peft_unet = None
    adapters_to_load = sorted(
        {
            str(v["adapter"]).strip()
            for v in profiles.values()
            if isinstance(v, dict) and str(v.get("adapter", "")).strip()
        }
    )
    for adapter_name in adapters_to_load:
        adapter_dir = bank_root / "adapters" / adapter_name
        weight = adapter_dir / "adapter_model.safetensors"
        if not weight.is_file():
            print(f"[WARN] adapter {adapter_name!r} missing at {weight} — skipping.")
            continue
        if peft_unet is None:
            peft_unet = PeftModel.from_pretrained(
                pipeline.unet, str(adapter_dir), adapter_name=adapter_name
            )
        else:
            peft_unet.load_adapter(str(adapter_dir), adapter_name=adapter_name)
        loaded.append(adapter_name)

    if peft_unet is None:
        raise FileNotFoundError(
            f"No adapters materialised under {bank_root}/adapters. "
            "Run `dvc pull` for adapter_model.safetensors files."
        )
    pipeline.unet = peft_unet

    default_intensity = str(params.get("default_intensity", "medium")).strip().lower()
    if default_intensity not in profiles:
        default_intensity = "medium" if "medium" in profiles else next(iter(profiles.keys()))

    sample_rate = int(pipeline.vocoder.config.sampling_rate)
    print(f"[INFO] Loaded adapters={loaded}; default_intensity={default_intensity}")
    return _State(
        pipeline=pipeline,
        sample_rate=sample_rate,
        params=params,
        bank_root=bank_root,
        profiles=profiles,
        loaded_adapters=loaded,
        default_intensity=default_intensity,
    )


def _resolve_intensity(state: _State, intensity: Optional[str]) -> str:
    if intensity is None:
        return state.default_intensity
    key = str(intensity).strip().lower()
    if key not in INTENSITIES:
        raise ValueError(f"unknown intensity {intensity!r}; expected one of {INTENSITIES}")
    if key not in state.profiles:
        raise ValueError(f"intensity profile {key!r} not declared in registry params")
    return key


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


def _highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return audio.astype(np.float32, copy=False)
    from scipy import signal

    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def _lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return audio.astype(np.float32, copy=False)
    from scipy import signal

    sos = signal.butter(4, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
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


def generate(
    state: _State,
    seed: Optional[int] = None,
    intensity: Optional[str] = None,
    wind_intensity: Optional[str] = None,
    weather_type: Optional[str] = None,
    **_ignored,
) -> dict:
    # Accept weather_type for compatibility with existing Layer B dev UI.
    if weather_type and str(weather_type).strip().lower() != "wind":
        raise ValueError("wind_intensity_bank only supports weather_type='wind'")

    resolved = _resolve_intensity(state, wind_intensity or intensity)
    profile = dict(state.profiles[resolved] or {})

    # Resolve adapter routing.
    adapter_name = str(profile.get("adapter", "")).strip()
    derived_from = str(profile.get("derived_from", "")).strip()
    if adapter_name:
        if adapter_name not in state.loaded_adapters:
            raise FileNotFoundError(
                f"adapter {adapter_name!r} for intensity={resolved!r} not loaded. Run dvc pull."
            )
        state.pipeline.unet.set_adapter(adapter_name)
        routed_adapter = adapter_name
        derived = False
    elif derived_from:
        if derived_from not in state.loaded_adapters:
            raise FileNotFoundError(
                f"derived_from adapter {derived_from!r} for intensity={resolved!r} not loaded."
            )
        state.pipeline.unet.set_adapter(derived_from)
        routed_adapter = derived_from
        derived = True
    else:
        raise ValueError(f"intensity profile {resolved!r} must declare `adapter` or `derived_from`")

    prompt = str(profile["prompt"]).strip()
    guidance_scale = float(profile.get("guidance_scale", state.params.get("guidance_scale", 3.0)))
    num_inference_steps = int(
        profile.get("num_inference_steps", state.params.get("num_inference_steps", 200))
    )
    audio_length_in_s = float(
        profile.get("audio_length_in_s", state.params.get("audio_length_in_s", 8.0))
    )
    output_target_rms = float(
        profile.get("output_target_rms", state.params.get("output_target_rms", 0.06))
    )
    highpass_hz = float(profile.get("highpass_hz", state.params.get("highpass_hz", 80.0)))
    lowpass_hz = float(profile.get("lowpass_hz", state.params.get("lowpass_hz", 0.0)))
    negative_prompt = str(
        profile.get("negative_prompt", state.params.get("negative_prompt", ""))
    ).strip()

    denoise_enabled = bool(profile.get("denoise_enabled", state.params.get("denoise_enabled", True)))
    denoise_strength = float(profile.get("denoise_strength", state.params.get("denoise_strength", 0.15)))
    denoise_quantile = float(
        profile.get("denoise_noise_quantile", state.params.get("denoise_noise_quantile", 0.2))
    )
    denoise_floor_ratio = float(
        profile.get("denoise_floor_ratio", state.params.get("denoise_floor_ratio", 0.40))
    )
    denoise_hop_length = int(
        profile.get("denoise_hop_length", state.params.get("denoise_hop_length", 512))
    )
    fade_ms = float(profile.get("fade_ms", state.params.get("fade_ms", 80.0)))

    device = _get_device()
    rng = torch.Generator(device)
    if seed is not None:
        rng.manual_seed(int(seed))
    else:
        rng.seed()

    raw = state.pipeline(
        prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=num_inference_steps,
        audio_length_in_s=audio_length_in_s,
        guidance_scale=guidance_scale,
        generator=rng,
    ).audios[0]

    sr = state.sample_rate
    before = _audio_stats(raw, sr)
    audio = _highpass(raw, sr, highpass_hz)
    after_highpass = _audio_stats(audio, sr)
    if lowpass_hz > 0:
        audio = _lowpass(audio, sr, lowpass_hz)
    after_lowpass = _audio_stats(audio, sr)
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
        "generator": "audioldm2_lora_layer_b_wind_intensity_bank",
        "prompt_locked": True,
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "base_model": state.params.get("base_model"),
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "audio_length_in_s": audio_length_in_s,
        "guidance_scale": guidance_scale,
        "weather_type": "wind",
        "requested_intensity": intensity,
        "requested_wind_intensity": wind_intensity,
        "resolved_intensity": resolved,
        "adapter": routed_adapter,
        "derived": derived,
        "audio": _audio_stats(audio, sr),
        "postprocess": {
            "highpass_hz": highpass_hz,
            "lowpass_hz": lowpass_hz if lowpass_hz > 0 else None,
            "output_target_rms": output_target_rms,
            "before": before,
            "after_highpass": after_highpass,
            "after_lowpass": after_lowpass,
            "denoise": {
                "enabled": denoise_enabled,
                "strength": denoise_strength if denoise_enabled else 0.0,
                "noise_quantile": denoise_quantile if denoise_enabled else None,
                "floor_ratio": denoise_floor_ratio if denoise_enabled else None,
                "hop_length": denoise_hop_length if denoise_enabled else None,
                "after_denoise": after_denoise,
            },
            "fade_ms": fade_ms,
            "after_fade": after_fade,
        },
        "layer_b": {
            "weather_type": "wind",
            "mode": "generate",
            "attempt": "murphy__mvp_1__wind_intensity_bank",
        },
    }
    return {
        "wav_bytes": _wav_bytes(audio, sr),
        "mel_db": mel_db,
        "metadata": metadata,
    }
