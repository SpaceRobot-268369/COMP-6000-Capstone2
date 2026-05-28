"""Layer C attempt handler — AudioGen LoRA, boobook owl smoke test.

Registry-facing interface (see ``server/registry.py``):
    load(checkpoint_dir, params, extra) -> state
    generate(state, seed, prompt=None, **kw) -> dict

Returns generated event audio as in-memory WAV bytes (no disk writes).
"""

from __future__ import annotations

import io
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .layer_a_visualization import waveform_to_layer_a_mel_db


# --- AudioCraft compatibility shims ---------------------------------------


def _install_xformers_stub() -> None:
    """AudioCraft imports xformers unconditionally; stub it on macOS."""
    if "xformers" in sys.modules:
        return
    xformers = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")

    def _missing(*_a, **_kw):
        raise ImportError("xformers not installed (mac smoke path).")

    class LowerTriangularMask: ...
    ops.unbind = torch.unbind
    ops.memory_efficient_attention = _missing
    ops.LowerTriangularMask = LowerTriangularMask
    xformers.ops = ops
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops


def _patch_audiocraft_mps_autocast() -> None:
    if not torch.backends.mps.is_available():
        return
    from audiocraft.modules import conditioners
    from audiocraft.models import genmodel
    from audiocraft.utils import autocast as autocast_module

    original = autocast_module.TorchAutocast

    class MpsSafeTorchAutocast:
        def __init__(self, *a, **kw):
            device_type = kw.get("device_type") or (a[0] if a else None)
            self.autocast = (
                None if kw.get("enabled", True) and device_type == "mps"
                else original(*a, **kw)
            )

        def __enter__(self):
            return None if self.autocast is None else self.autocast.__enter__()

        def __exit__(self, *a, **kw):
            return None if self.autocast is None else self.autocast.__exit__(*a, **kw)

    autocast_module.TorchAutocast = MpsSafeTorchAutocast
    conditioners.TorchAutocast = MpsSafeTorchAutocast
    genmodel.TorchAutocast = MpsSafeTorchAutocast


# --- helpers --------------------------------------------------------------


def _choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _audio_stats(audio: np.ndarray, sample_rate: int) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    return {
        "sample_rate": sample_rate,
        "duration_s": float(audio.shape[-1] / sample_rate),
        "rms":  float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
    }


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
    sf.write(buf, audio.astype(np.float32), sample_rate, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return buf.read()


# --- state + handler ------------------------------------------------------


@dataclass
class _State:
    model: object       # AudioGen
    sample_rate: int
    params: dict
    checkpoint: Path


def load(checkpoint_dir: Path, params: dict, extra: dict | None = None) -> _State:
    if checkpoint_dir is None or not Path(checkpoint_dir).exists():
        raise FileNotFoundError(f"AudioGen LoRA checkpoint missing: {checkpoint_dir}")

    _install_xformers_stub()
    from audiocraft.models import AudioGen
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    _patch_audiocraft_mps_autocast()

    base = params.get("base_model", "facebook/audiogen-medium")
    device = _choose_device()
    print(f"[INFO] Loading AudioGen from {base}...")
    audiogen = AudioGen.get_pretrained(base, device=device)

    print(f"[INFO] Injecting LoRA from {checkpoint_dir}...")
    config = LoraConfig.from_pretrained(str(checkpoint_dir))
    audiogen.lm = inject_adapter_in_model(config, audiogen.lm)
    adapter = Path(checkpoint_dir) / "adapter_model.safetensors"
    if not adapter.exists():
        raise FileNotFoundError(f"missing LoRA weights: {adapter}")
    set_peft_model_state_dict(audiogen.lm, load_file(adapter))
    audiogen.lm.to(device).eval()

    return _State(
        model=audiogen,
        sample_rate=int(audiogen.sample_rate),
        params=params,
        checkpoint=Path(checkpoint_dir),
    )


def generate(
    state: _State,
    seed: Optional[int] = None,
    prompt: Optional[str] = None,
    duration: Optional[float] = None,
    **_ignored,
) -> dict:
    """Generate one event clip.

    Accepts: seed (required), prompt (optional override), duration (optional).
    Defaults come from registry.yaml params.
    """
    p = state.params
    prompt          = prompt or p.get("default_prompt", "a single boobook owl call")
    duration        = float(duration if duration is not None else p.get("duration", 5.0))
    guidance_scale  = float(p.get("guidance_scale", 3.0))
    temperature     = float(p.get("temperature", 1.0))
    top_k           = int(p.get("top_k", 250))
    top_p           = float(p.get("top_p", 0.0))
    target_rms      = float(p.get("output_target_rms", 0.02))

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    state.model.set_generation_params(
        use_sampling=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        duration=duration,
        cfg_coef=guidance_scale,
    )

    print(f"[INFO] Generating Layer C event: '{prompt}'  (seed={seed})")
    with torch.no_grad():
        generated = state.model.generate([prompt], progress=False)

    audio = generated[0].detach().float().cpu().numpy()
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    sr = state.sample_rate
    before = _audio_stats(audio, sr)
    audio = _match_rms(audio, target_rms)
    mel_db = waveform_to_layer_a_mel_db(audio, sr)

    metadata = {
        "generator":     "audiogen_lora",
        "prompt":        prompt,
        "base_model":    p.get("base_model"),
        "seed":          seed,
        "duration":      duration,
        "guidance_scale": guidance_scale,
        "temperature":   temperature,
        "top_k":         top_k,
        "top_p":         top_p,
        "audio":         _audio_stats(audio, sr),
        "postprocess":   {"output_target_rms": target_rms, "before": before},
    }
    return {
        "wav_bytes": _wav_bytes(audio, sr),
        "mel_db":    mel_db,
        "metadata":  metadata,
    }
