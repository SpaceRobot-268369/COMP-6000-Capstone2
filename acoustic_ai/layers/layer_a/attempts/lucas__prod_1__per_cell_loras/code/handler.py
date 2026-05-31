"""Layer A attempt handler — per-cell LoRA bank (mvp_2).

This is a *router* handler. Unlike the single-adapter smoke handlers, it loads
the frozen AudioLDM2 base once and registers all 16 (season, diel) LoRA
adapters as named PEFT adapters on the shared UNet. At generation time it
switches to the requested cell with ``set_adapter`` (instant — no reload) and
uses that cell's locked prompt.

Registry-facing interface (see ``server/registry.py``):
    load(checkpoint_dir, params, extra) -> state
    generate(state, seed, season=..., diel=...) -> dict

Dev-generation contract (see CLAUDE.md): the runtime accepts only ``seed`` plus
the cell selector ``(season, diel)``. Prompt, guidance, steps, length, RMS and
high-pass are all owned server-side via the registry ``params`` block. If no
cell is given, the registry ``default_cell`` is used.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .layer_a_visualization import waveform_to_layer_a_mel_db

# Valid axis values — used to validate the (season, diel) selector.
SEASONS = ("spring", "summer", "autumn", "winter")
DIELS = ("dawn", "morning", "afternoon", "night")


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
    cells: dict           # cell_name -> {"prompt": str}
    loaded_cells: list = field(default_factory=list)
    default_cell: str = ""


def load(checkpoint_dir: Optional[Path], params: dict, extra: dict | None = None) -> _State:
    """Build the AudioLDM2 pipeline and register every cell adapter by name."""
    from diffusers import AudioLDM2Pipeline
    from peft import PeftModel
    from transformers import GPT2LMHeadModel

    base_model = params.get("base_model", "cvssp/audioldm2")
    cells = dict(params.get("cells") or {})
    if not cells:
        raise ValueError(
            "mvp_2 per-cell bank requires a non-empty `cells:` map in the "
            "registry params (cell_name -> {prompt})."
        )
    if checkpoint_dir is None:
        raise ValueError("mvp_2 per-cell bank requires a checkpoint dir (bank root).")
    bank_root = Path(checkpoint_dir)

    device = _get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[INFO] Loading AudioLDM2 pipeline from {base_model}...")
    pipeline = AudioLDM2Pipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipeline.language_model = GPT2LMHeadModel.from_pretrained(
        base_model, subfolder="language_model", torch_dtype=dtype,
    )
    pipeline = pipeline.to(device)

    # Register each cell's adapter under its own name. The first call wraps the
    # UNet in a PeftModel; subsequent ones attach more named adapters.
    loaded: list[str] = []
    peft_unet = None
    for cell in sorted(cells):
        adapter_dir = bank_root / cell
        weight = adapter_dir / "adapter_model.safetensors"
        if not weight.is_file():
            print(f"[WARN] {cell}: adapter missing at {weight} — skipping. "
                  f"Run `dvc pull {adapter_dir}/adapter_model.safetensors`.")
            continue
        if peft_unet is None:
            peft_unet = PeftModel.from_pretrained(
                pipeline.unet, str(adapter_dir), adapter_name=cell)
        else:
            peft_unet.load_adapter(str(adapter_dir), adapter_name=cell)
        loaded.append(cell)

    if peft_unet is None:
        raise FileNotFoundError(
            f"No cell adapters materialised under {bank_root}. Run `dvc pull` "
            "for model/candidates/lucas/mvp_2__per_cell_loras/*/adapter_model.safetensors."
        )
    pipeline.unet = peft_unet

    default_cell = params.get("default_cell") or (
        "spring_night" if "spring_night" in loaded else loaded[0])
    print(f"[INFO] Loaded {len(loaded)} cell adapters; default={default_cell}.")

    sample_rate = int(pipeline.vocoder.config.sampling_rate)
    return _State(pipeline=pipeline, sample_rate=sample_rate, params=params,
                  bank_root=bank_root, cells=cells, loaded_cells=loaded,
                  default_cell=default_cell)


def _resolve_cell(state: _State, season: Optional[str], diel: Optional[str]) -> str:
    """Map a (season, diel) selector to a loaded cell name; validate."""
    if season is None and diel is None:
        return state.default_cell
    if season is None or diel is None:
        raise ValueError("Provide both `season` and `diel`, or neither.")
    season = str(season).strip().lower()
    diel = str(diel).strip().lower()
    if season not in SEASONS:
        raise ValueError(f"unknown season {season!r}; expected one of {SEASONS}")
    if diel not in DIELS:
        raise ValueError(f"unknown diel {diel!r}; expected one of {DIELS}")
    cell = f"{season}_{diel}"
    if cell not in state.cells:
        raise ValueError(f"no cell {cell!r} in the bank")
    if cell not in state.loaded_cells:
        raise FileNotFoundError(
            f"cell {cell!r} is declared but its adapter is not loaded "
            "(weights not on disk — run `dvc pull`).")
    return cell


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


def generate(state: _State, seed: Optional[int] = None,
             season: Optional[str] = None, diel: Optional[str] = None,
             **_ignored) -> dict:
    """Generate one ambient bed from the (season, diel) cell adapter.

    Runtime accepts only ``seed`` + the ``(season, diel)`` selector. Every
    other parameter comes from the registry ``params`` block.
    """
    p = state.params
    cell = _resolve_cell(state, season, diel)
    state.pipeline.unet.set_adapter(cell)

    prompt              = state.cells[cell]["prompt"]
    guidance_scale      = float(p.get("guidance_scale", 2.0))
    num_inference_steps = int(p.get("num_inference_steps", 200))
    audio_length_in_s   = float(p.get("audio_length_in_s", 10.0))
    output_target_rms   = float(p.get("output_target_rms", 0.0))
    highpass_hz         = float(p.get("highpass_hz", 0.0))

    device = _get_device()
    rng = torch.Generator(device)
    if seed is not None:
        rng.manual_seed(int(seed))
    else:
        rng.seed()

    print(f"[INFO] cell={cell} seed={seed} :: '{prompt[:70]}…'")
    raw = state.pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        audio_length_in_s=audio_length_in_s,
        guidance_scale=guidance_scale,
        generator=rng,
    ).audios[0]

    sr = state.sample_rate
    before = _audio_stats(raw, sr)
    audio = _highpass(raw, sr, highpass_hz)
    audio = _match_rms(audio, output_target_rms)
    mel_db = waveform_to_layer_a_mel_db(audio, sr)

    metadata = {
        "generator":             "audioldm2_lora_per_cell",
        "prompt_locked":         True,
        "cell":                  cell,
        "season":                cell.split("_")[0],
        "diel":                  cell.split("_")[1],
        "prompt":                prompt,
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
        },
    }
    return {
        "wav_bytes": _wav_bytes(audio, sr),
        "mel_db":    mel_db,
        "metadata":  metadata,
    }
