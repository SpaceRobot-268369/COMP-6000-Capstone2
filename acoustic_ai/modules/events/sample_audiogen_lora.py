"""Generate Layer C event samples with AudioCraft AudioGen + LoRA."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from debug_paths import DEBUG_ROOT  # noqa: E402
from modules.ambient.diffusion.layer_a_visualization import (  # noqa: E402
    LAYER_A_SPEC_CMAP,
    LAYER_A_SPEC_DPI,
    LAYER_A_SPEC_FIGSIZE,
    waveform_to_layer_a_mel_db,
)
from modules.ambient.preprocess import SPEC_CFG  # noqa: E402


DEFAULT_OUTPUT_DIR = DEBUG_ROOT / "layer_c" / "audiogen" / "samples"
DEFAULT_MODEL = "facebook/audiogen-medium"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Layer C event audio with AudioCraft AudioGen + LoRA."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--pretrained_model_name", default=DEFAULT_MODEL)
    parser.add_argument("--lora_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seeds. Keeps the model loaded and writes one bundle per seed.",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output_target_rms",
        type=float,
        default=0.02,
        help="Final WAV RMS target for foreground event snippets. Use <=0 to disable.",
    )
    return parser.parse_args()


def choose_device(value: str | None) -> str:
    if value:
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def install_xformers_import_stub() -> None:
    import sys
    import types

    if "xformers" in sys.modules:
        return

    xformers = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")

    def _missing(*_args, **_kwargs):
        raise ImportError(
            "xformers is not installed; this macOS AudioGen smoke path must use "
            "AudioCraft configs with memory_efficient/checkpointing xformers disabled."
        )

    class LowerTriangularMask:
        pass

    ops.unbind = torch.unbind
    ops.memory_efficient_attention = _missing
    ops.LowerTriangularMask = LowerTriangularMask
    xformers.ops = ops
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops


def patch_audiocraft_mps_autocast() -> None:
    if not torch.backends.mps.is_available():
        return

    from audiocraft.modules import conditioners
    from audiocraft.models import genmodel
    from audiocraft.utils import autocast as autocast_module

    original_torch_autocast = autocast_module.TorchAutocast

    class MpsSafeTorchAutocast:
        def __init__(self, *args, **kwargs):
            device_type = kwargs.get("device_type")
            if device_type is None and args:
                device_type = args[0]
            if kwargs.get("enabled", True) and device_type == "mps":
                self.autocast = None
            else:
                self.autocast = original_torch_autocast(*args, **kwargs)

        def __enter__(self):
            if self.autocast is None:
                return None
            return self.autocast.__enter__()

        def __exit__(self, *args, **kwargs):
            if self.autocast is None:
                return None
            return self.autocast.__exit__(*args, **kwargs)

    autocast_module.TorchAutocast = MpsSafeTorchAutocast
    conditioners.TorchAutocast = MpsSafeTorchAutocast
    genmodel.TorchAutocast = MpsSafeTorchAutocast


def slugify(value: str, max_len: int = 64) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return value[:max_len] or "sample"


def parse_seed_values(args: argparse.Namespace) -> list[int]:
    if not args.seeds:
        return [args.seed]
    values: list[int] = []
    for value in args.seeds.split(","):
        value = value.strip()
        if value:
            values.append(int(value))
    if not values:
        raise ValueError("--seeds was provided but no integer seeds were parsed")
    return values


def resolve_output_path(args: argparse.Namespace, seed: int) -> Path:
    if args.output_path:
        if args.seeds:
            raise ValueError("--output_path cannot be used with --seeds; use --output_dir instead")
        return args.output_path
    run_name = args.run_name or f"{slugify(args.prompt)}__seed_{seed:04d}"
    if args.run_name and args.seeds:
        run_name = f"{args.run_name}__seed_{seed:04d}"
    lora_name = slugify(args.lora_dir.name, max_len=80)
    return args.output_dir / lora_name / run_name / "generated_event.wav"


def audio_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    audio = np.asarray(audio, dtype=np.float32)
    return {
        "sample_rate": sample_rate,
        "duration_s": float(audio.shape[-1] / sample_rate),
        "min": float(audio.min()),
        "max": float(audio.max()),
        "mean": float(audio.mean()),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
        "clip_pct": float(np.mean(np.abs(audio) >= 0.999) * 100.0),
    }


def match_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
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


def render_spectrogram(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel_db = waveform_to_layer_a_mel_db(audio, sample_rate)
    duration_s = float(audio.shape[-1] / sample_rate)
    fig, ax = plt.subplots(figsize=LAYER_A_SPEC_FIGSIZE)
    img = ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        extent=[0, duration_s, 0, SPEC_CFG["n_mels"]],
        cmap=LAYER_A_SPEC_CMAP,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_title("Layer C - Generated Event Spectrogram")
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=LAYER_A_SPEC_DPI)
    plt.close(fig)
    buf.seek(0)
    path.write_bytes(buf.read())


def load_lora_into_lm(lm: torch.nn.Module, lora_dir: Path) -> torch.nn.Module:
    config = LoraConfig.from_pretrained(lora_dir)
    lm = inject_adapter_in_model(config, lm)
    adapter_path = lora_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"missing LoRA weights: {adapter_path}")
    set_peft_model_state_dict(lm, load_file(adapter_path))
    return lm


def main() -> int:
    args = parse_args()

    install_xformers_import_stub()
    try:
        from audiocraft.models import AudioGen
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "AudioCraft is required for facebook/audiogen-medium. "
            "Run this from the AudioGen environment, e.g. "
            "`cd acoustic_ai && ./.venv-audiogen/bin/python "
            "modules/events/sample_audiogen_lora.py ...`."
        ) from exc
    patch_audiocraft_mps_autocast()

    device = choose_device(args.device)
    seeds = parse_seed_values(args)

    print(f"Loading AudioCraft AudioGen from {args.pretrained_model_name}...")
    audiogen = AudioGen.get_pretrained(args.pretrained_model_name, device=device)
    print(f"Loading LoRA adapter from {args.lora_dir}...")
    audiogen.lm = load_lora_into_lm(audiogen.lm, args.lora_dir)
    audiogen.lm.to(device)
    audiogen.lm.eval()

    audiogen.set_generation_params(
        use_sampling=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        duration=args.duration,
        cfg_coef=args.guidance_scale,
    )

    sample_rate = int(audiogen.sample_rate)
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"Generating event audio for prompt: {args.prompt!r} seed={seed}")
        with torch.no_grad():
            generated = audiogen.generate([args.prompt], progress=True)

        audio = generated[0].detach().float().cpu().numpy()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        before = audio_stats(audio, sample_rate)
        audio = match_rms(audio, args.output_target_rms)
        after = audio_stats(audio, sample_rate)

        output_path = resolve_output_path(args, seed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(output_path, sample_rate, audio.astype(np.float32))

        spec_path = output_path.with_name(f"{output_path.stem}_spectrogram.png")
        meta_path = output_path.with_name(f"{output_path.stem}_metadata.json")
        render_spectrogram(audio, sample_rate, spec_path)
        meta_path.write_text(
            json.dumps(
                {
                    "prompt": args.prompt,
                    "pretrained_model_name": args.pretrained_model_name,
                    "lora_dir": str(args.lora_dir),
                    "seed": seed,
                    "duration": args.duration,
                    "guidance_scale": args.guidance_scale,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "output_target_rms": args.output_target_rms,
                    "audio_before_postprocess": before,
                    "audio": after,
                    "artifacts": {
                        "wav": str(output_path),
                        "spectrogram": str(spec_path),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Saved generated event bundle to {output_path.parent}")
        print(f"  WAV:  {output_path.name}")
        print(f"  PNG:  {spec_path.name}")
        print(f"  JSON: {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
