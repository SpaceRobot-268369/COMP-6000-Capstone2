"""Inference script for Layer B wind generation with AudioLDM2 + LoRA."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AudioLDM2Pipeline
from peft import PeftModel
from scipy import signal
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from debug_paths import DEBUG_ROOT  # noqa: E402
from .layer_a_visualization import (  # noqa: E402
    render_layer_a_mel_png_bytes,
    waveform_to_layer_a_mel_db,
)

DEFAULT_OUTPUT_DIR = DEBUG_ROOT / "layer_b" / "audioldm2_wind" / "samples"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio with AudioLDM2 + LoRA")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the target wind texture",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="cvssp/audioldm2",
        help="Base AudioLDM2 model",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="Path to the directory containing trained LoRA weights",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the generated WAV file. Defaults to debug/layer_b/audioldm2_wind/samples/...",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated WAV/PNG/JSON artifacts when --output_path is not set",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional readable output folder/name")
    parser.add_argument("--num_inference_steps", type=int, default=200, help="Denoising steps")
    parser.add_argument("--audio_length_in_s", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument(
        "--output_target_rms",
        type=float,
        default=0.02,
        help="Final WAV RMS target for generated wind stems. Use <=0 to disable.",
    )
    parser.add_argument(
        "--highpass_hz",
        type=float,
        default=80.0,
        help="High-pass cutoff for sub-bass rumble cleanup. Use <=0 to disable.",
    )
    return parser.parse_args()


def _slugify(value: str, max_len: int = 64) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return value[:max_len] or "prompt"


def _lora_slug(lora_dir: str) -> str:
    return _slugify(Path(lora_dir).name, max_len=80)


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)

    run_name = args.run_name or f"{_slugify(args.prompt)}__seed_{args.seed:04d}"
    return args.output_dir / _lora_slug(args.lora_dir) / run_name / "generated_wind.wav"


def _render_spectrogram(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    mel_db = waveform_to_layer_a_mel_db(audio, sample_rate)
    png_bytes = render_layer_a_mel_png_bytes(
        mel_db,
        duration_s=float(audio.shape[0] / sample_rate),
    )
    path.write_bytes(png_bytes)


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


def _postprocess_audio(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    before = _audio_stats(audio, sample_rate)
    processed = _highpass(audio, sample_rate, args.highpass_hz)
    processed = _match_rms(processed, args.output_target_rms)
    return processed, {
        "highpass_hz": args.highpass_hz if args.highpass_hz > 0 else None,
        "output_target_rms": args.output_target_rms if args.output_target_rms > 0 else None,
        "before": before,
        "after": _audio_stats(processed, sample_rate),
    }

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    from transformers import GPT2LMHeadModel

    print(f"Loading base pipeline from {args.pretrained_model_name}...")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        args.pretrained_model_name, 
        torch_dtype=torch_dtype
    )
    
    pipeline.language_model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model_name,
        subfolder="language_model",
        torch_dtype=torch_dtype,
    )

    pipeline = pipeline.to(device)
    
    print(f"Loading LoRA weights from {args.lora_dir}...")
    # Inject LoRA weights into the UNet
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.lora_dir)
    
    print(f"Generating audio for prompt: '{args.prompt}'")
    generator = torch.Generator(device).manual_seed(args.seed)
    
    # Generate the audio
    raw_audio = pipeline(
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        audio_length_in_s=args.audio_length_in_s,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).audios[0]
    
    # Save the output bundle
    output_path = _resolve_output_path(args)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = pipeline.vocoder.config.sampling_rate
    audio, postprocess = _postprocess_audio(raw_audio, sample_rate, args)
    wavfile.write(output_path, rate=sample_rate, data=audio.astype(np.float32))

    spec_path = output_path.with_name(f"{output_path.stem}_spectrogram.png")
    meta_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    _render_spectrogram(audio, sample_rate, spec_path)
    meta_path.write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "pretrained_model_name": args.pretrained_model_name,
                "lora_dir": args.lora_dir,
                "num_inference_steps": args.num_inference_steps,
                "audio_length_in_s": args.audio_length_in_s,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "spectrogram_renderer": "acoustic_ai.layers.layer_b.attempts.murphy__smoke_1__audioldm2_wind.code.layer_a_visualization",
                "spectrogram_type": "log_mel",
                "layer_b": {
                    "weather_type": "wind",
                    "mode": "generate",
                    "attempt": "murphy__smoke_1__audioldm2_wind",
                },
                "audio": _audio_stats(audio, sample_rate),
                "postprocess": postprocess,
                "artifacts": {
                    "wav": str(output_path),
                    "spectrogram": str(spec_path),
                },
            },
            indent=2,
        )
    )

    print(f"Saved generated audio bundle to {output_dir}")
    print(f"  WAV:  {output_path.name}")
    print(f"  PNG:  {spec_path.name}")
    print(f"  JSON: {meta_path.name}")

if __name__ == "__main__":
    main()
