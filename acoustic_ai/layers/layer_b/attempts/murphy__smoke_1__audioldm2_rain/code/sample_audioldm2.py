"""Inference script for Layer B rain generation with AudioLDM2 + LoRA."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AudioLDM2Pipeline
from peft import PeftModel
from scipy.io import wavfile

ACOUSTIC_AI_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ACOUSTIC_AI_ROOT))

from debug_paths import DEBUG_ROOT  # noqa: E402
try:  # noqa: E402
    from .bwe import apply_bwe, linear_phase_highpass
    from .layer_a_visualization import (
        render_layer_a_mel_png_bytes,
        waveform_to_layer_a_mel_db,
    )
except ImportError:  # direct script execution fallback
    from bwe import apply_bwe, linear_phase_highpass  # type: ignore
    from layer_a_visualization import (  # type: ignore
        render_layer_a_mel_png_bytes,
        waveform_to_layer_a_mel_db,
    )

DEFAULT_OUTPUT_DIR = DEBUG_ROOT / "layer_b" / "audioldm2_rain" / "samples"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio with AudioLDM2 + LoRA")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the target rain texture",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Optional negative prompt to suppress hiss/static/birds/insects.",
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
        help="Path to save the generated WAV file. Defaults to debug/layer_b/audioldm2_rain/samples/...",
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
        default=0.05,
        help="Final WAV RMS target for generated rain stems. Use <=0 to disable.",
    )
    parser.add_argument(
        "--highpass_hz",
        type=float,
        default=80.0,
        help="High-pass cutoff for sub-bass rumble cleanup. Use <=0 to disable.",
    )
    parser.add_argument(
        "--fade_ms",
        type=float,
        default=20.0,
        help="Final fade-in/out after BWE, high-pass, and RMS matching.",
    )
    parser.add_argument(
        "--disable_bwe",
        action="store_true",
        help="Disable the validated 24 kHz BWE postprocess.",
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
    return args.output_dir / _lora_slug(args.lora_dir) / run_name / "generated_rain.wav"


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

    return linear_phase_highpass(audio, sample_rate, cutoff_hz, numtaps=513)


def _match_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    if target_rms <= 0:
        return audio.astype(np.float32, copy=False)

    audio = audio.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if not np.isfinite(rms) or rms <= 1e-8:
        return audio

    return (audio * (target_rms / rms)).astype(np.float32)


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


def _peak_limit(audio: np.ndarray, ceiling: float = 0.98) -> tuple[np.ndarray, float]:
    audio = audio.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak <= ceiling:
        return audio, 1.0
    gain = ceiling / peak
    return (audio * gain).astype(np.float32), gain


def _postprocess_audio(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    before = _audio_stats(audio, sample_rate)
    bwe_metadata = {"enabled": False}
    if not args.disable_bwe:
        processed, sample_rate, bwe_metadata = apply_bwe(
            audio, sample_rate, seed=args.seed
        )
    else:
        processed = audio.astype(np.float32, copy=False)
    after_bwe = _audio_stats(processed, sample_rate)
    processed = _highpass(processed, sample_rate, args.highpass_hz)
    after_highpass = _audio_stats(processed, sample_rate)
    processed = _match_rms(processed, args.output_target_rms)
    after_rms = _audio_stats(processed, sample_rate)
    processed = _apply_fade(processed, sample_rate, args.fade_ms)
    after_fade = _audio_stats(processed, sample_rate)
    processed, limiter_gain = _peak_limit(processed, ceiling=0.98)
    return processed, {
        "sample_rate": sample_rate,
        "post_order": ["BWE mix", "80 Hz highpass", "RMS match", "20 ms fade", "peak limit if needed"],
        "highpass_hz": args.highpass_hz if args.highpass_hz > 0 else None,
        "output_target_rms": args.output_target_rms if args.output_target_rms > 0 else None,
        "fade_ms": args.fade_ms,
        "before": before,
        "bwe": bwe_metadata,
        "after_bwe": after_bwe,
        "after_highpass": after_highpass,
        "after_rms": after_rms,
        "after_fade": after_fade,
        "limiter": {
            "ceiling": 0.98,
            "gain": limiter_gain,
            "after_limiter": _audio_stats(processed, sample_rate),
        },
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
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
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
    sample_rate = int(postprocess["sample_rate"])
    wavfile.write(output_path, rate=sample_rate, data=audio.astype(np.float32))

    if output_path.name == "audio.wav":
        spec_path = output_path.with_name("spectrogram.png")
        meta_path = output_path.with_name("metadata.json")
    else:
        spec_path = output_path.with_name(f"{output_path.stem}_spectrogram.png")
        meta_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    _render_spectrogram(audio, sample_rate, spec_path)
    meta_path.write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "pretrained_model_name": args.pretrained_model_name,
                "lora_dir": args.lora_dir,
                "num_inference_steps": args.num_inference_steps,
                "audio_length_in_s": args.audio_length_in_s,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "spectrogram_renderer": "acoustic_ai.layers.layer_b.attempts.murphy__smoke_1__audioldm2_rain.code.layer_a_visualization",
                "spectrogram_type": "log_mel",
                "layer_b": {
                    "weather_type": "rain",
                    "mode": "generate",
                    "attempt": "murphy__smoke_1__audioldm2_rain",
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
