#!/usr/bin/env python3
"""Generate audit samples from a Stable Audio 3 LoRA checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3.inference.sampling import sample_diffusion
from stable_audio_3.loading_utils import copy_state_dict
from stable_audio_3.model_configs import base_models
from stable_audio_3.models.lora.loader import load_and_apply_loras


DEFAULT_PROMPT = (
    "clear Horsfield's Bronze-cuckoo call, isolated foreground bird call, "
    "natural Australian woodland recording"
)


def slugify(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value.lower()).strip("_")


def load_base_model(model_name: str, device: torch.device):
    model_cfg = base_models[model_name]
    local_config, local_ckpt = model_cfg.resolve()
    with open(local_config) as f:
        model_config = json.load(f)
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_file(local_ckpt))
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    if model.pretransform is not None:
        model.pretransform.enable_grad = False
    return model, model_config


def render_spectrogram(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    mel = librosa.feature.melspectrogram(
        y=mono,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=0,
        fmax=min(sample_rate / 2, 11025),
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=min(sample_rate / 2, 11025),
        ax=ax,
        cmap="magma",
    )
    ax.set_title("SA3 LoRA generated event spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def generate_one(
    model,
    model_config: dict,
    prompt: str,
    seed: int,
    duration_s: float,
    steps: int,
    cfg_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    sample_rate = int(model.sample_rate)
    downsample = int(model.pretransform.downsampling_ratio)
    sample_size = (int(duration_s * sample_rate) // downsample) * downsample
    if sample_size <= 0:
        raise ValueError("duration is too short after downsampling alignment")

    latent_size = sample_size // downsample
    noise = torch.randn(
        1,
        int(model.io_channels),
        latent_size,
        device=device,
        dtype=torch.bfloat16,
    )
    conditioning = [{"prompt": prompt, "seconds_total": float(duration_s)}]
    negative = [{"prompt": "", "seconds_total": float(duration_s)}]

    cond_tensors = model.conditioner(conditioning, device)
    negative_tensors = model.conditioner(negative, device)
    zero_mask = torch.zeros(1, 1, latent_size, device=device, dtype=torch.bfloat16)
    zero_masked_input = torch.zeros(
        1,
        int(model.io_channels),
        latent_size,
        device=device,
        dtype=torch.bfloat16,
    )
    cond_tensors["inpaint_mask"] = [zero_mask, None]
    cond_tensors["inpaint_masked_input"] = [zero_masked_input, None]
    negative_tensors["inpaint_mask"] = [zero_mask, None]
    negative_tensors["inpaint_masked_input"] = [zero_masked_input, None]
    cond_inputs = {
        **model.get_conditioning_inputs(cond_tensors),
        **model.get_conditioning_inputs(negative_tensors, negative=True),
    }

    with torch.no_grad():
        audio = sample_diffusion(
            model.model,
            noise,
            cond_inputs,
            diffusion_objective=model.diffusion_objective,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_rate=sample_rate,
            pretransform=model.pretransform,
            mask_padding_attention=model.mask_padding_attention,
            use_effective_length_for_schedule=model.use_effective_length_for_schedule,
            dist_shift=model.sampling_dist_shift,
            sampler_type="euler",
            disable_tqdm=True,
        )
    audio_np = audio.detach().float().cpu().numpy()[0]
    audio_np = np.swapaxes(audio_np, 0, 1)
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak > 1.0:
        audio_np = audio_np / peak
    return np.clip(audio_np, -1.0, 1.0), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="small-sfx-base")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = load_base_model(args.model, device)
    load_and_apply_loras(model, [str(checkpoint)], "diffusion_cond")
    model.eval()

    rows = []
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        item_dir = out_dir / f"seed_{seed:04d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        wav_path = item_dir / "generated_event.wav"
        png_path = item_dir / "generated_event_spectrogram.png"
        meta_path = item_dir / "generated_event_metadata.json"

        audio, sample_rate = generate_one(
            model,
            model_config,
            args.prompt,
            seed,
            args.duration,
            args.steps,
            args.cfg_scale,
            device,
        )
        sf.write(wav_path, audio, sample_rate, subtype="PCM_16")
        render_spectrogram(audio, sample_rate, png_path)
        metadata = {
            "generator": "stable_audio_3_lora",
            "base_model": args.model,
            "checkpoint": str(checkpoint),
            "prompt": args.prompt,
            "seed": seed,
            "duration_s": args.duration,
            "sample_rate": sample_rate,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "audio_path": str(wav_path),
            "spectrogram_path": str(png_path),
        }
        meta_path.write_text(json.dumps(metadata, indent=2))
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "audio_path": str(wav_path),
                "spectrogram_path": str(png_path),
                "metadata_path": str(meta_path),
                "manual_verdict": "",
                "notes": "",
            }
        )

    audit_csv = out_dir / "sample_audit.csv"
    with audit_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "sample_audit_absolute.m3u").open("w") as f:
        for row in rows:
            f.write(str(Path(row["audio_path"]).resolve()) + "\n")
    print(f"Wrote audit bundle to {out_dir}")


if __name__ == "__main__":
    main()
