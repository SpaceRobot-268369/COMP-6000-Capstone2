"""Inference: env conditions → latent (via diffusion) → mel → wav.

Builds the 12-dim conditioning vector matching the schema in `dataset.py`,
samples a latent with DDIM, decodes through the frozen VAE decoder, and
vocodes with the existing HiFi-GAN.

Usage:
  python3 acoustic_ai/modules/ambient/diffusion/sample.py \\
      --hour 6 --month 10 --season spring --diel dawn \\
      --out generated.wav
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acoustic_ai.modules.ambient.diffusion.dataset import COND_COLUMNS
from acoustic_ai.modules.ambient.diffusion.model import LatentDenoiser
from acoustic_ai.modules.ambient.diffusion.schedule import NoiseSchedule, ddim_sample

SEASONS = ("spring", "summer", "autumn", "winter")
DIEL = ("dawn", "morning", "afternoon", "night")


def build_cond(*, hour: float, month: int, season: str, diel: str) -> np.ndarray:
    """Build a single (12,) conditioning vector matching COND_COLUMNS order."""
    if season not in SEASONS:
        raise ValueError(f"season must be one of {SEASONS}")
    if diel not in DIEL:
        raise ValueError(f"diel must be one of {DIEL}")

    # Cyclic encodings
    hour_rad = 2 * math.pi * (hour % 24) / 24.0
    month_rad = 2 * math.pi * ((month - 1) % 12) / 12.0
    cyc = [math.sin(hour_rad), math.cos(hour_rad),
           math.sin(month_rad), math.cos(month_rad)]

    season_oh = [1.0 if s == season else 0.0 for s in SEASONS]
    diel_oh = [1.0 if d == diel else 0.0 for d in DIEL]

    values = cyc + season_oh + diel_oh
    assert len(values) == len(COND_COLUMNS), \
        f"cond length {len(values)} != COND_COLUMNS {len(COND_COLUMNS)}"
    return np.asarray(values, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=PROJECT_ROOT / "acoustic_ai" / "checkpoints" / "ambient_diffusion" / "best.pt")
    p.add_argument("--vae-checkpoint", type=Path,
                   default=PROJECT_ROOT / "acoustic_ai" / "checkpoints" / "ambient" / "best.pt")
    p.add_argument("--vocoder-checkpoint", type=Path,
                   default=PROJECT_ROOT / "acoustic_ai" / "checkpoints" / "vocoder" / "best.pt")
    p.add_argument("--params", type=Path, default=PROJECT_ROOT / "params.yaml")
    p.add_argument("--out", type=Path, required=True)

    p.add_argument("--hour", type=float, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--season", choices=SEASONS, required=True)
    p.add_argument("--diel", choices=DIEL, required=True)

    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def pick_device(arg: str | None) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_denoiser(ckpt_path: Path, cfg: dict, device: torch.device) -> LatentDenoiser:
    model = LatentDenoiser(
        latent_dim=cfg["latent_dim"],
        cond_dim=cfg["cond_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema") or ckpt.get("model") or ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_vae_decoder(ckpt_path: Path, device: torch.device):
    """Load just the decoder part of the trained SoundscapeModel."""
    # Add ambient module dir to path so its internal flat imports (e.g. preprocess) work
    ambient_dir = str(PROJECT_ROOT / "acoustic_ai" / "modules" / "ambient")
    if ambient_dir not in sys.path:
        sys.path.insert(0, ambient_dir)

    from acoustic_ai.modules.ambient.model import SoundscapeModel  # noqa: E402

    model = SoundscapeModel().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    model.load_state_dict(state)
    model.eval()
    return model.decoder


def vocode(mel: torch.Tensor, vocoder_ckpt: Path, device: torch.device) -> np.ndarray:
    """mel: (1, n_mels, T) torch tensor in dB. Returns waveform ndarray (samples,)."""
    # Defer import — vocoder may not be available in all environments.
    try:
        from acoustic_ai.server.inference import mel_db_to_wav_ecoacoustic
        import io
        import soundfile as sf
    except Exception:  # pragma: no cover
        raise RuntimeError(
            "Could not import vocoder helper. Ensure acoustic_ai.server.inference "
            "is accessible and mel_db_to_wav_ecoacoustic is defined."
        )

    # Convert tensor to numpy (128, T) for the inference helper
    mel_np = mel.squeeze(0).cpu().numpy()

    # The inference helper returns WAV bytes. We need to decode them back
    # to a numpy array for peak normalisation and final saving.
    wav_bytes = mel_db_to_wav_ecoacoustic(mel_np)
    wav, sr = sf.read(io.BytesIO(wav_bytes))
    return wav


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.params))["diffusion"]
    if args.cfg_scale is not None:
        cfg["cfg_scale"] = args.cfg_scale
    if args.steps is not None:
        cfg["num_inference_steps"] = args.steps

    device = pick_device(args.device)
    print(f"device: {device}")

    # ---- conditioning ----
    cond_np = build_cond(
        hour=args.hour, month=args.month,
        season=args.season, diel=args.diel,
    )
    cond = torch.from_numpy(cond_np).unsqueeze(0).to(device)       # (1, 12)
    print(f"cond: {cond_np.tolist()}")

    # ---- denoiser ----
    denoiser = load_denoiser(args.checkpoint, cfg, device)

    # ---- diffusion sampling ----
    schedule = NoiseSchedule(
        num_train_timesteps=cfg["num_train_timesteps"],
        schedule=cfg["schedule"],
    ).to(device)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    z0 = ddim_sample(
        denoiser,
        cond=cond,
        schedule=schedule,
        num_inference_steps=cfg["num_inference_steps"],
        cfg_scale=cfg["cfg_scale"],
        generator=gen,
    )
    print(f"sampled latent: shape={tuple(z0.shape)}, "
          f"mean={z0.mean().item():.3f}, std={z0.std().item():.3f}")

    # ---- VAE decode ----
    decoder = load_vae_decoder(args.vae_checkpoint, device)
    with torch.no_grad():
        mel = decoder(z0)                                          # (1, 1, 128, T)
    mel = mel.squeeze(1)                                           # (1, 128, T)
    print(f"mel: shape={tuple(mel.shape)}")

    # ---- Vocode ----
    wav = vocode(mel, args.vocoder_checkpoint, device)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)

    # Light peak normalisation to avoid clipping on save
    peak = float(np.max(np.abs(wav)) + 1e-9)
    if peak > 1.0:
        wav = wav / peak

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav, 22050)
    print(f"wrote {args.out}  ({wav.size / 22050:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
