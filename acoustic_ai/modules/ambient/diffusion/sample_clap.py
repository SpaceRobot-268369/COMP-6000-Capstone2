"""CLAP-conditioned inference: text prompt → latent → mel → waveform.

Full pipeline:
  text prompt
    → CLAP text encoder (laion/clap-htsat-unfused)
    → 512-dim embedding
    → DDIM reverse diffusion (LatentDenoiser)
    → 256-dim latent
    → frozen VAE decoder
    → (1, 128, T) mel-spectrogram
    → HiFi-GAN vocoder
    → waveform WAV

Usage:
    python3 acoustic_ai/modules/ambient/diffusion/sample_clap.py \\
        --prompt "spring night, ambient soundscape, Bowra dry woodland, Australia, cool 15C" \\
        --out debug/layer_a/clap_diffusion/samples/generated_spring_night.wav

    # Multiple prompts in one pass:
    python3 acoustic_ai/modules/ambient/diffusion/sample_clap.py \\
        --prompt "spring night, ambient soundscape, Bowra dry woodland, Australia, cool 15C" \\
                 "spring night, ambient soundscape, Bowra dry woodland, Australia, warm 20C" \\
        --out debug/layer_a/clap_diffusion/samples/sample_a.wav debug/layer_a/clap_diffusion/samples/sample_b.wav

    # Dry-run — just prints shapes, does not write files:
    python3 acoustic_ai/modules/ambient/diffusion/sample_clap.py --prompt "..." --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai" / "modules" / "ambient"))

from debug_paths import DEBUG_ROOT                                      # noqa: E402
from modules.ambient.diffusion.clap_encoder import ClapTextEncoder     # noqa: E402
from modules.ambient.diffusion.model import LatentDenoiser              # noqa: E402
from modules.ambient.diffusion.schedule import NoiseSchedule, ddim_sample  # noqa: E402


DEFAULT_CLAP_CKPT = (
    PROJECT_ROOT / "model" / "candidates" / "lucas" / "layer-a-ambient-diffusion-clap" / "best.pt"
)
DEFAULT_VAE_CKPT = PROJECT_ROOT / "model" / "candidates" / "lucas" / "vae-site257-30epoch" / "best.pt"
DEFAULT_VOC_CKPT = PROJECT_ROOT / "model" / "candidates" / "lucas" / "vocoder-hifigan-site257" / "best.pt"
DEFAULT_SAMPLE_DIR = DEBUG_ROOT / "layer_a" / "clap_diffusion" / "samples"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",   nargs="+", required=True,
                   help="One or more text prompts.")
    p.add_argument("--out",      nargs="+", required=False, default=None,
                   help="Output WAV path(s). Must match number of prompts if given.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_SAMPLE_DIR,
                   help="Directory for default generated WAV files.")
    p.add_argument("--checkpoint",     type=Path, default=DEFAULT_CLAP_CKPT)
    p.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CKPT)
    p.add_argument("--vocoder-checkpoint", type=Path, default=DEFAULT_VOC_CKPT)
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--steps",     type=int,   default=None)
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--device",    type=str,   default=None)
    p.add_argument("--dry-run",   action="store_true",
                   help="Print shapes and skip file write.")
    return p.parse_args()


def pick_device(arg: str | None) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_denoiser(ckpt_path: Path, device: torch.device) -> tuple[LatentDenoiser, dict]:
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg    = ckpt.get("config", {})
    model  = LatentDenoiser(
        latent_dim=cfg.get("latent_dim", 256),
        cond_dim  =cfg.get("cond_dim",   512),
        hidden_dim=cfg.get("hidden_dim", 512),
        num_blocks=cfg.get("num_blocks", 6),
    ).to(device)
    state = ckpt.get("ema") or ckpt.get("model")
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def load_vae_decoder(ckpt_path: Path, device: torch.device):
    from modules.ambient.model import SoundscapeModel         # noqa: E402
    from modules.ambient.dataset import N_ENV_FEATURES        # noqa: E402

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    from modules.ambient.preprocess import FRAMES_PER_CLIP, SPEC_CFG  # noqa: E402

    crop_secs   = saved_args.get("crop_seconds", 30.0)
    crop_frames = (
        int(crop_secs * SPEC_CFG["sample_rate"] / SPEC_CFG["hop_length"])
        if crop_secs > 0 else FRAMES_PER_CLIP
    )
    full_model = SoundscapeModel(
        env_dim    =saved_args.get("env_dim",    N_ENV_FEATURES),
        embed_dim  =saved_args.get("embed_dim",  512),
        latent_dim =saved_args.get("latent_dim", 256),
        target_frames=crop_frames,
    ).to(device)
    full_model.load_state_dict(ckpt["model"])
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad_(False)
    return full_model.decoder


def load_vocoder(ckpt_path: Path, device: torch.device):
    """Load the HiFi-GAN vocoder checkpoint."""
    from modules.ambient.train_vocoder import HiFiGANGenerator  # noqa: E402
    import yaml
    from modules.ambient.preprocess import SPEC_CFG              # noqa: E402

    params_path = PROJECT_ROOT / "params.yaml"
    voc_cfg     = yaml.safe_load(open(params_path))["vocoder"]

    generator = HiFiGANGenerator(
        n_mels        =voc_cfg.get("n_mels", SPEC_CFG["n_mels"]),
        base_channels =voc_cfg.get("base_channels", 128),
        upsample_rates=voc_cfg.get("upsample_rates", [8, 8, 4, 2]),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("generator") or ckpt.get("model") or ckpt
    generator.load_state_dict(state)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    return generator


def mel_to_wav(mel: torch.Tensor, vocoder, device: torch.device) -> np.ndarray:
    """mel: (B, 1, n_mels, T) or (B, n_mels, T) → waveform ndarray.

    The HiFiGAN vocoder was trained on `mel_norm = (power_to_db(mel, ref=np.max) + 80) / 80`,
    so every training-time mel had peak == 1.0 by construction (ref=np.max).

    The VAE decoder however outputs mels that peak around 0.7–0.9 — never reaching 1.0
    — putting them in a slightly out-of-distribution regime for the vocoder, which
    responds with low-energy noise (sounds like quiet machinery hum).

    Fix: apply per-clip min-max stretch to [0, 1] before vocoding. This replicates
    the vocoder's training-time normalisation exactly (peak=1, floor=0).
    """
    # Ensure (B, n_mels, T) — remove the channel dim if present
    if mel.dim() == 4:
        mel = mel.squeeze(1)                                    # (B, 128, T)

    # Per-clip min-max stretch to [0, 1] — match vocoder's training-time mel range.
    # Compute over (n_mels, T) per batch element so each clip is independently normalised.
    flat = mel.reshape(mel.shape[0], -1)
    mn = flat.amin(dim=-1, keepdim=True).unsqueeze(-1)          # (B, 1, 1)
    mx = flat.amax(dim=-1, keepdim=True).unsqueeze(-1)          # (B, 1, 1)
    mel = (mel - mn) / (mx - mn + 1e-9)
    mel = mel.clamp(0.0, 1.0)

    with torch.no_grad():
        wav = vocoder(mel.to(device))                           # (B, 1, samples)
    wav = wav.squeeze().cpu().numpy()                           # (samples,)
    return wav.astype(np.float32)


def render_mel(mel: np.ndarray, out_path: Path, title: str) -> None:
    from modules.ambient.preprocess import SPEC_CFG  # noqa: E402

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = mel.shape[-1]
    duration = frames * SPEC_CFG["hop_length"] / SPEC_CFG["sample_rate"]
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, SPEC_CFG["n_mels"]],
        cmap="magma",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_title(title)
    fig.colorbar(img, ax=ax, label="normalised mel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def audio_stats(wav: np.ndarray, sample_rate: int) -> dict:
    return {
        "sample_rate": sample_rate,
        "duration_s": float(len(wav) / sample_rate),
        "min": float(wav.min()),
        "max": float(wav.max()),
        "mean": float(wav.mean()),
        "rms": float(np.sqrt(np.mean(np.square(wav)))),
        "peak": float(np.max(np.abs(wav))),
    }


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")

    prompts = args.prompt

    # Validate output paths
    if not args.dry_run:
        if args.out is None:
            args.out = [str(args.out_dir / f"generated_{i:02d}.wav") for i in range(len(prompts))]
        if len(args.out) != len(prompts):
            print(f"[error] --out count ({len(args.out)}) != --prompt count ({len(prompts)})")
            return 1
        out_paths = [Path(o) for o in args.out]
    else:
        out_paths = [None] * len(prompts)

    # ---- CLAP encode ----
    print(f"Encoding {len(prompts)} prompt(s) with CLAP …")
    clap = ClapTextEncoder(device=device)
    text_embs = clap(prompts)                                   # (N, 512)
    print(f"  text_embs: {tuple(text_embs.shape)}  "
          f"norms: {torch.linalg.norm(text_embs, dim=-1).tolist()}")

    # ---- Load denoiser ----
    if not args.checkpoint.exists():
        print(f"[error] checkpoint not found: {args.checkpoint}\n"
              "Train first: python3 acoustic_ai/modules/ambient/diffusion/train_clap.py")
        return 1

    print(f"Loading denoiser: {args.checkpoint}")
    denoiser, cfg = load_denoiser(args.checkpoint, device)

    cfg_scale   = args.cfg_scale if args.cfg_scale is not None else cfg.get("cfg_scale", 3.0)
    num_steps   = args.steps     if args.steps     is not None else cfg.get("num_inference_steps", 50)

    schedule = NoiseSchedule(
        num_train_timesteps=cfg.get("num_train_timesteps", 1000),
        schedule           =cfg.get("schedule", "cosine"),
    ).to(device)

    # ---- DDIM sample ----
    print(f"DDIM sampling  steps={num_steps}  cfg_scale={cfg_scale} …")
    gen      = torch.Generator(device=device).manual_seed(args.seed)
    null_cond = torch.zeros_like(text_embs)

    latents = ddim_sample(
        denoiser,
        cond               =text_embs,
        schedule           =schedule,
        num_inference_steps=num_steps,
        cfg_scale          =cfg_scale,
        null_cond          =null_cond,
        generator          =gen,
    )                                                           # (N, 256)
    print(f"  latents: {tuple(latents.shape)}  "
          f"mean={latents.mean():.3f}  std={latents.std():.3f}")

    if args.dry_run:
        print("\n[dry-run] skipping VAE decode + vocoder — pipeline shapes OK")
        return 0

    # ---- De-normalise latents (inverse of SmokeTestDataset per-dim z-score) ----
    scale_path = (
        PROJECT_ROOT
        / "resources" / "site_257_bowra-dry-a"
        / "smoking_test_dataset" / "precomputed" / "latent_scale.json"
    )
    if scale_path.exists():
        import json
        scale     = json.loads(scale_path.read_text())
        lat_mean  = torch.tensor(scale["mean"], dtype=torch.float32, device=device)
        lat_std   = torch.tensor(scale["std"],  dtype=torch.float32, device=device)
        latents   = latents * lat_std + lat_mean
        print(f"  de-normalised latents: mean={latents.mean():.3f}  std={latents.std():.3f}")
    else:
        print(f"  [warn] latent_scale.json not found at {scale_path} — skipping de-normalisation")

    # ---- VAE decode ----
    print(f"Loading VAE decoder: {args.vae_checkpoint}")
    decoder = load_vae_decoder(args.vae_checkpoint, device)

    with torch.no_grad():
        mels = decoder(latents)                                 # (N, 1, 128, T)
    print(f"  mels: {tuple(mels.shape)}")

    # ---- Vocoder ----
    if not args.vocoder_checkpoint.exists():
        print(f"[warn] vocoder checkpoint not found: {args.vocoder_checkpoint}")
        print("Saving mel-only outputs …")
        for i, (prompt, out_p) in enumerate(zip(prompts, out_paths)):
            mel_np = mels[i].squeeze().cpu().numpy()
            np.save(out_p.with_suffix(".mel.npy"), mel_np)
            print(f"  [{i+1}] {out_p.with_suffix('.mel.npy')}")
        return 0

    print(f"Loading vocoder: {args.vocoder_checkpoint}")
    vocoder = load_vocoder(args.vocoder_checkpoint, device)

    # ---- Write outputs ----
    from modules.ambient.preprocess import SPEC_CFG  # noqa: E402
    sr = SPEC_CFG["sample_rate"]

    for i, (prompt, out_p) in enumerate(zip(prompts, out_paths)):
        out_p = Path(out_p)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        wav = mel_to_wav(mels[i : i + 1], vocoder, device)     # (samples,)
        mel_np = mels[i].squeeze().detach().cpu().numpy().astype(np.float32)

        # Light peak normalisation
        peak = float(np.abs(wav).max() + 1e-9)
        if peak > 1.0:
            wav /= peak

        sf.write(str(out_p), wav, sr)
        mel_npy = out_p.with_suffix(".mel.npy")
        mel_png = out_p.with_name(f"{out_p.stem}_mel.png")
        meta_p = out_p.with_name(f"{out_p.stem}_metadata.json")
        np.save(mel_npy, mel_np)
        render_mel(mel_np, mel_png, f"CLAP diffusion mel - sample {i:02d}")
        meta_p.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "seed": args.seed,
                    "cfg_scale": cfg_scale,
                    "steps": num_steps,
                    "checkpoint": str(args.checkpoint),
                    "vae_checkpoint": str(args.vae_checkpoint),
                    "vocoder_checkpoint": str(args.vocoder_checkpoint),
                    "audio": audio_stats(wav, sr),
                    "mel_shape": list(mel_np.shape),
                    "artifacts": {
                        "wav": str(out_p),
                        "mel_npy": str(mel_npy),
                        "mel_png": str(mel_png),
                    },
                },
                indent=2,
            )
        )
        dur = len(wav) / sr
        print(f"  [{i+1}/{len(prompts)}] {out_p}  ({dur:.1f}s @ {sr} Hz)")
        print(f"        mel: {mel_png}")
        print(f"        prompt: {prompt[:80]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
