"""Latent diffusion training loop for Layer A (Method 3).

Usage (from project root):
  python3 acoustic_ai/modules/ambient/diffusion/train.py
  python3 acoustic_ai/modules/ambient/diffusion/train.py --epochs 50

Deps come from `params.yaml` `diffusion:` section. Latents and index must
exist under `acoustic_ai/data/ambient/latents/` — run
`precompute/precompute_ambient_latents.py` first.

Training objective:
  loss = MSE(v_pred, v_target)
where v_target = alpha_t · noise - sigma_t · x_0  (v-prediction).

Classifier-free guidance support:
  with prob `cond_dropout_p`, replace cond with zeros so the model also
  learns the unconditional score. Sampler combines them at inference time.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# allow `python3 acoustic_ai/modules/ambient/diffusion/train.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from modules.ambient.diffusion.dataset import AmbientLatentDataset  # noqa: E402
from modules.ambient.diffusion.model import LatentDenoiser           # noqa: E402
from modules.ambient.diffusion.schedule import (                     # noqa: E402
    NoiseSchedule, add_noise, v_target,
)


DEFAULT_LATENTS = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "latents" / "ambient_latents.npy"
DEFAULT_INDEX = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "latents" / "ambient_latents_index.csv"
DEFAULT_OUT = PROJECT_ROOT / "acoustic_ai" / "checkpoints" / "ambient_diffusion"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--latents", type=Path, default=DEFAULT_LATENTS)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--params", type=Path, default=PROJECT_ROOT / "params.yaml")
    p.add_argument("--epochs", type=int, default=None, help="Override params.yaml epochs.")
    p.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from.")
    p.add_argument("--device", type=str, default=None,
                   help="cpu / cuda / mps. Defaults to cuda > mps > cpu.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_device(arg: str | None) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for eb, b in zip(ema_model.buffers(), model.buffers()):
        eb.data.copy_(b.data)


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.params))["diffusion"]
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    if not args.latents.exists() or not args.index.exists():
        print(
            f"missing precomputed latents:\n  {args.latents}\n  {args.index}\n"
            "run precompute/precompute_ambient_latents.py first.",
            file=sys.stderr,
        )
        return 1

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    # ---- data ----
    dataset = AmbientLatentDataset(args.latents, args.index)
    print(f"dataset: {len(dataset)} latents, dim={dataset.latents.shape[1]}, "
          f"cond_dim={dataset.cond.shape[1]}")

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    # ---- model + schedule ----
    model = LatentDenoiser(
        latent_dim=cfg["latent_dim"],
        cond_dim=cfg["cond_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"denoiser params: {n_params/1e6:.2f}M")

    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    schedule = NoiseSchedule(
        num_train_timesteps=cfg["num_train_timesteps"],
        schedule=cfg["schedule"],
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    start_epoch = 1
    best_loss = float("inf")

    if args.resume and args.resume.exists():
        print(f"resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        if "optimiser" in ckpt:
            optimiser.load_state_dict(ckpt["optimiser"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("loss", float("inf"))
        print(f"  starting from epoch {start_epoch}, best_loss={best_loss:.5f}")

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- training loop ----
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        t0 = time.time()

        for z0, cond in loader:
            z0 = z0.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            B = z0.shape[0]

            # Sample timesteps and noise
            t = torch.randint(0, schedule.T, (B,), device=device)
            noise = torch.randn_like(z0)
            alpha, sigma = schedule.alpha_sigma(t)

            z_t = add_noise(z0, noise, alpha, sigma)
            v_tgt = v_target(z0, noise, alpha, sigma)

            # Classifier-free guidance dropout
            if cfg["cond_dropout_p"] > 0:
                drop_mask = (torch.rand(B, device=device) < cfg["cond_dropout_p"]).float().unsqueeze(-1)
                cond_in = cond * (1.0 - drop_mask)
            else:
                cond_in = cond

            v_pred = model(z_t, t, cond_in)
            loss = F.mse_loss(v_pred, v_tgt)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            update_ema(ema_model, model, cfg["ema_decay"])

            epoch_loss += loss.item() * B
            epoch_n += B

        avg_loss = epoch_loss / max(epoch_n, 1)
        dt = time.time() - t0
        print(f"epoch {epoch:4d}/{cfg['epochs']}  loss={avg_loss:.5f}  ({dt:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "loss": avg_loss,
                },
                args.out / "best.pt",
            )

    # always save a final checkpoint too
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "config": cfg,
            "epoch": cfg["epochs"],
            "loss": avg_loss,
        },
        args.out / "last.pt",
    )
    print(f"done. best loss {best_loss:.5f}. checkpoints in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
