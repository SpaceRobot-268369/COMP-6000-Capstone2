"""CLAP-conditioned latent diffusion training — smoke-test run.

Trains a LatentDenoiser on the 50 smoke-test clips.  The model receives
CLAP text embeddings (512-dim) as conditioning instead of the env one-hot
vector used by the original env-conditioned model.

The goal of this run is end-to-end pipeline verification (loss decreases,
model reconstructs training samples reasonably), NOT production quality.
Overfitting on 50 clips is expected and intentional.

Usage (from project root):
    python3 acoustic_ai/modules/ambient/diffusion/train_clap.py
    python3 acoustic_ai/modules/ambient/diffusion/train_clap.py --epochs 200 --device mps

Deps:
    Run first:
        python3 acoustic_ai/precompute/precompute_smoke_latents.py
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
from torch.utils.data import DataLoader, RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from modules.ambient.diffusion.smoke_dataset import SmokeTestDataset   # noqa: E402
from modules.ambient.diffusion.model import LatentDenoiser              # noqa: E402
from modules.ambient.diffusion.schedule import (                        # noqa: E402
    NoiseSchedule, add_noise, v_target,
)

PRECOMPUTED_DIR = (
    PROJECT_ROOT
    / "resources" / "site_257_bowra-dry-a"
    / "smoking_test_dataset" / "precomputed"
)
DEFAULT_OUT = PROJECT_ROOT / "acoustic_ai" / "checkpoints" / "ambient_diffusion_clap"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--precomputed", type=Path, default=PRECOMPUTED_DIR)
    p.add_argument("--out",         type=Path, default=DEFAULT_OUT)
    p.add_argument("--params",      type=Path, default=PARAMS_PATH)
    p.add_argument("--epochs",      type=int,  default=None, help="Override params epochs.")
    p.add_argument("--device",      type=str,  default=None)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--log-every",   type=int,  default=10)
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
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ep, p in zip(ema.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for eb, b in zip(ema.buffers(), model.buffers()):
        eb.data.copy_(b.data)


def main() -> int:
    args = parse_args()
    raw_cfg = yaml.safe_load(open(args.params))
    cfg = raw_cfg.get("diffusion_clap", raw_cfg.get("diffusion", {}))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    # ---- dataset ----
    dataset = SmokeTestDataset(args.precomputed)
    print(
        f"dataset: {len(dataset)} samples  "
        f"latent_dim={dataset.latent_dim}  cond_dim={dataset.cond_dim}"
    )

    # With only 50 samples, repeat within each epoch so we get reasonable
    # gradient statistics even at small batch sizes.
    samples_per_epoch = max(len(dataset), cfg.get("batch_size", 16) * 16)
    sampler = RandomSampler(dataset, replacement=True, num_samples=samples_per_epoch)
    loader  = DataLoader(
        dataset,
        batch_size =cfg.get("batch_size", 16),
        sampler    =sampler,
        num_workers=0,
        drop_last  =False,
        pin_memory =(device.type == "cuda"),
    )

    # ---- model ----
    model = LatentDenoiser(
        latent_dim =cfg.get("latent_dim",  dataset.latent_dim),
        cond_dim   =cfg.get("cond_dim",    dataset.cond_dim),
        hidden_dim =cfg.get("hidden_dim",  512),
        num_blocks =cfg.get("num_blocks",  6),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"LatentDenoiser  params: {n_params/1e6:.2f}M  "
          f"(cond_dim={model.cond_dim}  hidden={cfg.get('hidden_dim', 512)})")

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    # ---- schedule ----
    schedule = NoiseSchedule(
        num_train_timesteps=cfg.get("num_train_timesteps", 1000),
        schedule           =cfg.get("schedule", "cosine"),
    ).to(device)

    # ---- optimiser ----
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr          =cfg.get("lr", 2e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # Cosine LR warm-up / decay
    total_steps   = cfg.get("epochs", 500) * len(loader)
    warmup_steps  = min(200, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr      =cfg.get("lr", 2e-4),
        total_steps =total_steps,
        pct_start   =warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- training loop ----
    best_loss    = float("inf")
    cond_drop_p  = cfg.get("cond_dropout_p", 0.1)
    ema_decay    = cfg.get("ema_decay", 0.999)
    epochs       = cfg.get("epochs", 500)

    print(f"\nTraining for {epochs} epochs  "
          f"batch={cfg.get('batch_size', 16)}  "
          f"cond_drop={cond_drop_p}  "
          f"lr={cfg.get('lr', 2e-4)}\n"
          f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_n    = 0
        t0         = time.time()

        for z0, cond in loader:
            z0   = z0.to(device,   non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            B    = z0.shape[0]

            # Sample timesteps + noise
            t     = torch.randint(0, schedule.T, (B,), device=device)
            noise = torch.randn_like(z0)
            alpha, sigma = schedule.alpha_sigma(t)

            z_t   = add_noise(z0, noise, alpha, sigma)
            v_tgt = v_target(z0, noise, alpha, sigma)

            # Classifier-free guidance dropout
            if cond_drop_p > 0:
                drop  = (torch.rand(B, device=device) < cond_drop_p).float().unsqueeze(-1)
                cond_in = cond * (1.0 - drop)
            else:
                cond_in = cond

            v_pred = model(z_t, t, cond_in)
            loss   = F.mse_loss(v_pred, v_tgt)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()
            update_ema(ema, model, ema_decay)

            epoch_loss += loss.item() * B
            epoch_n    += B

        avg_loss = epoch_loss / max(epoch_n, 1)
        dt       = time.time() - t0

        if epoch % args.log_every == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"epoch {epoch:5d}/{epochs}  loss={avg_loss:.5f}  "
                  f"lr={lr_now:.2e}  ({dt:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "model":     model.state_dict(),
                "ema":       ema.state_dict(),
                "config":    cfg | {"latent_dim": model.latent_dim,
                                    "cond_dim":   model.cond_dim},
                "epoch":     epoch,
                "loss":      avg_loss,
                "cond_type": "clap",
            }
            torch.save(ckpt, args.out / "best.pt")

    # Final checkpoint
    torch.save(
        {
            "model":     model.state_dict(),
            "ema":       ema.state_dict(),
            "config":    cfg | {"latent_dim": model.latent_dim, "cond_dim": model.cond_dim},
            "epoch":     epochs,
            "loss":      avg_loss,
            "cond_type": "clap",
        },
        args.out / "last.pt",
    )

    print(f"\n{'─'*60}")
    print(f"Done.  best loss: {best_loss:.5f}  checkpoints: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
