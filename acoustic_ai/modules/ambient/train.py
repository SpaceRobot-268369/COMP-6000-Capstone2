"""Training loop for the SoundscapeModel (VAE).

Usage (from project root):
  python3 acoustic_ai/modules/ambient/train.py
  python3 acoustic_ai/modules/ambient/train.py --epochs 30 --batch-size 8 --lr 1e-4
  python3 acoustic_ai/modules/ambient/train.py --resume acoustic_ai/checkpoints/ambient/epoch_05.pt

VAE loss:
  loss = MSE(recon, input) + beta_kl * KL(q(z|x) || N(0,I))

  beta_kl=0.001 keeps the KL term small at the start so the model first
  learns to reconstruct well, then the latent space gradually regularises.
  Increase beta_kl (e.g. 0.01) for a more structured latent space at the
  cost of slightly higher reconstruction loss.

Checkpoints are saved to acoustic_ai/checkpoints/ambient/ after every epoch.
Best checkpoint (lowest val loss) is also saved as best.pt.
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SoundscapeDataset, N_ENV_FEATURES
from .model import SoundscapeModel
from .preprocess import SPEC_CFG

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_PATH  = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "site_257_training_manifest.csv"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "ambient"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SoundscapeModel (VAE).")
    p.add_argument("--manifest",    type=Path, default=MANIFEST_PATH)
    p.add_argument("--epochs",      type=int,  default=30)
    p.add_argument("--batch-size",  type=int,  default=16)
    p.add_argument("--lr",          type=float,default=1e-4)
    p.add_argument("--beta-kl",     type=float,default=0.001,
                   help="Weight on KL divergence loss. Start small (0.001), "
                        "increase to 0.01 for a more structured latent space.")
    p.add_argument("--val-frac",    type=float,default=0.15)
    p.add_argument("--num-workers", type=int,  default=0,
                   help="DataLoader workers. Default 0 avoids MPS/multiprocessing issues on Mac.")
    p.add_argument("--crop-seconds", type=float, default=30.0,
                   help="Random time crop per training step (seconds). 0 = no crop.")
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--resume",      type=Path, default=None,
                   help="Path to checkpoint .pt to resume from.")
    p.add_argument("--embed-dim",   type=int,  default=512)
    p.add_argument("--latent-dim",  type=int,  default=256)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# VAE loss
# ---------------------------------------------------------------------------
def vae_loss(
    recon:   torch.Tensor,
    target:  torch.Tensor,
    mu:      torch.Tensor,
    log_var: torch.Tensor,
    beta:    float,
) -> tuple[torch.Tensor, float, float]:
    """Compute VAE loss = MSE + β·KL.

    Returns (total_loss, recon_loss_item, kl_loss_item).
    KL = -0.5 * mean(1 + log_var - mu² - exp(log_var))
    """
    recon_loss = F.mse_loss(recon, target)
    kl_loss    = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
    total      = recon_loss + beta * kl_loss
    return total, recon_loss.item(), kl_loss.item()


# ---------------------------------------------------------------------------
# Train / val one epoch
# ---------------------------------------------------------------------------
def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
    beta_kl:   float,
    train:     bool,
) -> tuple[float, float, float]:
    """Returns (total_loss, recon_loss, kl_loss) averaged over the dataset."""
    model.train(train)
    total_total = 0.0
    total_recon = 0.0
    total_kl    = 0.0

    with torch.set_grad_enabled(train):
        for mel, env, _ in tqdm(loader, leave=False, desc="train" if train else "val "):
            mel = mel.to(device)
            env = env.to(device)

            recon, mu, log_var = model(mel, env)
            loss, recon_item, kl_item = vae_loss(recon, mel, mu, log_var, beta_kl)

            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            n = mel.size(0)
            total_total += loss.item() * n
            total_recon += recon_item  * n
            total_kl    += kl_item     * n

    N = len(loader.dataset)
    return total_total / N, total_recon / N, total_kl / N


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model: nn.Module, optimiser, scheduler) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimiser.load_state_dict(ckpt["optimiser"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  Resumed from {path}  (epoch {ckpt['epoch']}, best_val={ckpt['best_val']:.6f})")
    return ckpt["epoch"], ckpt["best_val"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args   = parse_args()
    device = get_device()
    print(f"Device    : {device}")
    print(f"Manifest  : {args.manifest}")
    print(f"Beta KL   : {args.beta_kl}")

    # ── Crop frames ───────────────────────────────────────────────────────────
    crop_frames = (
        int(args.crop_seconds * SPEC_CFG["sample_rate"] / SPEC_CFG["hop_length"])
        if args.crop_seconds > 0 else None
    )
    if crop_frames:
        print(f"Crop frames: {crop_frames}  ({args.crop_seconds:.0f}s per training step)")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = SoundscapeDataset(
        manifest_path=str(args.manifest),
        project_root=str(PROJECT_ROOT),
        split="train",
        val_fraction=args.val_frac,
        seed=args.seed,
        crop_frames=crop_frames,
    )
    val_ds = SoundscapeDataset(
        manifest_path=str(args.manifest),
        project_root=str(PROJECT_ROOT),
        split="val",
        val_fraction=args.val_frac,
        seed=args.seed,
        stats=train_ds.get_stats(),
        crop_frames=crop_frames,
    )
    print(f"Train clips: {len(train_ds)}  |  Val clips: {len(val_ds)}")
    print(f"Env dim    : {train_ds.env_dim}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    from preprocess import FRAMES_PER_CLIP
    model = SoundscapeModel(
        env_dim=N_ENV_FEATURES,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        target_frames=crop_frames if crop_frames else FRAMES_PER_CLIP,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    start_epoch = 1
    best_val    = math.inf

    if args.resume:
        start_epoch, best_val = load_checkpoint(args.resume, model, optimiser, scheduler)
        start_epoch += 1

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs (starting at {start_epoch})\n")
    print(f"{'Epoch':>6}  {'train':>10}  {'val':>10}  {'recon':>10}  {'kl':>10}  {'lr':>10}  {'time':>6}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_recon, train_kl = run_epoch(
            model, train_loader, optimiser, device, args.beta_kl, train=True
        )
        val_loss, val_recon, val_kl = run_epoch(
            model, val_loader, optimiser, device, args.beta_kl, train=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(
            f"{epoch:03d}/{args.epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"recon={val_recon:.6f}  kl={val_kl:.4f}  "
            f"lr={lr_now:.2e}  {elapsed:.0f}s"
        )

        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val":  best_val,
            "args":      vars(args),
        }

        save_checkpoint(state, CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(state, CHECKPOINT_DIR / "best.pt")
            print(f"  ↳ New best val loss: {best_val:.6f}")

    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
