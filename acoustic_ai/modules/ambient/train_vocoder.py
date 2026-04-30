"""Train an ecoacoustic HiFi-GAN vocoder.

Trains a HiFi-GAN-style generator natively on site 257 ecoacoustic audio:
  - Input : 128-bin mel-spectrogram at 22 050 Hz  (matches our model exactly)
  - Output: waveform at 22 050 Hz
  - No mel bin interpolation, no 16→22 kHz resampling

This replaces the speech-trained SpeechT5 HiFi-GAN used in Stage 2.

Loss:
  - L1 mel reconstruction loss  (keeps spectral shape)
  - Multi-resolution STFT loss  (encourages phase coherence at multiple scales)
  No discriminator — keeps training simple and stable; results are smoother than
  GAN output but far better than the speech-domain vocoder on ecoacoustic audio.

Usage (from project root):
  python3 acoustic_ai/modules/ambient/train_vocoder.py
  python3 acoustic_ai/modules/ambient/train_vocoder.py --epochs 100 --max-clips 500

Output:
  acoustic_ai/checkpoints/vocoder/best.pt   ← used by inference.py

Estimated training time (Mac M-series):
  100 epochs × 500 clips × 30 s crops ≈ 2–3 hours
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT     = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_PATH    = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "site_257_training_manifest.csv"
VOCODER_CKPT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "vocoder"


# ---------------------------------------------------------------------------
# Spectrogram config (must match preprocess.py exactly)
# ---------------------------------------------------------------------------
SR         = 22_050
N_FFT      = 1024
HOP_LENGTH = 512
N_MELS     = 128
FMIN       = 50
FMAX       = 11_000
TOP_DB     = 80

# Upsample ratios — product must equal HOP_LENGTH (512)
# 8 × 8 × 4 × 2 = 512  ✓
UPSAMPLE_RATES    = [8, 8, 4, 2]
UPSAMPLE_KERNELS  = [16, 16, 8, 4]   # must be 2× upsample rate

# Residual block kernel sizes and dilations (HiFi-GAN V1 defaults)
RESBLOCK_KERNELS   = [3, 7, 11]
RESBLOCK_DILATIONS = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


# ---------------------------------------------------------------------------
# Generator (HiFi-GAN-style)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with multiple dilated convolution stacks."""

    def __init__(self, channels: int, kernel_size: int, dilations: list):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size - 1) * d // 2
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=pad),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=1,
                          padding=(kernel_size - 1) // 2),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN generator for 128-bin mel at 22 050 Hz.

    Architecture mirrors HiFi-GAN V1 but with:
      - n_mels=128 input channels (not 80)
      - upsample_rates=[8,8,4,2] to match hop_length=512
      - output at 22 050 Hz (no resampling needed)
    """

    def __init__(
        self,
        n_mels:            int  = N_MELS,
        base_channels:     int  = 128,
        upsample_rates:    list = None,
        upsample_kernels:  list = None,
        resblock_kernels:  list = None,
        resblock_dils:     list = None,
    ):
        super().__init__()
        upsample_rates   = upsample_rates   or UPSAMPLE_RATES
        upsample_kernels = upsample_kernels or UPSAMPLE_KERNELS
        resblock_kernels = resblock_kernels or RESBLOCK_KERNELS
        resblock_dils    = resblock_dils    or RESBLOCK_DILATIONS

        self.conv_pre = nn.Conv1d(n_mels, base_channels, kernel_size=7, padding=3)

        ch = base_channels
        self.ups   = nn.ModuleList()
        self.resbs = nn.ModuleList()
        for rate, ksize in zip(upsample_rates, upsample_kernels):
            self.ups.append(nn.ConvTranspose1d(
                ch, ch // 2,
                kernel_size=ksize, stride=rate,
                padding=(ksize - rate) // 2,
            ))
            ch //= 2
            for k, d in zip(resblock_kernels, resblock_dils):
                self.resbs.append(ResBlock(ch, k, d))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(ch, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

        self.n_resblocks_per_up = len(resblock_kernels)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T) → waveform: (B, 1, T * hop_length)"""
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = None
            for j in range(self.n_resblocks_per_up):
                rb = self.resbs[i * self.n_resblocks_per_up + j]
                xs = rb(x) if xs is None else xs + rb(x)
            x = xs / self.n_resblocks_per_up
        return self.conv_post(x)


# ---------------------------------------------------------------------------
# Multi-resolution STFT loss
# ---------------------------------------------------------------------------

def stft(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """Return magnitude STFT of (B, T) waveform."""
    window = torch.hann_window(win, device=x.device)
    S = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                   window=window, return_complex=True)
    return S.abs()


def stft_loss(pred: torch.Tensor, target: torch.Tensor,
              n_fft: int, hop: int, win: int) -> torch.Tensor:
    """Spectral convergence + log STFT magnitude loss at one resolution."""
    P = stft(pred,   n_fft, hop, win)
    T = stft(target, n_fft, hop, win)
    sc  = torch.norm(T - P, "fro") / (torch.norm(T, "fro") + 1e-8)
    mag = F.l1_loss(torch.log(P + 1e-7), torch.log(T + 1e-7))
    return sc + mag


def multi_resolution_stft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum of STFT losses at three resolutions."""
    configs = [
        (1024, 120, 600),
        (2048, 240, 1200),
        (512,   50, 240),
    ]
    loss = torch.tensor(0.0, device=pred.device)
    for n_fft, hop, win in configs:
        loss = loss + stft_loss(pred, target, n_fft, hop, win)
    return loss / len(configs)


def mel_loss(pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
    """L1 loss in mel-spectrogram domain."""
    import librosa
    def to_mel(w):
        mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                                        fmin=FMIN, fmax=FMAX)
        mel_t = torch.FloatTensor(mel_basis).to(w.device)
        window = torch.hann_window(N_FFT, device=w.device)
        S = torch.stft(w, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
                       window=window, return_complex=True).abs().pow(2)
        mel = torch.matmul(mel_t, S)
        return torch.log(mel.clamp(min=1e-9))

    P = to_mel(pred_wav)
    T = to_mel(target_wav)
    return F.l1_loss(P, T)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VocoderDataset(Dataset):
    """Loads real wav files and returns (mel, waveform) pairs.

    mel      : (n_mels, crop_frames) float32 tensor  [0, 1] normalised
    waveform : (crop_samples,)        float32 tensor  [-1, 1]
    """

    def __init__(
        self,
        manifest_path: str,
        project_root:  str,
        max_clips:     int   = 500,
        crop_seconds:  float = 1.0,
        split:         str   = "train",
        val_fraction:  float = 0.1,
        seed:          int   = 42,
    ):
        import pandas as pd

        self.project_root = Path(project_root)
        self.crop_samples = int(crop_seconds * SR)
        self.crop_frames  = self.crop_samples // HOP_LENGTH

        df = pd.read_csv(manifest_path)

        # Use only clips that have a .wav file
        rng = random.Random(seed)
        all_paths = []
        for _, row in df.iterrows():
            wav_path = self.project_root / (str(row["clip_path"]) + ".wav")
            if wav_path.exists():
                all_paths.append(str(wav_path))

        # Shuffle and cap
        rng.shuffle(all_paths)
        all_paths = all_paths[:max_clips]

        # Train / val split
        n_val = max(1, int(len(all_paths) * val_fraction))
        if split == "val":
            self.paths = all_paths[:n_val]
        else:
            self.paths = all_paths[n_val:]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        import librosa

        path = self.paths[idx]
        try:
            wav, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            wav = np.zeros(self.crop_samples, dtype=np.float32)

        # Ensure long enough
        if len(wav) < self.crop_samples + 1:
            repeats = (self.crop_samples // len(wav)) + 2
            wav = np.tile(wav, repeats)

        # Random crop
        max_start = len(wav) - self.crop_samples
        start = random.randint(0, max_start)
        wav_crop = wav[start : start + self.crop_samples].astype(np.float32)

        # Compute mel for this crop
        mel = librosa.feature.melspectrogram(
            y=wav_crop, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
        # Normalise dB [-80, 0] → [0, 1]  (matches training manifest)
        mel_norm = (log_mel + TOP_DB) / TOP_DB
        mel_norm = mel_norm[:, :self.crop_frames].astype(np.float32)

        # Align waveform length to mel frames
        wav_crop = wav_crop[: self.crop_frames * HOP_LENGTH]

        return (
            torch.from_numpy(mel_norm),           # (n_mels, crop_frames)
            torch.from_numpy(wav_crop),            # (crop_samples,)
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    p = argparse.ArgumentParser(description="Train ecoacoustic HiFi-GAN vocoder.")
    p.add_argument("--manifest",      type=Path,  default=MANIFEST_PATH)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--max-clips",     type=int,   default=500,
                   help="Max training wav files to use (default 500).")
    p.add_argument("--crop-seconds",  type=float, default=1.0,
                   help="Waveform crop length per sample (seconds).")
    p.add_argument("--base-channels", type=int,   default=128)
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--resume",        type=Path,  default=None)
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device      : {device}")
    print(f"Max clips   : {args.max_clips}")
    print(f"Crop seconds: {args.crop_seconds}")

    train_ds = VocoderDataset(
        manifest_path=str(args.manifest),
        project_root=str(PROJECT_ROOT),
        max_clips=args.max_clips,
        crop_seconds=args.crop_seconds,
        split="train",
        seed=args.seed,
    )
    val_ds = VocoderDataset(
        manifest_path=str(args.manifest),
        project_root=str(PROJECT_ROOT),
        max_clips=args.max_clips,
        crop_seconds=args.crop_seconds,
        split="val",
        seed=args.seed,
    )
    print(f"Train clips : {len(train_ds)}  |  Val clips: {len(val_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No .wav files found. Run the download + ffmpeg conversion first.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = HiFiGANGenerator(base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Generator   : {n_params:,} parameters")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.999)

    start_epoch = 1
    best_val    = math.inf

    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["best_val"]
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    print(f"\nTraining for {args.epochs} epochs\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for mel, wav in train_loader:
            mel = mel.to(device)   # (B, n_mels, T_frames)
            wav = wav.to(device)   # (B, T_samples)

            pred = model(mel).squeeze(1)          # (B, T_samples)

            # Trim to same length (upsample may add/remove 1 sample)
            min_len = min(pred.shape[-1], wav.shape[-1])
            pred = pred[..., :min_len]
            wav  = wav[...,  :min_len]

            loss_stft = multi_resolution_stft_loss(pred, wav)
            loss_mel  = mel_loss(pred, wav)
            loss      = loss_stft + 45.0 * loss_mel   # weight mel higher

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mel, wav in val_loader:
                mel  = mel.to(device)
                wav  = wav.to(device)
                pred = model(mel).squeeze(1)
                min_len = min(pred.shape[-1], wav.shape[-1])
                pred = pred[..., :min_len]
                wav  = wav[...,  :min_len]
                loss_stft = multi_resolution_stft_loss(pred, wav)
                loss_mel  = mel_loss(pred, wav)
                val_loss += (loss_stft + 45.0 * loss_mel).item()
        val_loss /= max(len(val_loader), 1)

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
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
        save_checkpoint(state, VOCODER_CKPT_DIR / f"epoch_{epoch:03d}.pt")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(state, VOCODER_CKPT_DIR / "best.pt")
            print(f"  ↳ New best val loss: {best_val:.4f}")

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoints → {VOCODER_CKPT_DIR}")


if __name__ == "__main__":
    main()
