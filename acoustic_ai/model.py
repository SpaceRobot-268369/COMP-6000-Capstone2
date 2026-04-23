"""Soundscape model — CNN encoder + environmental conditioning.

Architecture (Pilot stage):
  ┌─────────────────────────────────────┐
  │  Input mel-spectrogram              │
  │  (1, 128, T)                        │
  └──────────────┬──────────────────────┘
                 │
         ┌───────▼────────┐
         │  CNN Encoder   │  4-block conv encoder
         │  → (512,)      │  audio embedding
         └───────┬────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │  concat([audio_emb, env_features])  │
  │  (512 + 29 = 541,)                  │
  └──────────────┬──────────────────────┘
                 │
         ┌───────▼────────┐
         │  Fusion MLP    │  projects to latent space
         │  → (256,)      │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Decoder       │  reconstructs mel-spectrogram
         │  → (1, 128, T) │
         └────────────────┘

The encoder-decoder structure lets the model learn a compact latent
representation of the soundscape conditioned on environmental variables.
Loss: MSE reconstruction on mel-spectrogram (sufficient for pilot feasibility).

Swap the decoder for a diffusion or GAN head in Stage 3.
"""

import torch
import torch.nn as nn
from preprocess import SPEC_CFG, FRAMES_PER_CLIP


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → GELU → optional MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransposeBlock(nn.Module):
    """ConvTranspose2d → BatchNorm → GELU for decoder upsampling."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """4-block CNN that maps (1, n_mels, T) → (embed_dim,)."""

    def __init__(self, n_mels: int = SPEC_CFG["n_mels"], embed_dim: int = 512):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(1,   32),   # → (32,  n_mels/2,  T/2)
            ConvBlock(32,  64),   # → (64,  n_mels/4,  T/4)
            ConvBlock(64,  128),  # → (128, n_mels/8,  T/8)
            ConvBlock(128, 256),  # → (256, n_mels/16, T/16)
        )
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.project = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)          # (B, 256, H', W')
        x = self.pool(x)            # (B, 256, 1, 1)
        x = x.flatten(1)            # (B, 256)
        return self.project(x)      # (B, embed_dim)


# ---------------------------------------------------------------------------
# Fusion MLP
# ---------------------------------------------------------------------------

class FusionMLP(nn.Module):
    """Fuses audio embedding + env vector into a latent representation."""

    def __init__(self, audio_dim: int, env_dim: int, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim + env_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, latent_dim),
            nn.GELU(),
        )

    def forward(self, audio_emb: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        x = torch.cat([audio_emb, env], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class MelDecoder(nn.Module):
    """Latent vector → (1, n_mels, T) mel-spectrogram.

    Reconstructs the same spatial size as the encoder input via transposed
    convolutions. The spatial seed (4 × 4) is chosen so that after 4
    upsampling steps (×2 each = ×16) the output matches n_mels=128 in the
    frequency axis. Time axis is then cropped/padded to target_frames.
    """

    def __init__(
        self,
        latent_dim:    int = 256,
        n_mels:        int = SPEC_CFG["n_mels"],
        target_frames: int = FRAMES_PER_CLIP,
    ):
        super().__init__()
        self.n_mels        = n_mels
        self.target_frames = target_frames

        # Project latent → spatial seed (256 channels, 4×4)
        self.project = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decode  = nn.Sequential(
            TransposeBlock(256, 128),  # → (128, 8,  8)
            TransposeBlock(128,  64),  # → (64,  16, 16)
            TransposeBlock(64,   32),  # → (32,  32, 32)
            TransposeBlock(32,   16),  # → (16,  64, 64)
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            # → (1, 128, 128)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x = self.project(z).view(B, 256, 4, 4)
        x = self.decode(x)                                  # (B, 1, 128, 128)
        # Resize time axis to target_frames via interpolation
        x = nn.functional.interpolate(
            x, size=(self.n_mels, self.target_frames), mode="bilinear", align_corners=False
        )
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SoundscapeModel(nn.Module):
    """End-to-end: (mel, env) → reconstructed mel."""

    def __init__(
        self,
        env_dim:    int = 29,
        embed_dim:  int = 512,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.encoder = AudioEncoder(embed_dim=embed_dim)
        self.fusion  = FusionMLP(audio_dim=embed_dim, env_dim=env_dim, latent_dim=latent_dim)
        self.decoder = MelDecoder(latent_dim=latent_dim)

    def encode(self, mel: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """Returns the latent vector z."""
        audio_emb = self.encoder(mel)
        return self.fusion(audio_emb, env)

    def forward(self, mel: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed mel-spectrogram, same shape as input."""
        z = self.encode(mel, env)
        return self.decoder(z)
