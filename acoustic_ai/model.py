"""Soundscape model — CNN encoder + environmental conditioning.

Architecture (Stage 3 — VAE):
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
         │  Fusion MLP    │  projects to μ and log σ²
         │  → mu (256,)   │
         │  → log_var(256)│
         └───────┬────────┘
                 │  reparameterise: z = μ + ε·σ
         ┌───────▼────────┐
         │  Decoder       │  reconstructs mel-spectrogram
         │  → (1, 128, T) │
         └────────────────┘

VAE training objective:
  loss = MSE(recon, input) + β × KL(q(z|x) ‖ N(0,I))

The KL term regularises the latent space so that random sampling from
N(0,1) at generation time produces coherent soundscapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Fusion MLP  (VAE — outputs mu and log_var)
# ---------------------------------------------------------------------------

class FusionMLP(nn.Module):
    """Fuses audio embedding + env vector into VAE distribution parameters.

    Returns (mu, log_var) — both shape (B, latent_dim).
    The shared trunk maps the concatenated input to a hidden representation;
    two separate linear heads produce mu and log_var independently.
    """

    def __init__(self, audio_dim: int, env_dim: int, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(audio_dim + env_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        # Distribution heads
        self.fc_mu      = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

    def forward(self, audio_emb: torch.Tensor, env: torch.Tensor):
        """Returns (mu, log_var) — each (B, latent_dim)."""
        x       = torch.cat([audio_emb, env], dim=-1)
        hidden  = self.trunk(x)
        mu      = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        return mu, log_var


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

    # Frequency seed height: 4 rows → ×5 upsamples (×2 each) → 128 mel bins
    # Time seed width:      64 cols → ×5 upsamples            → 2048 frames
    #   then crop to target_frames — no bilinear stretch needed.
    _FREQ_SEED = 4
    _TIME_SEED = 64

    def __init__(
        self,
        latent_dim:    int = 256,
        n_mels:        int = SPEC_CFG["n_mels"],
        target_frames: int = FRAMES_PER_CLIP,
    ):
        super().__init__()
        self.n_mels        = n_mels
        self.target_frames = target_frames

        # Project latent → spatial seed (256 channels, 4×64)
        self.project = nn.Linear(latent_dim, 256 * self._FREQ_SEED * self._TIME_SEED)
        self.decode  = nn.Sequential(
            TransposeBlock(256, 128),  # → (128, 8,   128)
            TransposeBlock(128,  64),  # → (64,  16,  256)
            TransposeBlock(64,   32),  # → (32,  32,  512)
            TransposeBlock(32,   16),  # → (16,  64,  1024)
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            # → (1, 128, 2048)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x = self.project(z).view(B, 256, self._FREQ_SEED, self._TIME_SEED)
        x = self.decode(x)                                   # (B, 1, 128, 2048)
        # Crop (or pad) time axis — no stretch interpolation
        t = x.shape[-1]
        if t >= self.target_frames:
            x = x[:, :, :, :self.target_frames]
        else:
            x = F.pad(x, (0, self.target_frames - t))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SoundscapeModel(nn.Module):
    """End-to-end VAE: (mel, env) → reconstructed mel.

    encode() returns (mu, log_var).
    forward() returns (recon, mu, log_var) so the training loop can compute
    the KL loss alongside the reconstruction loss.

    At generation time, sample z ~ N(0, I) and pass through decoder.
    At analysis time, use mu directly (deterministic, no noise).
    """

    def __init__(
        self,
        env_dim:       int = 29,
        embed_dim:     int = 512,
        latent_dim:    int = 256,
        target_frames: int = FRAMES_PER_CLIP,
    ):
        super().__init__()
        self.encoder = AudioEncoder(embed_dim=embed_dim)
        self.fusion  = FusionMLP(audio_dim=embed_dim, env_dim=env_dim, latent_dim=latent_dim)
        self.decoder = MelDecoder(latent_dim=latent_dim, target_frames=target_frames)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + ε·σ using the reparameterisation trick.

        During eval (torch.no_grad) this still samples — call with mu directly
        if you want the deterministic embedding (analysis mode).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, mel: torch.Tensor, env: torch.Tensor):
        """Returns (mu, log_var) — both (B, latent_dim)."""
        audio_emb      = self.encoder(mel)
        mu, log_var    = self.fusion(audio_emb, env)
        return mu, log_var

    def forward(self, mel: torch.Tensor, env: torch.Tensor):
        """Returns (recon, mu, log_var).

        recon   : (B, 1, n_mels, T) reconstructed mel-spectrogram
        mu      : (B, latent_dim)
        log_var : (B, latent_dim)
        """
        mu, log_var = self.encode(mel, env)
        z           = self.reparameterise(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var
