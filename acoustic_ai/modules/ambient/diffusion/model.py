"""Env-conditioned 1D denoising network for the VAE latent space.

The VAE produces a (B, 256) latent vector — a single point per segment, not
a feature map. So the denoiser is not a 2D conv UNet; it is a stack of
residual MLP blocks with timestep + condition injected via AdaLN.

Architecture:

  x_t  ──▶ in_proj ─▶ ┐
                      │
  t   ──▶ sin emb ─▶ MLP ─▶ t_emb ──┐
                                     ├──▶ AdaLN(γ, β) ─▶ ResBlock ─▶ ... ─▶ out_proj ─▶ v
  cond ──▶ cond MLP ─▶ c_emb ───────┘
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (Vaswani et al. positional encoding)
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("timestep embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long → (B, dim) float."""
        half = self.dim // 2
        device = t.device
        # standard "10000^(2i/d)" frequencies
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)        # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# AdaLN-conditioned residual MLP block
# ---------------------------------------------------------------------------

class AdaLNResBlock(nn.Module):
    """LayerNorm + scale/shift from condition + MLP + residual.

    The (γ, β) come from a small MLP over `cond_emb`. Initial γ is biased
    toward 0 so each block starts as identity, which stabilises early
    training (akin to "zero conv" trick).
    """

    def __init__(self, dim: int, cond_dim: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, 2 * dim)
        # Init modulation to zero so γ=β=0 → block is identity at init.
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.modulation(cond_emb)             # (B, 2D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)          # each (B, D)
        h = self.norm(x) * (1.0 + gamma) + beta
        return x + self.mlp(h)


# ---------------------------------------------------------------------------
# Full denoiser
# ---------------------------------------------------------------------------

class LatentDenoiser(nn.Module):
    """v-prediction denoiser over (B, latent_dim) latents.

    Args:
      latent_dim:   VAE latent dimensionality (256 in our case).
      cond_dim:     env conditioning dimensionality (12).
      hidden_dim:   internal MLP width.
      num_blocks:   number of AdaLN res blocks.
      time_emb_dim: timestep sinusoid embedding width.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        cond_dim: int = 12,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Project x_t into hidden space
        self.in_proj = nn.Linear(latent_dim, hidden_dim)

        # Timestep embedding pipeline: sin → MLP → time_emb
        self.t_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition embedding pipeline: env vector → MLP → cond_emb
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # The combined embedding fed to AdaLN is (t_emb + cond_emb)
        self.blocks = nn.ModuleList(
            [AdaLNResBlock(hidden_dim, cond_dim=hidden_dim) for _ in range(num_blocks)]
        )

        # Final norm (also AdaLN-modulated) + projection back to latent space
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)

        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        # Init final projection small so initial outputs are near zero (i.e.
        # near-zero v at init → x_0 ≈ alpha·x_t, behaviour close to identity).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x_t: (B, latent_dim). t: (B,) long. cond: (B, cond_dim).

        Returns v-prediction of shape (B, latent_dim).
        """
        h = self.in_proj(x_t)                                  # (B, H)
        t_emb = self.t_embed(t)                                # (B, H)
        c_emb = self.cond_embed(cond)                          # (B, H)
        emb = t_emb + c_emb                                    # (B, H)

        for block in self.blocks:
            h = block(h, emb)

        # Final AdaLN
        gamma_beta = self.final_modulation(emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = self.final_norm(h) * (1.0 + gamma) + beta

        return self.out_proj(h)                                # (B, latent_dim)
