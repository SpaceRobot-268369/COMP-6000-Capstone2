"""Noise schedule + v-prediction utilities + DDIM sampler.

All math operates on tensors of shape (B, latent_dim). Timestep `t` is
`(B,)` long tensor of integers in [0, T-1].

References:
  - Nichol & Dhariwal (2021), "Improved DDPM" — cosine schedule
  - Salimans & Ho (2022), "Progressive Distillation" — v-prediction
  - Song et al. (2021), "DDIM" — deterministic sampler
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Cosine alpha-bar schedule (Nichol & Dhariwal 2021, eqn 17)
# ---------------------------------------------------------------------------

def make_cosine_alpha_bars(num_train_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Returns alpha_bar[t] for t in [0, T], length T+1."""
    steps = num_train_timesteps + 1
    t = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float64)
    f = torch.cos(((t / num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = (f / f[0]).clamp(min=1e-8, max=1.0)
    return alpha_bar.float()


class NoiseSchedule:
    """Holds alpha_bar and exposes alpha_t / sigma_t lookups."""

    def __init__(self, num_train_timesteps: int = 1000, schedule: str = "cosine"):
        if schedule != "cosine":
            raise ValueError(f"only 'cosine' supported, got {schedule!r}")
        self.T = num_train_timesteps
        self.alpha_bar = make_cosine_alpha_bars(num_train_timesteps)

    def to(self, device: torch.device) -> "NoiseSchedule":
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ab = self.alpha_bar[t]
        alpha = ab.sqrt().unsqueeze(-1)
        sigma = (1.0 - ab).sqrt().unsqueeze(-1)
        return alpha, sigma


# ---------------------------------------------------------------------------
# v-prediction conversions
# ---------------------------------------------------------------------------

def add_noise(x0: torch.Tensor, noise: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return alpha * x0 + sigma * noise


def v_target(x0: torch.Tensor, noise: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return alpha * noise - sigma * x0


def x0_from_v(x_t: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return alpha * x_t - sigma * v


def eps_from_v(x_t: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return sigma * x_t + alpha * v


# ---------------------------------------------------------------------------
# DDIM deterministic sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(
    model,
    cond: torch.Tensor,
    schedule: NoiseSchedule,
    *,
    num_inference_steps: int = 50,
    cfg_scale: float = 1.0,
    null_cond: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample latents with DDIM (eta=0, deterministic).

    Args:
      model:     callable model(x_t, t, cond) → v_pred
      cond:      (B, cond_dim) conditioning tensor
      schedule:  NoiseSchedule on the right device
      cfg_scale: > 1.0 amplifies the conditional direction. 1.0 = no CFG.
      null_cond: (B, cond_dim) zero-conditioning; required if cfg_scale != 1.
      generator: optional torch.Generator for reproducible noise.

    Returns:
      x_0: (B, latent_dim) sampled latents.
    """
    device = cond.device
    B, _ = cond.shape
    latent_dim = model.latent_dim

    timesteps = torch.linspace(schedule.T - 1, 0, num_inference_steps + 1, device=device).long()
    x = torch.randn((B, latent_dim), device=device, generator=generator)

    use_cfg = cfg_scale != 1.0
    if use_cfg and null_cond is None:
        null_cond = torch.zeros_like(cond)

    for i in range(num_inference_steps):
        t_now = timesteps[i].expand(B)
        t_next = timesteps[i + 1].expand(B)

        a_now, s_now = schedule.alpha_sigma(t_now)
        a_next, s_next = schedule.alpha_sigma(t_next)

        if use_cfg:
            v_uncond = model(x, t_now, null_cond)
            v_cond = model(x, t_now, cond)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t_now, cond)

        x0 = x0_from_v(x, v, a_now, s_now)
        eps = eps_from_v(x, v, a_now, s_now)
        x = a_next * x0 + s_next * eps

    return x
