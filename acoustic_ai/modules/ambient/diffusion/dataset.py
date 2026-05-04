"""Dataset for latent diffusion training.

Loads precomputed latents and their 12-dim ambient-relevant conditioning
vector.

Expected files (produced by `precompute/precompute_ambient_latents.py`):
  data/ambient/latents/ambient_latents.npy            shape (N, 256) float32
  data/ambient/latents/ambient_latents_index.csv      N rows, columns:
      segment_id,
      hour_sin, hour_cos, month_sin, month_cos,
      season_spring, season_summer, season_autumn, season_winter,
      diel_dawn, diel_morning, diel_afternoon, diel_night
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


COND_COLUMNS = (
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "season_spring", "season_summer", "season_autumn", "season_winter",
    "diel_dawn", "diel_morning", "diel_afternoon", "diel_night",
)


class AmbientLatentDataset(Dataset):
    """Returns (latent (D,), cond (12,)) pairs from precomputed files."""

    def __init__(self, latents_npy: Path, index_csv: Path):
        self.latents = np.load(latents_npy)                       # (N, D)
        if self.latents.dtype != np.float32:
            self.latents = self.latents.astype(np.float32)

        self.cond = self._load_cond(index_csv)                     # (N, 12)

        if len(self.cond) != len(self.latents):
            raise ValueError(
                f"latents has {len(self.latents)} rows but index has {len(self.cond)}"
            )

    @staticmethod
    def _load_cond(path: Path) -> np.ndarray:
        rows: list[list[float]] = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append([float(r[c]) for c in COND_COLUMNS])
        return np.asarray(rows, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int):
        z = torch.from_numpy(self.latents[idx])
        c = torch.from_numpy(self.cond[idx])
        return z, c
