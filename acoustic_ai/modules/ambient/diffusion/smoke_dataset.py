"""Dataset for CLAP-conditioned diffusion training on the smoke-test clips.

Expects precomputed files from `precompute/precompute_smoke_latents.py`:
  <precomputed_dir>/latents.npy           shape (N, 256)  float32
  <precomputed_dir>/text_embeddings.npy   shape (N, 512)  float32
  <precomputed_dir>/index.csv             columns: clip_id, caption

Returns (latent (256,), text_emb (512,)) pairs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SmokeTestDataset(Dataset):
    """Smoke-test latent + CLAP embedding dataset.

    Args:
        precomputed_dir: directory containing latents.npy, text_embeddings.npy,
                         and index.csv (produced by precompute_smoke_latents.py).
    """

    LATENTS_FILE   = "latents.npy"
    TEXT_EMB_FILE  = "text_embeddings.npy"
    INDEX_FILE     = "index.csv"

    def __init__(self, precomputed_dir: Path):
        precomputed_dir = Path(precomputed_dir)

        lat_path  = precomputed_dir / self.LATENTS_FILE
        emb_path  = precomputed_dir / self.TEXT_EMB_FILE
        idx_path  = precomputed_dir / self.INDEX_FILE

        for p in (lat_path, emb_path, idx_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"precomputed file missing: {p}\n"
                    "Run: python3 acoustic_ai/precompute/precompute_smoke_latents.py"
                )

        self.latents   = np.load(lat_path).astype(np.float32)    # (N, 256)
        self.text_embs = np.load(emb_path).astype(np.float32)    # (N, 512)

        with open(idx_path) as f:
            self.index = list(csv.DictReader(f))

        if not (len(self.latents) == len(self.text_embs) == len(self.index)):
            raise ValueError(
                f"shape mismatch: latents={len(self.latents)}, "
                f"text_embs={len(self.text_embs)}, index={len(self.index)}"
            )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def latent_dim(self) -> int:
        return self.latents.shape[1]

    @property
    def cond_dim(self) -> int:
        return self.text_embs.shape[1]

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int):
        z = torch.from_numpy(self.latents[idx])     # (256,)
        c = torch.from_numpy(self.text_embs[idx])   # (512,)
        return z, c
