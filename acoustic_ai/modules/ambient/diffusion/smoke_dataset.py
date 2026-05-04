"""Dataset for CLAP-conditioned diffusion training on the smoke-test clips.

Expects precomputed files from `precompute/precompute_smoke_latents.py`:
  <precomputed_dir>/latents.npy           shape (N, 256)  float32  (raw VAE mu)
  <precomputed_dir>/text_embeddings.npy   shape (N, 512)  float32
  <precomputed_dir>/latent_scale.json     per-dim mean/std for normalisation
  <precomputed_dir>/index.csv             columns: clip_id, caption

Latents are per-dimension z-score normalised before being returned, so the
diffusion model always sees zero-mean unit-variance inputs — critical to
prevent activation explosion in the MLP denoiser (standard LDM practice).

Returns (latent_normalised (256,), text_emb (512,)) pairs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SmokeTestDataset(Dataset):
    """Smoke-test latent + CLAP embedding dataset.

    Args:
        precomputed_dir: directory containing latents.npy, text_embeddings.npy,
                         latent_scale.json, and index.csv (produced by
                         precompute_smoke_latents.py).
    """

    LATENTS_FILE   = "latents.npy"
    TEXT_EMB_FILE  = "text_embeddings.npy"
    SCALE_FILE     = "latent_scale.json"
    INDEX_FILE     = "index.csv"

    def __init__(self, precomputed_dir: Path):
        precomputed_dir = Path(precomputed_dir)

        lat_path   = precomputed_dir / self.LATENTS_FILE
        emb_path   = precomputed_dir / self.TEXT_EMB_FILE
        scale_path = precomputed_dir / self.SCALE_FILE
        idx_path   = precomputed_dir / self.INDEX_FILE

        for p in (lat_path, emb_path, idx_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"precomputed file missing: {p}\n"
                    "Run: python3 acoustic_ai/precompute/precompute_smoke_latents.py"
                )

        raw_latents    = np.load(lat_path).astype(np.float32)    # (N, 256)
        self.text_embs = np.load(emb_path).astype(np.float32)    # (N, 512)

        # Per-dimension normalisation (zero mean, unit std)
        if scale_path.exists():
            scale = json.loads(scale_path.read_text())
            self._lat_mean = np.array(scale["mean"], dtype=np.float32)  # (256,)
            self._lat_std  = np.array(scale["std"],  dtype=np.float32)  # (256,)
        else:
            # Fallback: fit on this dataset (only used if scale file is missing)
            self._lat_mean = raw_latents.mean(axis=0)
            self._lat_std  = np.maximum(raw_latents.std(axis=0), 1e-4)

        self.latents = (raw_latents - self._lat_mean) / self._lat_std  # (N, 256)

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
