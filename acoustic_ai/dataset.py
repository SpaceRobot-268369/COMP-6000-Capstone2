"""PyTorch Dataset for the site 257 MVP training manifest.

Each item returns:
  mel      — (1, n_mels, time_frames) float32 tensor  — audio input
  env      — (N_ENV_FEATURES,) float32 tensor          — conditioning features
        meta     — dict with recording_id, clip_path, sample_bin, month_range, sample_local_date

Environmental features
----------------------
Numeric features are normalised with pre-computed mean/std (fit on training
split only). Categorical features are encoded as follows:
  - month_range   : one-hot (4 dims: December-February/March-May/June-August/September-November)
  - sample_bin    : one-hot (4 dims: dawn/morning/afternoon/night)
  - wind_direction: sin/cos encoding (2 dims, preserves circularity)
  - month         : sin/cos encoding (2 dims)
  - day_of_year   : sin/cos encoding (2 dims)
  cloud_clearness_index is empty for nighttime rows — filled with 0.0
  (the model should not rely on it at night; sample_bin encodes time-of-day).
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import numpy as np
from preprocess import audio_to_tensor

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

# Raw numeric columns (will be z-score normalised)
NUMERIC_COLS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "precipitation_mm",
    "solar_radiation_wm2",
    "cloud_clearness_index",  # NaN → 0.0 before normalisation
    "surface_pressure_kpa",
    "temp_max_c",
    "temp_min_c",
    "precipitation_daily_mm",
    "wind_max_ms",
    "days_since_rain",
    "daylight_hours",
    "hour_utc",
    "hour_local",
]

# Circular-encoded columns: (column, period)
CIRCULAR_COLS = [
    ("wind_direction_deg", 360),
    ("month",              12),
    ("day_of_year",        365),
]

# One-hot columns and their fixed category order
ONEHOT_COLS = {
    "month_range": ["december-february", "march-may", "june-august", "september-november"],
    "sample_bin": ["dawn", "morning", "afternoon", "night"],
}

# Total env feature dimension:
# 15 numeric + 3*2 circular + 4 month-range + 4 bin = 15 + 6 + 4 + 4 = 29
N_ENV_FEATURES = len(NUMERIC_COLS) + 2 * len(CIRCULAR_COLS) + sum(len(v) for v in ONEHOT_COLS.values())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

MEL_MIN_DB = -80.0   # matches top_db=80 in SPEC_CFG
MEL_MAX_DB =   0.0


def month_range_for_month(month: float) -> str:
    m = int(round(float(month)))
    if m == 12 or m in (1, 2):
        return "december-february"
    if 3 <= m <= 5:
        return "march-may"
    if 6 <= m <= 8:
        return "june-august"
    return "september-november"


class SoundscapeDataset(Dataset):
    """Loads clips from site_257_training_manifest.csv.

    Args:
        manifest_path : path to site_257_training_manifest.csv
        project_root  : project root used to resolve clip_path column
        split         : "train" | "val" | "all"
        val_fraction  : fraction of recordings held out for validation
        seed          : random seed for split (stratified by sample_bin)
        stats         : pre-computed (means, stds) dict for normalisation;
                        if None, computed from this split (use training split stats
                        when creating the val split)
        crop_frames   : if set, return a random time crop of this many frames
                        instead of the full 12 921-frame clip (speeds up training)
    """

    def __init__(
        self,
        manifest_path: str,
        project_root: str,
        split: str = "train",
        val_fraction: float = 0.15,
        seed: int = 42,
        stats: Optional[dict] = None,
        crop_frames: Optional[int] = None,
    ):
        self.crop_frames = crop_frames
        self.project_root = Path(project_root)
        df = pd.read_csv(manifest_path)

        # Replace empty strings with NaN then fill numeric NaNs
        df.replace("", np.nan, inplace=True)
        df["cloud_clearness_index"] = df["cloud_clearness_index"].fillna(0.0)
        for col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        for col, _ in CIRCULAR_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Train / val split — stratified by sample_bin
        if split != "all":
            rng = np.random.default_rng(seed)
            val_ids: set = set()
            for bin_name in ONEHOT_COLS["sample_bin"]:
                recs = df[df["sample_bin"] == bin_name]["recording_id"].unique()
                n_val = max(1, int(len(recs) * val_fraction))
                val_ids.update(rng.choice(recs, size=n_val, replace=False).tolist())

            if split == "train":
                df = df[~df["recording_id"].isin(val_ids)].reset_index(drop=True)
            else:  # val
                df = df[df["recording_id"].isin(val_ids)].reset_index(drop=True)

        self.df = df

        # Normalisation stats (fit on whatever split is passed in)
        if stats is None:
            self.stats = self._compute_stats(df)
        else:
            self.stats = stats

    # ── Public helpers ────────────────────────────────────────────────────────

    @property
    def env_dim(self) -> int:
        return N_ENV_FEATURES

    def get_stats(self) -> dict:
        """Return normalisation stats — pass to val/test split constructors."""
        return self.stats

    # ── Internal ─────────────────────────────────────────────────────────────

    def _compute_stats(self, df: pd.DataFrame) -> dict:
        means = df[NUMERIC_COLS].mean().to_dict()
        stds  = df[NUMERIC_COLS].std().fillna(1.0).replace(0, 1.0).to_dict()
        return {"means": means, "stds": stds}

    def _build_env_vector(self, row: pd.Series) -> torch.Tensor:
        parts = []

        # 1. Normalised numeric features
        for col in NUMERIC_COLS:
            val  = float(row[col])
            mean = self.stats["means"].get(col, 0.0)
            std  = self.stats["stds"].get(col, 1.0)
            parts.append((val - mean) / std)

        # 2. Circular encodings
        for col, period in CIRCULAR_COLS:
            val = float(row[col])
            parts.append(math.sin(2 * math.pi * val / period))
            parts.append(math.cos(2 * math.pi * val / period))

        # 3. One-hot encodings
        for col, categories in ONEHOT_COLS.items():
            if col == "month_range" and (
                col not in row or pd.isna(row[col]) or str(row[col]).strip() == ""
            ):
                val = month_range_for_month(row["month"])
            else:
                val = str(row[col]).strip().lower()
            parts.extend([1.0 if val == c else 0.0 for c in categories])

        return torch.tensor(parts, dtype=torch.float32)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        row  = self.df.iloc[idx]
        path = self.project_root / row["clip_path"]

        npy = Path(str(path) + ".npy")       # clip_001.webm.npy
        if npy.exists():
            mel = torch.from_numpy(np.load(str(npy)))
        else:
            mel = audio_to_tensor(str(Path(str(path) + ".wav")))  # clip_001.webm.wav

        # Normalise dB values from [−80, 0] → [0, 1]
        mel = (mel - MEL_MIN_DB) / (MEL_MAX_DB - MEL_MIN_DB)

        # Random time crop (avoids processing full 300 s tensor every step)
        if self.crop_frames is not None and mel.shape[-1] > self.crop_frames:
            max_start = mel.shape[-1] - self.crop_frames
            start = int(torch.randint(0, max_start, (1,)).item())
            mel = mel[:, :, start : start + self.crop_frames]

        env = self._build_env_vector(row)

        meta = {
            "recording_id":      row["recording_id"],
            "clip_path":         row["clip_path"],
            "sample_bin":        row["sample_bin"],
            "month_range":       month_range_for_month(row["month"]),
            "sample_local_date": row["sample_local_date"],
        }
        return mel, env, meta
