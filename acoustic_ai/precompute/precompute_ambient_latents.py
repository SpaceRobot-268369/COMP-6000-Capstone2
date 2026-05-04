"""Precompute VAE latents over the cleaned ambient segment pool.

For each segment in `data/ambient/ambient_index.csv`:
  1. Load the segment .wav
  2. Compute the mel spectrogram
  3. Look up the source clip's env row in the training manifest
  4. Build the 29-dim env vector via SoundscapeDataset's encoder
  5. Forward through the frozen VAE encoder + fusion → mu (256-dim)
  6. Save mu + the 12-dim ambient-relevant cond into output files

Outputs:
  acoustic_ai/data/ambient/latents/ambient_latents.npy           shape (N, 256)
  acoustic_ai/data/ambient/latents/ambient_latents_index.csv     N rows

The 12-dim conditioning vector uses only ambient-relevant features
(`hour_sin/cos`, `month_sin/cos`, season one-hot, diel one-hot) — matching
Layer A's retrieval philosophy. See
`.claude/context/ai/pipeline_design.md` § Layer A.

Also writes a brief stats report on latent activity per dimension; if the
encoder collapses cleaned segments into a narrow region, this surfaces it
early — see `.claude/context/dev/layer-a-ambient-method-3/plan.md` Decision 1.

Usage (from project root):
  python3 acoustic_ai/precompute/precompute_ambient_latents.py
  python3 acoustic_ai/precompute/precompute_ambient_latents.py --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai" / "modules" / "ambient"))

from modules.ambient.dataset import SoundscapeDataset, N_ENV_FEATURES, MEL_MIN_DB, MEL_MAX_DB  # noqa: E402
from modules.ambient.model import SoundscapeModel                                              # noqa: E402
from modules.ambient.preprocess import audio_to_tensor, SPEC_CFG, FRAMES_PER_CLIP              # noqa: E402

# Match dataset.py's onehot order so cond columns are deterministic.
SEASONS = ("spring", "summer", "autumn", "winter")
DIEL = ("dawn", "morning", "afternoon", "night")

DEFAULT_INDEX = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "ambient_index.csv"
DEFAULT_SEGMENTS = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "ambient_segments"
DEFAULT_MANIFEST = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "site_257_training_manifest.csv"
DEFAULT_VAE_CKPT = PROJECT_ROOT / "acoustic_ai" / "checkpoints" / "ambient" / "best.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "latents"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Encode this many segments per VAE forward.")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def pick_device(arg: str | None) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vae(checkpoint: Path, device: torch.device) -> SoundscapeModel:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    crop_secs = saved_args.get("crop_seconds", 30.0)
    crop_frames = (
        int(crop_secs * SPEC_CFG["sample_rate"] / SPEC_CFG["hop_length"])
        if crop_secs > 0 else FRAMES_PER_CLIP
    )
    model = SoundscapeModel(
        env_dim=N_ENV_FEATURES,
        embed_dim=saved_args.get("embed_dim", 512),
        latent_dim=saved_args.get("latent_dim", 256),
        target_frames=crop_frames,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_segment_to_manifest_row(
    ambient_index: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Join ambient segments to manifest by clip_path. Returns a single
    DataFrame with one row per segment and the joined manifest columns.

    The match key is the clip_path string; ambient_index.source_clip is the
    same form (relative to project root).
    """
    merged = ambient_index.merge(
        manifest, how="left", left_on="source_clip", right_on="clip_path"
    )
    missing = merged["clip_path"].isna().sum()
    if missing:
        print(f"[warn] {missing} segments could not be joined to manifest "
              f"(missing clip_path). They will be skipped.", file=sys.stderr)
    merged = merged[merged["clip_path"].notna()].reset_index(drop=True)
    return merged


def load_segment_mel(seg_path: Path) -> torch.Tensor:
    """Load .wav → log-mel → tensor, normalise to [0,1] like the trainer."""
    mel = audio_to_tensor(str(seg_path))                  # (1, n_mels, T)
    mel = (mel - MEL_MIN_DB) / (MEL_MAX_DB - MEL_MIN_DB)
    return mel


def cond12_from_segment(row: pd.Series) -> list[float]:
    """Build the 12-dim conditioning vector from the joined manifest row.

    Uses sample_bin (==diel) and season columns + cyclic hour/month built
    from the ambient_index row (which already provides hour_sin/cos and
    month_sin/cos directly).
    """
    season = str(row["season"]).strip().lower()
    diel = str(row["sample_bin"]).strip().lower()

    return [
        float(row["hour_sin"]), float(row["hour_cos"]),
        float(row["month_sin"]), float(row["month_cos"]),
        *[1.0 if s == season else 0.0 for s in SEASONS],
        *[1.0 if d == diel else 0.0 for d in DIEL],
    ]


def report_latent_stats(latents: np.ndarray) -> None:
    """Quick stats so we can spot collapse early."""
    mean = latents.mean(axis=0)
    std = latents.std(axis=0)
    active = (std > 0.05).sum()
    print(f"\nlatent stats over {len(latents)} segments:")
    print(f"  mean of |mu|        : {np.abs(mean).mean():.3f}")
    print(f"  mean per-dim std    : {std.mean():.3f}")
    print(f"  active dims (std>.05): {active} / {latents.shape[1]}")
    if active < latents.shape[1] * 0.25:
        print("  [warn] fewer than 25% of dims are active — encoder may be "
              "collapsing cleaned ambient. Consider VAE fine-tune. See plan.md.")


def main() -> int:
    args = parse_args()

    if not args.index.exists():
        print(f"ambient index not found: {args.index}", file=sys.stderr)
        return 1
    if not args.vae_checkpoint.exists():
        print(f"VAE checkpoint not found: {args.vae_checkpoint}", file=sys.stderr)
        return 1

    device = pick_device(args.device)
    print(f"device: {device}")

    # Manifest + ambient index
    print("loading manifest + ambient index...")
    manifest = pd.read_csv(args.manifest)
    ambient_index = pd.read_csv(args.index)
    merged = build_segment_to_manifest_row(ambient_index, manifest)
    print(f"  {len(merged)} segments joined to manifest")

    # We need SoundscapeDataset *only* for its env-encoding stats + builder.
    # Pass split="all" so it loads everything for normalisation stats.
    print("fitting env normalisation stats from training manifest...")
    env_builder = SoundscapeDataset(
        manifest_path=str(args.manifest),
        project_root=str(PROJECT_ROOT),
        split="all",
        crop_frames=None,
    )

    # Load VAE
    print(f"loading VAE checkpoint {args.vae_checkpoint}")
    vae = load_vae(args.vae_checkpoint, device)

    # Iterate
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = args.out_dir / "ambient_latents.npy"
    out_csv = args.out_dir / "ambient_latents_index.csv"

    cond_columns = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "season_spring", "season_summer", "season_autumn", "season_winter",
        "diel_dawn", "diel_morning", "diel_afternoon", "diel_night",
    ]

    all_latents: list[np.ndarray] = []
    all_rows: list[list] = []

    bs = max(1, args.batch_size)
    pbar = tqdm(range(0, len(merged), bs), desc="encode")
    with torch.no_grad():
        for start in pbar:
            chunk = merged.iloc[start : start + bs]
            mels: list[torch.Tensor] = []
            envs: list[torch.Tensor] = []
            keep: list[int] = []

            for idx, row in chunk.iterrows():
                seg_path = args.segments / f"{row['segment_id']}.wav"
                if not seg_path.exists():
                    continue
                try:
                    mel = load_segment_mel(seg_path)
                    env = env_builder._build_env_vector(row)        # noqa: SLF001
                except Exception as e:                              # pragma: no cover
                    print(f"\n[warn] {row['segment_id']}: {e}", file=sys.stderr)
                    continue
                mels.append(mel)
                envs.append(env)
                keep.append(idx)

            if not mels:
                continue

            # Pad/crop time axis so we can stack into a batch — VAE encoder is
            # fully convolutional + AdaptiveAvgPool, so any length works as
            # long as a batch is uniform.
            T = max(m.shape[-1] for m in mels)
            mels = [
                torch.nn.functional.pad(m, (0, T - m.shape[-1])) if m.shape[-1] < T else m
                for m in mels
            ]
            mel_batch = torch.stack(mels, dim=0).to(device)         # (B, 1, n_mels, T)
            env_batch = torch.stack(envs, dim=0).to(device)         # (B, 29)

            mu, _ = vae.encode(mel_batch, env_batch)                # (B, 256)
            all_latents.append(mu.cpu().numpy())

            for kept_idx, mu_i in zip(keep, mu.cpu().numpy()):
                row = merged.loc[kept_idx]
                all_rows.append([row["segment_id"], *cond12_from_segment(row)])

    if not all_latents:
        print("no segments encoded.", file=sys.stderr)
        return 1

    latents = np.concatenate(all_latents, axis=0).astype(np.float32)
    np.save(out_npy, latents)
    print(f"\nwrote {out_npy}  shape={latents.shape}  dtype={latents.dtype}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", *cond_columns])
        writer.writerows(all_rows)
    print(f"wrote {out_csv}  ({len(all_rows)} rows)")

    report_latent_stats(latents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
