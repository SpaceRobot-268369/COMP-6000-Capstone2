"""Pre-compute mean latent vectors for each month-range × sample_bin combination.

Runs all training clips through the encoder in best.pt, groups the resulting
latent vectors by (month_range, sample_bin), averages each group, and saves the
result to acoustic_ai/latent_templates.npy.

The generation endpoint loads this file at startup instead of sampling random
noise — producing spectrograms grounded in real soundscape structure.

Usage (from project root):
  python3 acoustic_ai/precompute_latents.py
  python3 acoustic_ai/precompute_latents.py --split all
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MANIFEST_PATH  = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "site_257_training_manifest.csv"
OUTPUT_PATH    = Path(__file__).resolve().parent / "latent_templates.npy"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",    type=Path, default=MANIFEST_PATH)
    p.add_argument("--checkpoint",  type=Path,
                   default=Path(__file__).resolve().parent / "checkpoints" / "best.pt")
    p.add_argument("--output",      type=Path, default=OUTPUT_PATH)
    p.add_argument("--split",       default="train", choices=["train", "all"])
    p.add_argument("--batch-size",  type=int, default=32)
    args = p.parse_args()

    device = get_device()
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    # ── Load model ─────────────────────────────────────────────────────────────
    from model import SoundscapeModel
    from dataset import SoundscapeDataset, N_ENV_FEATURES
    from preprocess import SPEC_CFG, FRAMES_PER_CLIP

    ckpt        = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args  = ckpt.get("args", {})
    crop_secs   = saved_args.get("crop_seconds", 30.0)
    crop_frames = int(crop_secs * SPEC_CFG["sample_rate"] / SPEC_CFG["hop_length"]) if crop_secs > 0 else None

    model = SoundscapeModel(
        env_dim=N_ENV_FEATURES,
        embed_dim=saved_args.get("embed_dim", 512),
        latent_dim=saved_args.get("latent_dim", 256),
        target_frames=crop_frames or FRAMES_PER_CLIP,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    latent_dim = saved_args.get("latent_dim", 256)
    print(f"Latent dim : {latent_dim}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    ds = SoundscapeDataset(
        manifest_path=str(args.manifest),
        project_root=str(PROJECT_ROOT),
        split=args.split,
        crop_frames=crop_frames,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Clips      : {len(ds)}\n")

    # ── Collect raw env values (in dataset order, for nearest-neighbour lookup) ─
    from dataset import NUMERIC_COLS, month_range_for_month
    all_env_raw: list = []
    for idx in range(len(ds)):
        row = ds.df.iloc[idx]
        entry = {col: float(row[col]) for col in NUMERIC_COLS}
        entry["month"]       = float(row["month"])
        entry["day_of_year"] = float(row["day_of_year"])
        entry["month_range"] = month_range_for_month(row["month"])
        entry["sample_bin"] = str(row["sample_bin"])
        all_env_raw.append(entry)

    # ── Encode ────────────────────────────────────────────────────────────────
    bucket:      dict[str, list] = defaultdict(list)
    all_latents: list            = []
    all_envs:    list            = []

    with torch.no_grad():
        for mel, env, meta in tqdm(loader, desc="Encoding", unit="batch"):
            mu, _ = model.encode(mel.to(device), env.to(device))  # use mu (deterministic)
            z_np   = mu.cpu().numpy()
            env_np = env.cpu().numpy()

            for i in range(len(z_np)):
                month_range = meta["month_range"][i]
                sample_bin = meta["sample_bin"][i]
                key = f"{month_range}|{sample_bin}"
                bucket[key].append(z_np[i])
                all_latents.append(z_np[i])
                all_envs.append(env_np[i])

    # ── Average (group templates) ─────────────────────────────────────────────
    templates = {}
    print("\nGroup sizes:")
    for key in sorted(bucket):
        vecs = np.stack(bucket[key])          # (N, latent_dim)
        templates[key] = vecs.mean(axis=0)    # (latent_dim,)
        print(f"  {key:<25} : {len(vecs):>5} clips")

    # ── Save group templates ───────────────────────────────────────────────────
    np.save(str(args.output), templates)
    print(f"\nSaved {len(templates)} templates → {args.output}")

    # ── Save per-clip database for nearest-neighbour generation ───────────────
    clips_path = args.output.parent / "latent_clips.npy"
    np.save(str(clips_path), {
        "latents":  np.stack(all_latents).astype(np.float32),   # (N, latent_dim)
        "env_vecs": np.stack(all_envs).astype(np.float32),      # (N, env_dim) normalised
        "env_raw":  all_env_raw,                                 # list of N dicts, raw values
    })
    print(f"Saved {len(all_latents)} per-clip latents → {clips_path}")


if __name__ == "__main__":
    main()
