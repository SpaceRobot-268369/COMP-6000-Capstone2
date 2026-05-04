"""Precompute VAE latents and CLAP text embeddings for the smoke-test dataset.

Reads `resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv`,
then for each clip:
  1. Loads audio.wav (16 kHz mono) → resamples to 22 050 Hz
  2. Computes 128-bin log-mel spectrogram (SPEC_CFG)
  3. Looks up the recording row in site_257_training_manifest.csv by
     source_clip path to build the 29-dim env vector
  4. Encodes (mel, env) through the frozen VAE → mu (256-dim latent)
  5. Reads caption.txt → encodes with CLAP (laion/clap-htsat-unfused) → 512-dim

Outputs (written to <smoke_dir>/precomputed/):
  latents.npy           (N, 256) float32 — VAE mu vectors
  text_embeddings.npy   (N, 512) float32 — CLAP-normalised text embeddings
  index.csv             N rows: clip_id, caption, status

Usage:
    python3 acoustic_ai/precompute/precompute_smoke_latents.py
    python3 acoustic_ai/precompute/precompute_smoke_latents.py --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai" / "modules" / "ambient"))

from modules.ambient.dataset import SoundscapeDataset, N_ENV_FEATURES, MEL_MIN_DB, MEL_MAX_DB  # noqa: E402
from modules.ambient.model import SoundscapeModel                                              # noqa: E402
from modules.ambient.preprocess import load_audio, waveform_to_melspec, melspec_to_tensor, SPEC_CFG  # noqa: E402
from modules.ambient.diffusion.clap_encoder import ClapTextEncoder                            # noqa: E402

SMOKE_DIR     = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "smoking_test_dataset"
MANIFEST      = SMOKE_DIR / "manifest.csv"
TRAINING_MAN  = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "site_257_training_manifest.csv"
VAE_CKPT      = PROJECT_ROOT / "acoustic_ai" / "checkpoints" / "ambient" / "best.pt"
OUT_DIR       = SMOKE_DIR / "precomputed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-dir",  type=Path, default=SMOKE_DIR)
    p.add_argument("--manifest",   type=Path, default=MANIFEST)
    p.add_argument("--training-manifest", type=Path, default=TRAINING_MAN)
    p.add_argument("--vae-checkpoint",    type=Path, default=VAE_CKPT)
    p.add_argument("--out-dir",    type=Path, default=OUT_DIR)
    p.add_argument("--batch-size", type=int,  default=8)
    p.add_argument("--clap-device", type=str, default=None,
                   help="Device for CLAP encoding. Defaults to same as main device.")
    p.add_argument("--device",     type=str,  default=None)
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
    """Load the frozen VAE from checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})

    from modules.ambient.preprocess import FRAMES_PER_CLIP  # noqa: E402
    crop_secs   = saved_args.get("crop_seconds", 30.0)
    crop_frames = (
        int(crop_secs * SPEC_CFG["sample_rate"] / SPEC_CFG["hop_length"])
        if crop_secs > 0
        else FRAMES_PER_CLIP
    )

    model = SoundscapeModel(
        env_dim    =saved_args.get("env_dim",    N_ENV_FEATURES),
        embed_dim  =saved_args.get("embed_dim",  512),
        latent_dim =saved_args.get("latent_dim", 256),
        target_frames=crop_frames,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_clip_as_mel(audio_path: Path) -> torch.Tensor:
    """Load 16 kHz WAV, resample to 22 050 Hz, return (1, n_mels, T) mel."""
    # load_audio already resamples to SPEC_CFG["sample_rate"] (22 050 Hz)
    waveform = load_audio(str(audio_path), target_sr=SPEC_CFG["sample_rate"])
    log_mel  = waveform_to_melspec(waveform)               # (128, T)
    mel      = melspec_to_tensor(log_mel)                  # (1, 128, T)
    # Normalise dB from [−80, 0] → [0, 1] (matches training)
    mel = (mel - MEL_MIN_DB) / (MEL_MAX_DB - MEL_MIN_DB)
    return mel


def build_env_builder(training_manifest: Path):
    """Create a SoundscapeDataset on the full training manifest — only used
    for its normalisation stats and _build_env_vector helper."""
    return SoundscapeDataset(
        manifest_path=str(training_manifest),
        project_root =str(PROJECT_ROOT),
        split        ="all",
        crop_frames  =None,
    )


def main() -> int:
    args = parse_args()

    for required in (args.manifest, args.training_manifest, args.vae_checkpoint):
        if not required.exists():
            print(f"[error] missing required file: {required}", file=sys.stderr)
            return 1

    device      = pick_device(args.device)
    clap_device = pick_device(args.clap_device) if args.clap_device else device
    print(f"device: {device}  |  clap_device: {clap_device}")

    # ---- load manifest ----
    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    print(f"smoke-test manifest: {len(rows)} clips")

    # ---- load training manifest (for env normalisation + row lookup) ----
    print("loading training manifest + fitting env stats …")
    import pandas as pd
    train_df   = pd.read_csv(args.training_manifest)
    train_df.replace("", float("nan"), inplace=True)
    train_df["cloud_clearness_index"] = train_df["cloud_clearness_index"].fillna(0.0)
    from modules.ambient.dataset import NUMERIC_COLS, CIRCULAR_COLS  # noqa: E402
    for col in NUMERIC_COLS:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)
    for col, _ in CIRCULAR_COLS:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)

    env_builder = build_env_builder(args.training_manifest)
    # Build a lookup: clip_path → pd.Series row
    clip_path_index = {row["clip_path"]: row for _, row in train_df.iterrows()}

    # ---- load VAE ----
    print(f"loading VAE checkpoint: {args.vae_checkpoint}")
    vae = load_vae(args.vae_checkpoint, device)

    # ---- CLAP encoder (lazy-loaded on first call) ----
    clap = ClapTextEncoder(device=clap_device)

    # ---- encode ----
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_latents:   list[np.ndarray] = []
    all_text_embs: list[np.ndarray] = []
    index_rows:    list[dict]       = []

    bs    = max(1, args.batch_size)
    total = len(rows)

    for start in tqdm(range(0, total, bs), desc="encode batches"):
        chunk = rows[start : start + bs]

        mels:     list[torch.Tensor] = []
        envs:     list[torch.Tensor] = []
        captions: list[str]          = []
        clip_ids: list[str]          = []
        statuses: list[str]          = []

        for r in chunk:
            clip_id  = r["clip_id"]
            clip_dir = args.smoke_dir / "clips" / clip_id
            audio_p  = clip_dir / "audio.wav"
            cap_p    = clip_dir / "caption.txt"
            meta_p   = clip_dir / "meta.json"

            # source_clip lives in meta.json (not in the manifest)
            src_clip = ""
            if meta_p.exists():
                import json
                try:
                    src_clip = json.loads(meta_p.read_text()).get("source_clip", "")
                except Exception:
                    pass

            # Read caption
            if cap_p.exists():
                caption = cap_p.read_text().strip()
            else:
                caption = r.get("caption", "")

            # Load mel
            if not audio_p.exists():
                print(f"\n[skip] {clip_id} — audio not found: {audio_p}")
                statuses.append("audio_missing")
                # Still encode CLAP (no mel/env needed)
                captions.append(caption)
                clip_ids.append(clip_id)
                mels.append(None)
                envs.append(None)
                continue

            try:
                mel = load_clip_as_mel(audio_p)
            except Exception as e:
                print(f"\n[skip] {clip_id} — mel error: {e}")
                statuses.append("mel_error")
                captions.append(caption)
                clip_ids.append(clip_id)
                mels.append(None)
                envs.append(None)
                continue

            # Build env vector — look up the training manifest row
            # source_clip is like "resources/.../site_257_item_5392/site_257_item_5392_clip_001.webm"
            env_row = clip_path_index.get(src_clip)
            if env_row is None:
                # Fall back to zeros — env vector doesn't affect the diffusion
                # conditioning (that's CLAP), only the VAE encoder quality
                env = torch.zeros(N_ENV_FEATURES, dtype=torch.float32)
            else:
                try:
                    env = env_builder._build_env_vector(env_row)   # noqa: SLF001
                except Exception:
                    env = torch.zeros(N_ENV_FEATURES, dtype=torch.float32)

            mels.append(mel)
            envs.append(env)
            captions.append(caption)
            clip_ids.append(clip_id)
            statuses.append("ok")

        # --- VAE encode those with audio ---
        valid_mask = [i for i, m in enumerate(mels) if m is not None]

        if valid_mask:
            T = max(mels[i].shape[-1] for i in valid_mask)
            mel_batch = torch.stack(
                [
                    torch.nn.functional.pad(mels[i], (0, T - mels[i].shape[-1]))
                    if mels[i].shape[-1] < T else mels[i]
                    for i in valid_mask
                ],
                dim=0,
            ).to(device)                                               # (B, 1, 128, T)
            env_batch = torch.stack([envs[i] for i in valid_mask], dim=0).to(device)  # (B, 29)

            with torch.no_grad():
                mu, _ = vae.encode(mel_batch, env_batch)               # (B, 256)
            mu_np = mu.cpu().numpy()
        else:
            mu_np = np.zeros((0, 256), dtype=np.float32)

        # Map results back
        valid_iter = iter(range(len(valid_mask)))
        for i, (clip_id, caption, status) in enumerate(zip(clip_ids, captions, statuses)):
            if status == "ok":
                vi = next(valid_iter)
                lat = mu_np[vi]
            else:
                lat = np.zeros(256, dtype=np.float32)

            all_latents.append(lat)
            index_rows.append({"clip_id": clip_id, "caption": caption, "status": status})

        # --- CLAP encode all captions in this batch ---
        text_embs = clap(captions)                                     # (B, 512) on clap_device
        all_text_embs.append(text_embs.cpu().numpy())

    # ---- save ----
    latents_arr   = np.stack(all_latents, axis=0).astype(np.float32)  # (N, 256)
    text_embs_arr = np.concatenate(all_text_embs, axis=0).astype(np.float32)  # (N, 512)

    np.save(args.out_dir / "latents.npy",         latents_arr)
    np.save(args.out_dir / "text_embeddings.npy", text_embs_arr)

    with open(args.out_dir / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "caption", "status"])
        w.writeheader()
        w.writerows(index_rows)

    # ---- latent normalisation stats ----
    # Compute per-dimension mean and std so the diffusion model trains on
    # zero-mean unit-variance latents (standard LDM practice; prevents
    # activation explosion when some VAE dims have very large absolute values).
    lat_mean = latents_arr.mean(axis=0)                        # (256,)
    lat_std  = np.maximum(latents_arr.std(axis=0), 1e-4)       # (256,) — floor avoids /0
    import json
    scale_path = args.out_dir / "latent_scale.json"
    with open(scale_path, "w") as f:
        json.dump({
            "mean": lat_mean.tolist(),
            "std":  lat_std.tolist(),
            "global_mean": float(latents_arr.mean()),
            "global_std":  float(latents_arr.std()),
        }, f)

    ok_count = sum(1 for r in index_rows if r["status"] == "ok")
    print(f"\nDone: {ok_count}/{len(index_rows)} clips encoded successfully")
    print(f"  latents        : {latents_arr.shape}  → {args.out_dir / 'latents.npy'}")
    print(f"  text_embeddings: {text_embs_arr.shape}  → {args.out_dir / 'text_embeddings.npy'}")
    print(f"  latent_scale   : {scale_path}")
    print(f"  index          : {args.out_dir / 'index.csv'}")

    # Quick stats (on raw latents)
    print(f"\nLatent stats (raw): "
          f"min={latents_arr.min():.3f}  max={latents_arr.max():.3f}  "
          f"global_std={latents_arr.std():.3f}  "
          f"active-dims(std>0.05)={int((lat_std > 0.05).sum())}/{latents_arr.shape[1]}")

    # Verify normalised range
    lat_norm = (latents_arr - lat_mean) / lat_std
    print(f"Latent stats (normalised): "
          f"min={lat_norm.min():.3f}  max={lat_norm.max():.3f}  "
          f"std={lat_norm.std():.3f}  (expect ≈1)")

    emb_norms = np.linalg.norm(text_embs_arr, axis=1)
    print(f"CLAP emb norms: min={emb_norms.min():.3f}  max={emb_norms.max():.3f}  "
          f"mean={emb_norms.mean():.3f}  (expect ≈1 if normalised)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
