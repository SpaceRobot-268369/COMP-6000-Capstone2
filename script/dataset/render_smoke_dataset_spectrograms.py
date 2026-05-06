"""
Render Layer A-format mel-spectrogram artifacts for a smoke-test dataset.

This is for audit/inspection artifacts only. It does not change audio, captions,
manifests, model checkpoints, or training code.

Usage:
    python3 script/dataset/render_smoke_dataset_spectrograms.py \
      --dataset-dir resources/site_257_bowra-dry-a/smoking_test_dataset
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "acoustic_ai"))

SPECTROGRAM_DURATION_S = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Smoke-test dataset directory containing manifest.csv and clips/.",
    )
    parser.add_argument("--duration-s", type=float, default=SPECTROGRAM_DURATION_S)
    return parser.parse_args()


def write_mel_spectrogram(audio_path: Path, duration_s: float) -> None:
    from modules.ambient.diffusion.layer_a_visualization import (
        render_layer_a_mel_png_bytes,
        waveform_to_layer_a_mel_db,
    )

    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    target_samples = int(round(duration_s * sample_rate))
    if waveform.shape[0] > target_samples:
        waveform = waveform[:target_samples]
    elif waveform.shape[0] < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.shape[0]))

    mel_db = waveform_to_layer_a_mel_db(waveform, sample_rate).astype(np.float32)
    np.save(audio_path.with_name("mel_spectrogram.npy"), mel_db)
    audio_path.with_name("mel_spectrogram.png").write_bytes(
        render_layer_a_mel_png_bytes(mel_db, duration_s)
    )


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir

    manifest = dataset_dir / "manifest.csv"
    if not manifest.exists():
        print(f"[error] missing manifest: {manifest}", file=sys.stderr)
        return 1

    with open(manifest) as f:
        rows = list(csv.DictReader(f))

    errors: list[str] = []
    for row in tqdm(rows, desc="render spectrograms", unit="clip"):
        if row.get("status") and row["status"] != "ok":
            continue

        audio_path = dataset_dir / "clips" / row["clip_id"] / "audio.wav"
        if not audio_path.exists():
            errors.append(f"{row['clip_id']}: missing audio.wav")
            continue

        try:
            write_mel_spectrogram(audio_path, args.duration_s)
        except Exception as exc:
            errors.append(f"{row['clip_id']}: {exc}")

    if errors:
        print("\nErrors:")
        for err in errors[:20]:
            print(f"  {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return 1

    print(f"\nRendered {len(rows)} spectrogram artifact pairs in {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
