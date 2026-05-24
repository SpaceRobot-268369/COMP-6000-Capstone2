#!/usr/bin/env python3
"""
Prepare Layer C raw 30-second clips.

This script:
1. Reads audio files from an input folder.
2. Converts them to mono 22050 Hz.
3. Crops or pads to 30 seconds.
4. Writes standardized .wav files into smoke_test/layer_c/raw_clips/<scene>/.

Important:
This script does NOT reliably detect speech/music/machines.
Human audit is still required.
"""

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SCENES = {"summer_rain", "winter_snow", "forest_bird"}

def load_audio(path: Path, sr: int):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y

def is_waveform_ok(y: np.ndarray, sr: int, min_duration: float = 25.0) -> bool:
    if y is None or len(y) == 0:
        return False
    duration = len(y) / sr
    if duration < min_duration:
        return False
    if np.isnan(y).any() or np.isinf(y).any():
        return False
    # Reject near silence
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 0.001:
        return False
    # Reject likely clipping
    clipped_ratio = float(np.mean(np.abs(y) > 0.98))
    if clipped_ratio > 0.01:
        return False
    return True

def crop_or_pad(y: np.ndarray, target_samples: int) -> np.ndarray:
    if len(y) >= target_samples:
        start = max(0, (len(y) - target_samples) // 2)
        return y[start:start + target_samples]
    out = np.zeros(target_samples, dtype=np.float32)
    out[:len(y)] = y
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="smoke_test/layer_c/raw_clips")
    parser.add_argument("--scene", required=True, choices=sorted(SCENES))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_scene_dir = Path(args.output_dir) / args.scene
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in ALLOWED_EXTS]
    target_samples = int(args.sr * args.duration)

    saved = 0
    rejected = 0

    for path in tqdm(files, desc=f"Preparing {args.scene}"):
        if saved >= args.limit:
            break
        try:
            y = load_audio(path, args.sr)
            if not is_waveform_ok(y, args.sr):
                rejected += 1
                continue
            clip = crop_or_pad(y, target_samples)
            out_name = f"{args.scene}_{saved + 1:03d}.wav"
            sf.write(output_scene_dir / out_name, clip, args.sr)
            saved += 1
        except Exception as exc:
            rejected += 1
            print(f"[WARN] rejected {path}: {exc}")

    print(f"Scene: {args.scene}")
    print(f"Saved: {saved}")
    print(f"Rejected by technical checks: {rejected}")
    print(f"Output: {output_scene_dir}")
    print("Next: manually audit every saved clip.")

if __name__ == "__main__":
    main()
