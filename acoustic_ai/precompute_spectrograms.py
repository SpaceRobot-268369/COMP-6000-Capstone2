"""Pre-compute mel-spectrograms for all clips in the training manifest.

Reads every .wav file, computes a log-mel spectrogram using the parameters
defined in preprocess.SPEC_CFG, and saves the result as a .npy file alongside
the .wav (same folder, same stem, .npy extension).

The dataset.py loader will automatically use the .npy file if it exists,
falling back to computing from the .wav if not.

Usage (from project root):
  python3 acoustic_ai/precompute_spectrograms.py
  python3 acoustic_ai/precompute_spectrograms.py --workers 8
  python3 acoustic_ai/precompute_spectrograms.py --dry-run
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add acoustic_ai to path so preprocess can be imported in worker processes
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "site_257_training_manifest.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-compute mel-spectrograms for all clips.")
    p.add_argument("--manifest",  type=Path, default=MANIFEST_PATH)
    p.add_argument("--workers",   type=int,  default=8)
    p.add_argument("--overwrite", action="store_true", help="Recompute even if .npy exists.")
    p.add_argument("--dry-run",   action="store_true", help="Count files without processing.")
    return p.parse_args()


def process_clip(wav_path: str, overwrite: bool) -> tuple[bool, str]:
    """Compute and save spectrogram for one clip. Returns (success, message)."""
    wav  = Path(wav_path)
    npy  = wav.with_suffix(".npy")

    if npy.exists() and not overwrite:
        return True, f"[SKIP] {wav.name}"

    if not wav.exists():
        return False, f"[MISS] {wav.name} — .wav not found"

    try:
        from preprocess import load_audio, waveform_to_melspec, pad_or_crop, melspec_to_tensor
        waveform = load_audio(str(wav))
        log_mel  = waveform_to_melspec(waveform)
        tensor   = melspec_to_tensor(log_mel)
        tensor   = pad_or_crop(tensor)
        np.save(str(npy), tensor.numpy())
        return True, f"[OK]   {wav.name}"
    except Exception as exc:
        return False, f"[FAIL] {wav.name} — {exc}"


def main() -> None:
    args = parse_args()

    import csv
    rows = list(csv.DictReader(open(args.manifest)))
    print(f"Manifest rows : {len(rows)}")

    # Resolve .wav paths (clip_path in manifest points to .webm; .wav is alongside it)
    wav_paths = []
    missing   = []
    already   = []

    for row in rows:
        webm = PROJECT_ROOT / row["clip_path"]
        wav  = Path(str(webm) + ".wav")   # clip_001.webm.wav
        npy  = Path(str(webm) + ".npy")   # clip_001.webm.npy

        if not wav.exists():
            missing.append(str(wav))
            continue
        if npy.exists() and not args.overwrite:
            already.append(str(npy))
            continue
        wav_paths.append(str(wav))

    print(f"Already done  : {len(already)}")
    print(f"To process    : {len(wav_paths)}")
    print(f"Missing .wav  : {len(missing)}")

    if missing:
        print(f"\n  First missing: {missing[0]}")
        print(f"  → Run the webm→wav conversion first.")

    if args.dry_run or not wav_paths:
        print("\nDry run — no files written.")
        return

    # Process in parallel
    success = 0
    failure = 0
    failed  = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_clip, p, args.overwrite): p for p in wav_paths}
        with tqdm(total=len(wav_paths), unit="clip") as bar:
            for future in as_completed(futures):
                ok, msg = future.result()
                if ok:
                    success += 1
                else:
                    failure += 1
                    failed.append(msg)
                bar.update(1)

    print(f"\nDone — success: {success}  failed: {failure}")
    if failed:
        print("\nFailed clips:")
        for msg in failed[:20]:
            print(f"  {msg}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    # Final count
    total_npy = len(list(PROJECT_ROOT.rglob("downloaded_clips/**/*.npy")))
    print(f"\nTotal .npy files on disk: {total_npy}")


if __name__ == "__main__":
    main()
