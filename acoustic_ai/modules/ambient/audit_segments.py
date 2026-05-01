"""Audit the cleaned ambient segment pool.

Pulls a stratified random sample from `data/ambient/ambient_index.csv`,
copies the WAV files into `debug/ambient_audit/` and renders one mel
spectrogram per segment. Use this to listen + look before trusting the
pool downstream.

A clean ambient segment should:
  - sound like continuous texture (no audible bird calls, vehicles,
    helicopters, voices, branch snaps, etc.)
  - look horizontally streaky in the mel — stationary energy across time,
    no vertical transient bands.

Usage (from project root):
  python3 acoustic_ai/modules/ambient/audit_segments.py
  python3 acoustic_ai/modules/ambient/audit_segments.py --n 30 --seed 0
  python3 acoustic_ai/modules/ambient/audit_segments.py --stratify
"""

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from modules.ambient.preprocess import SPEC_CFG, load_audio, waveform_to_melspec  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INDEX = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "ambient_index.csv"
DEFAULT_SEGMENTS = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "ambient_segments"
DEFAULT_OUT = PROJECT_ROOT / "debug" / "ambient_audit"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n", type=int, default=30, help="Number of segments to sample.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stratify", action="store_true",
                   help="Stratify sample across diel_bin × season cells.")
    return p.parse_args()


def load_index(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["diel_bin"], r["season"])].append(r)

    picks: list[dict] = []
    cell_keys = list(cells.keys())
    rng.shuffle(cell_keys)
    while len(picks) < n and cell_keys:
        for k in list(cell_keys):
            if cells[k]:
                picks.append(cells[k].pop(rng.randrange(len(cells[k]))))
                if len(picks) >= n:
                    break
            else:
                cell_keys.remove(k)
    return picks


def render_spectrogram(wav_path: Path, png_path: Path, title: str) -> None:
    wave = load_audio(str(wav_path))
    mel_db = waveform_to_melspec(wave)

    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(
        mel_db, aspect="auto", origin="lower",
        extent=[0, wave.size / SPEC_CFG["sample_rate"], 0, SPEC_CFG["n_mels"]],
        cmap="magma",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_title(title, fontsize=9)
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.index.exists():
        print(f"index not found: {args.index}", file=sys.stderr)
        print("run precompute/build_ambient_index.py first.", file=sys.stderr)
        return 1

    rows = load_index(args.index)
    if not rows:
        print("index is empty — cleaning produced no segments.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    picks = stratified_sample(rows, args.n, rng) if args.stratify else rng.sample(rows, min(args.n, len(rows)))

    args.out.mkdir(parents=True, exist_ok=True)

    summary_path = args.out / "audit_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", "diel_bin", "season", "duration_s", "wav", "spectrogram"])
        for row in picks:
            seg_id = row["segment_id"]
            src_wav = args.segments / f"{seg_id}.wav"
            if not src_wav.exists():
                print(f"missing: {src_wav}", file=sys.stderr)
                continue
            dst_wav = args.out / f"{seg_id}.wav"
            dst_png = args.out / f"{seg_id}.png"
            shutil.copy2(src_wav, dst_wav)
            title = f"{seg_id}  |  {row['diel_bin']}  {row['season']}  |  {row['duration_s']}s"
            render_spectrogram(dst_wav, dst_png, title)
            writer.writerow([seg_id, row["diel_bin"], row["season"], row["duration_s"], dst_wav.name, dst_png.name])

    print(f"wrote {len(picks)} segments + spectrograms to {args.out}")
    print(f"summary: {summary_path}")
    print("\nlisten to each .wav and view the .png. pass criterion: ≥ 27/30 sound like ambience.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
