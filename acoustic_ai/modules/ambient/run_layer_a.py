"""Layer A — Ambient bed generation CLI.

Given environmental parameters, retrieve and blend an ambient bed, then produce
the scrutinization triplet: WAV, mel-spectrogram PNG, and JSON explanation.

Usage (from project root):
  python3 acoustic_ai/modules/ambient/run_layer_a.py \\
    --diel-bin dawn --season spring --hour 6 --month 3 --duration 120

  python3 acoustic_ai/modules/ambient/run_layer_a.py \\
    --diel-bin afternoon --season summer --hour 14 --month 1 --out-dir debug/test
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from retrieval import AmbientRetriever, LayerResult
from preprocess import SPEC_CFG, waveform_to_melspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INDEX = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "ambient_index.csv"
DEFAULT_SEGMENTS = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "ambient_segments"
DEFAULT_OUT = PROJECT_ROOT / "debug" / "layer_a"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Layer A ambient bed.")
    p.add_argument("--diel-bin", required=True,
                   choices=["dawn", "morning", "afternoon", "night"])
    p.add_argument("--season", required=True,
                   choices=["spring", "summer", "autumn", "winter"])
    p.add_argument("--hour", type=int, required=True, help="Local hour (0–23)")
    p.add_argument("--month", type=int, required=True, help="Month (1–12)")
    p.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    p.add_argument("--k", type=int, default=5, help="Number of segments to blend")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    return p.parse_args()


def render_spectrogram(waveform: np.ndarray, png_path: Path) -> None:
    """Render and save a mel-spectrogram."""
    mel_db = waveform_to_melspec(waveform)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(
        mel_db, aspect="auto", origin="lower",
        extent=[0, waveform.size / SPEC_CFG["sample_rate"], 0, SPEC_CFG["n_mels"]],
        cmap="magma",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_title("Layer A — Ambient Bed Spectrogram")
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    # Validate inputs.
    if not (0 <= args.hour < 24):
        print(f"error: hour must be 0–23, got {args.hour}")
        return 1
    if not (1 <= args.month <= 12):
        print(f"error: month must be 1–12, got {args.month}")
        return 1

    # Create output directory.
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize retriever.
    try:
        retriever = AmbientRetriever(args.index, args.segments)
    except Exception as e:
        print(f"error loading segment pool: {e}")
        return 1

    # Retrieve and blend.
    print(f"retrieving ambient bed: "
          f"{args.diel_bin}/{args.season} hour={args.hour} month={args.month}")
    try:
        result: LayerResult = retriever.retrieve(
            diel_bin=args.diel_bin,
            season=args.season,
            hour=args.hour,
            month=args.month,
            k=args.k,
            target_duration_s=args.duration,
        )
    except Exception as e:
        print(f"error retrieving: {e}")
        return 1

    # Write scrutinization triplet.
    wav_path = args.out_dir / "layer_a_bed.wav"
    png_path = args.out_dir / "layer_a_spec.png"
    json_path = args.out_dir / "layer_a_explanation.json"

    sf.write(wav_path, result.audio, result.sample_rate, subtype="PCM_16")
    render_spectrogram(result.audio, png_path)

    with open(json_path, "w") as f:
        json.dump(result.metadata, f, indent=2)

    print(f"✓ written to {args.out_dir}/")
    print(f"  WAV:  {wav_path.name}")
    print(f"  PNG:  {png_path.name}")
    print(f"  JSON: {json_path.name}")
    print(f"\nmetadata:")
    print(json.dumps(result.metadata, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
