from __future__ import annotations

import argparse
from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.pipelines.batch_pipeline import run_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch acoustic feature extraction.")
    parser.add_argument("--audio_dir", required=True, help="Directory containing raw audio files.")
    parser.add_argument("--env_file", required=False, help="Optional CSV/Parquet environment data file.")
    parser.add_argument(
        "--save_spectrograms",
        action="store_true",
        help="Save Mel spectrogram PNGs for human inspection.",
    )
    parser.add_argument(
        "--disable_segment_embedding_export",
        action="store_true",
        help="Do not save segment-level VGGish embeddings as JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_df = run_batch(
        audio_dir=args.audio_dir,
        environment_file=args.env_file,
        save_spectrogram=args.save_spectrograms,
        save_segment_embeddings=not args.disable_segment_embedding_export,
    )
    success_count = int((result_df["status"] == "success").sum()) if "status" in result_df else 0
    print(f"Processed {len(result_df)} audio files. Success: {success_count}.")


if __name__ == "__main__":
    main()
