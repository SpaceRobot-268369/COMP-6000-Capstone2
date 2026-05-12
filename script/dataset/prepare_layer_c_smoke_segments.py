#!/usr/bin/env python3
"""Prepare downloaded Layer C smoke-test event segments.

For each downloaded ``*.webm`` event segment, this writes artifacts beside the
source file:

  - audio.wav              16 kHz mono PCM WAV for AudioGen-style training
  - caption.txt            caption from manifest.csv
  - mel_spectrogram.npy    float32 log-mel array for audit/precompute
  - mel_spectrogram.png    rendered mel-spectrogram preview

The script expects ``manifest.csv`` at the dataset root and matches segment
folders back to manifest rows using ``audio_event_id``.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "resources"
    / "site_257_bowra-dry-a"
    / "smoking_test_1_layer_C_dataset_1"
)
TARGET_SR = 16_000
SPECTROGRAM_DURATION_S = 0.0
SPECTROGRAM_TITLE = "Layer C - Event Segment Spectrogram"

sys.path.insert(0, str(REPO_ROOT / "acoustic_ai"))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/capstone_matplotlib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Layer C smoke-test segment artifacts."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prepared-manifest", type=Path, default=None)
    parser.add_argument("--target-sr", type=int, default=TARGET_SR)
    parser.add_argument("--spectrogram-title", default=SPECTROGRAM_TITLE)
    parser.add_argument(
        "--spectrogram-duration-s",
        type=float,
        default=SPECTROGRAM_DURATION_S,
        help=(
            "Render duration in seconds. Use 0 to render the actual WAV duration "
            "without padding; this is the default for Layer C event clips."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_event_id = {
        (row.get("audio_event_id") or "").strip(): row
        for row in rows
        if (row.get("audio_event_id") or "").strip()
    }
    if not by_event_id:
        raise ValueError(f"manifest contains no audio_event_id values: {manifest_path}")
    return by_event_id


def event_id_from_webm(webm_path: Path) -> str:
    # Expected stem: site_257_item_<recording_id>_audioevent_<event_id>
    marker = "_audioevent_"
    stem = webm_path.stem
    if marker not in stem:
        return ""
    return stem.rsplit(marker, 1)[-1]


def convert_to_wav(webm_path: Path, wav_path: Path, target_sr: int, overwrite: bool) -> bool:
    if wav_path.exists() and not overwrite:
        return True
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(webm_path),
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ffmpeg error] {webm_path}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return wav_path.exists() and wav_path.stat().st_size > 0


def write_caption(row: dict[str, str], caption_path: Path, overwrite: bool) -> None:
    if caption_path.exists() and not overwrite:
        return
    caption = (row.get("caption") or "").strip()
    if not caption:
        common = (row.get("species_common_name") or "bird").strip()
        diel = (row.get("diel_bin") or "unknown time").strip()
        caption = f"{common} bird vocal event, Bowra dry woodland, {diel}"
    caption_path.write_text(caption + "\n", encoding="utf-8")


def write_mel_spectrogram(
    wav_path: Path,
    npy_path: Path,
    png_path: Path,
    duration_s: float,
    title: str,
    overwrite: bool,
) -> None:
    if npy_path.exists() and png_path.exists() and not overwrite:
        return

    from modules.ambient.diffusion.layer_a_visualization import waveform_to_layer_a_mel_db

    waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    render_duration_s = float(waveform.shape[0]) / float(sample_rate)
    target_samples = int(round(duration_s * sample_rate))
    if duration_s > 0 and target_samples > 0:
        if waveform.shape[0] > target_samples:
            waveform = waveform[:target_samples]
        elif waveform.shape[0] < target_samples:
            waveform = np.pad(waveform, (0, target_samples - waveform.shape[0]))
        render_duration_s = duration_s

    mel_db = waveform_to_layer_a_mel_db(waveform, int(sample_rate)).astype(np.float32)
    np.save(npy_path, mel_db)
    png_path.write_bytes(render_mel_png_bytes(mel_db, render_duration_s, title))


def render_mel_png_bytes(mel_db: np.ndarray, duration_s: float, title: str) -> bytes:
    """Render with the Layer A visual style but a Layer C-specific title."""
    import io

    from modules.ambient.preprocess import SPEC_CFG
    from modules.ambient.diffusion.layer_a_visualization import (
        LAYER_A_SPEC_CMAP,
        LAYER_A_SPEC_DPI,
        LAYER_A_SPEC_FIGSIZE,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=LAYER_A_SPEC_FIGSIZE)
    img = ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        extent=[0, duration_s, 0, SPEC_CFG["n_mels"]],
        cmap=LAYER_A_SPEC_CMAP,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_title(title)
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=LAYER_A_SPEC_DPI)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else REPO_ROOT / args.dataset_dir
    manifest_path = args.manifest or dataset_dir / "manifest.csv"
    prepared_manifest_path = args.prepared_manifest or dataset_dir / "prepared_manifest.csv"
    if not dataset_dir.exists():
        print(f"[error] dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1
    if not manifest_path.exists():
        print(f"[error] manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest_by_event = load_manifest(manifest_path)
    webm_paths = sorted(dataset_dir.glob("site_257_item_*/site_257_item_*_audioevent_*/*.webm"))
    if not webm_paths:
        print(f"[error] no downloaded webm segments found under {dataset_dir}", file=sys.stderr)
        return 1

    ok_count = 0
    skipped_not_in_manifest = 0
    errors: list[str] = []
    prepared_rows: list[dict[str, str]] = []

    for index, webm_path in enumerate(webm_paths, start=1):
        event_id = event_id_from_webm(webm_path)
        row = manifest_by_event.get(event_id)
        if row is None:
            skipped_not_in_manifest += 1
            continue

        segment_dir = webm_path.parent
        wav_path = segment_dir / "audio.wav"
        caption_path = segment_dir / "caption.txt"
        npy_path = segment_dir / "mel_spectrogram.npy"
        png_path = segment_dir / "mel_spectrogram.png"

        if not convert_to_wav(webm_path, wav_path, args.target_sr, args.overwrite):
            errors.append(f"{webm_path}: wav conversion failed")
            continue

        try:
            write_caption(row, caption_path, args.overwrite)
            write_mel_spectrogram(
                wav_path=wav_path,
                npy_path=npy_path,
                png_path=png_path,
                duration_s=args.spectrogram_duration_s,
                title=args.spectrogram_title,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            errors.append(f"{webm_path}: {exc}")
            continue

        prepared_row = dict(row)
        prepared_row.update(
            {
                "source_webm_path": str(webm_path.relative_to(REPO_ROOT)),
                "audio_path": str(wav_path.relative_to(REPO_ROOT)),
                "caption_path": str(caption_path.relative_to(REPO_ROOT)),
                "mel_spectrogram_npy_path": str(npy_path.relative_to(REPO_ROOT)),
                "mel_spectrogram_png_path": str(png_path.relative_to(REPO_ROOT)),
                "prepare_status": "ok",
            }
        )
        prepared_rows.append(prepared_row)
        ok_count += 1
        if ok_count <= 5 or ok_count % 25 == 0:
            print(f"[ok] {ok_count:03d} {webm_path.parent.name}")

    print(
        f"[done] prepared={ok_count} skipped_not_in_manifest={skipped_not_in_manifest} "
        f"errors={len(errors)} total_webm={len(webm_paths)}"
    )
    if errors:
        print("[errors]")
        for error in errors[:20]:
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return 1

    if prepared_rows:
        fieldnames: list[str] = []
        for row in prepared_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with prepared_manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prepared_rows)
        print(f"[manifest] {prepared_manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
