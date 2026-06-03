#!/usr/bin/env python3
"""Build a Layer C bandpass review package from extracted event segments.

This keeps the broad event crop style: original extracted segment, full
bandpass, and a moderate target-band energy crop. It does not perform the
short clean-call/motif crop that was removed from the workflow.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


BANDS = {
    "chestnut_rumped_thornbill": (3500, 9500),
    "southern_boobook": (480, 800),
    "crested_bellbird": (700, 4000),
    "white_browed_woodswallow": (1200, 8000),
    "red_capped_robin": (1800, 8500),
    "superb_fairywren": (2800, 10000),
}


def bandpass_audio(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = sr / 2
    low = max(20.0, min(low_hz, nyq - 100.0)) / nyq
    high = max(low + 0.01, min(high_hz, nyq - 50.0)) / nyq
    sos = signal.butter(6, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs <= 1e-8:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sr)


def render_mel(path: Path, audio: np.ndarray, sr: int, title: str, high_hz: int | None = None) -> None:
    fmax = min(sr / 2, max(high_hz or sr / 2, 11000))
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=256,
        n_mels=128,
        fmin=0,
        fmax=fmax,
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
        fmax=fmax,
        ax=ax,
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def moderate_energy_crop(audio: np.ndarray, sr: int, event_start: float, event_end: float, max_duration: float) -> tuple[np.ndarray, float, float]:
    duration = len(audio) / sr
    # Preserve the annotated event plus a modest tail/head. This is deliberately
    # not a motif crop; it just avoids reviewing the entire +/-3 s buffer.
    start = max(0.0, event_start - 0.5)
    end = min(duration, event_end + 0.8)
    if end - start > max_duration:
        center = (event_start + event_end) / 2.0
        start = max(0.0, center - max_duration / 2.0)
        end = min(duration, start + max_duration)
    start_i = int(round(start * sr))
    end_i = int(round(end * sr))
    return audio[start_i:end_i], start, end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    rows = []
    per_species_count: dict[str, int] = {}
    for _, row in manifest.iterrows():
        event_type = str(row["event_type"])
        if event_type not in BANDS:
            continue
        segment_path = Path(str(row["segment_path"]))
        if not segment_path.is_absolute():
            segment_path = Path.cwd() / segment_path
        if not segment_path.exists() or segment_path.stat().st_size == 0:
            continue

        per_species_count[event_type] = per_species_count.get(event_type, 0) + 1
        audit_index = per_species_count[event_type]
        audio_event_id = int(row["audio_event_id"])
        item_dir = out_base / event_type / "review_items" / f"audit_{audit_index:03d}_audioevent_{audio_event_id}"
        item_dir.mkdir(parents=True, exist_ok=True)

        audio, sr = librosa.load(segment_path, sr=None, mono=True)
        audio = normalize(audio.astype(np.float32))
        low, high = BANDS[event_type]
        band = normalize(bandpass_audio(audio, sr, low, high))

        extracted_start = float(row["extracted_start_seconds"])
        event_start = max(0.0, float(row["event_start_seconds"]) - extracted_start)
        event_end = max(event_start + 0.1, float(row["event_end_seconds"]) - extracted_start)
        max_duration = 6.0 if event_type == "southern_boobook" else 5.0
        tight, crop_start, crop_end = moderate_energy_crop(band, sr, event_start, event_end, max_duration)
        tight = normalize(tight)

        original_webm = item_dir / "original.webm"
        original_full_wav = item_dir / "original_full.wav"
        original_full_mel = item_dir / "original_full_mel.png"
        band_full_wav = item_dir / f"bandpass_full_{low}_{high}hz.wav"
        band_full_mel = item_dir / f"bandpass_full_{low}_{high}hz_mel.png"
        band_tight_wav = item_dir / f"bandpass_tightcrop_{low}_{high}hz.wav"
        band_tight_mel = item_dir / f"bandpass_tightcrop_{low}_{high}hz_mel.png"

        shutil.copy2(segment_path, original_webm)
        write_wav(original_full_wav, audio, sr)
        write_wav(band_full_wav, band, sr)
        write_wav(band_tight_wav, tight, sr)
        title = f"{row['species_common_name']} audit {audit_index:03d} event {audio_event_id} score {float(row['score']):.3f}"
        render_mel(original_full_mel, audio, sr, f"{title} original full")
        render_mel(band_full_mel, band, sr, f"{title} full bandpass {low}-{high} Hz", high)
        render_mel(band_tight_mel, tight, sr, f"{title} bandpass crop {low}-{high} Hz", high)

        out = row.to_dict()
        out.update(
            {
                "audit_index": audit_index,
                "review_item_dir": str(item_dir),
                "original_webm": str(original_webm),
                "original_full_wav": str(original_full_wav),
                "original_full_mel_png": str(original_full_mel),
                "bandpass_low_hz": low,
                "bandpass_high_hz": high,
                "bandpass_full_wav": str(band_full_wav),
                "bandpass_full_mel_png": str(band_full_mel),
                "bandpass_tightcrop_wav": str(band_tight_wav),
                "bandpass_tightcrop_mel_png": str(band_tight_mel),
                "tightcrop_start_seconds_in_extracted": round(crop_start, 3),
                "tightcrop_end_seconds_in_extracted": round(crop_end, 3),
                "tightcrop_duration_seconds": round(crop_end - crop_start, 3),
                "crop_threshold_db": "",
                "manual_verdict": "",
                "manual_notes": "",
            }
        )
        (item_dir / "metadata.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        with (item_dir / "play_this_item.m3u").open("w", encoding="utf-8") as f:
            f.write(str(band_tight_wav.resolve()) + "\n")
            f.write(str(band_full_wav.resolve()) + "\n")
            f.write(str(original_full_wav.resolve()) + "\n")
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_base / "manual_audit_all_6species_bandpass_package.csv", index=False)
    for event_type, group in out_df.groupby("event_type", sort=False):
        species_dir = out_base / event_type
        group.to_csv(species_dir / f"manual_audit_{event_type}_bandpass_package.csv", index=False)
        with (species_dir / f"manual_audit_{event_type}_bandpass_tightcrop_absolute.m3u").open("w", encoding="utf-8") as f:
            for path in group["bandpass_tightcrop_wav"]:
                f.write(str(Path(path).resolve()) + "\n")

    print(f"wrote {out_base}")
    if len(out_df):
        print(out_df.groupby("event_type").size().to_string())


if __name__ == "__main__":
    main()
