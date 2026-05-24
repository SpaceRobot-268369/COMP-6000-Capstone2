#!/usr/bin/env python3
"""Automatic sanity checks for Layer C generated event samples.

This script is intentionally lightweight and offline-only. It does not try to
replace manual species audit; it catches obvious audio-quality failures such as
silence, clipping, very low foreground energy, and odd spectral balance.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


EPS = 1e-12


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, EPS))


def band_ratio(power_spectrum: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    total = float(np.sum(power_spectrum))
    if total <= EPS:
        return 0.0
    return float(np.sum(power_spectrum[mask]) / total)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    return mono, sample_rate


def evaluate_audio(path: Path) -> dict[str, str | float]:
    audio, sample_rate = load_mono(path)
    duration_s = len(audio) / sample_rate if sample_rate else 0.0

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    clip_ratio = float(np.mean(np.abs(audio) >= 0.999)) if len(audio) else 0.0

    frame_length = min(2048, max(256, int(0.046 * sample_rate)))
    hop_length = max(128, frame_length // 4)
    frame_rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    frame_db = librosa.amplitude_to_db(frame_rms, ref=1.0, amin=EPS)
    active_frames = frame_db > -45.0
    active_ratio = float(np.mean(active_frames)) if len(active_frames) else 0.0
    active_duration_s = active_ratio * duration_s

    n_fft = 2048
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, center=True))
    power = np.square(stft)
    mean_power = np.mean(power, axis=1) if power.size else np.zeros(n_fft // 2 + 1)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    centroid = librosa.feature.spectral_centroid(S=stft, sr=sample_rate)[0] if stft.size else np.array([0.0])
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sample_rate)[0] if stft.size else np.array([0.0])
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sample_rate, roll_percent=0.85)[0] if stft.size else np.array([0.0])
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0] if len(audio) else np.array([0.0])

    low_ratio = band_ratio(mean_power, freqs, 0.0, 1000.0)
    bird_band_ratio = band_ratio(mean_power, freqs, 1000.0, 8000.0)
    high_ratio = band_ratio(mean_power, freqs, 8000.0, sample_rate / 2.0)

    flags: list[str] = []
    if duration_s < 1.0:
        flags.append("too_short")
    if rms < 10 ** (-45.0 / 20.0):
        flags.append("too_quiet")
    if active_duration_s < 0.2 or active_ratio < 0.05:
        flags.append("near_silent")
    if peak > 10 ** (-0.1 / 20.0) or clip_ratio > 0.001:
        flags.append("possible_clipping")
    if low_ratio > 0.80:
        flags.append("mostly_low_freq")
    if bird_band_ratio < 0.15:
        flags.append("low_1_8khz_energy")

    return {
        "duration_s": round(duration_s, 3),
        "sample_rate": sample_rate,
        "rms_dbfs": round(dbfs(rms), 2),
        "peak_dbfs": round(dbfs(peak), 2),
        "clipping_ratio": round(clip_ratio, 6),
        "active_ratio_db_gt_minus45": round(active_ratio, 3),
        "active_duration_s": round(active_duration_s, 3),
        "spectral_centroid_hz": round(float(np.mean(centroid)), 1),
        "spectral_bandwidth_hz": round(float(np.mean(bandwidth)), 1),
        "spectral_rolloff85_hz": round(float(np.mean(rolloff)), 1),
        "zero_crossing_rate": round(float(np.mean(zcr)), 5),
        "band_0_1khz_ratio": round(low_ratio, 4),
        "band_1_8khz_ratio": round(bird_band_ratio, 4),
        "band_8khz_plus_ratio": round(high_ratio, 4),
        "auto_flags": ";".join(flags),
        "auto_verdict": "review" if flags else "pass_auto",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument(
        "--audio_column",
        default="audio_path",
        help="Column containing the generated WAV path.",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input_csv.open()))
    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")

    output_rows = []
    for row in rows:
        audio_path = Path(row[args.audio_column])
        if not audio_path.exists():
            metrics = {
                "auto_flags": "missing_audio",
                "auto_verdict": "review",
            }
        else:
            metrics = evaluate_audio(audio_path)
        output_rows.append({**row, **metrics})

    fieldnames = list(output_rows[0].keys())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    verdict_counts: dict[str, int] = {}
    for row in output_rows:
        verdict = str(row.get("auto_verdict", ""))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    print(f"Wrote {len(output_rows)} rows to {args.output_csv}")
    print("Auto verdict counts:", verdict_counts)


if __name__ == "__main__":
    main()
