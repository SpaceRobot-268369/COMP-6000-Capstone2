#!/usr/bin/env python3
"""S3a-style post-process for SA3 cuckoo text-to-audio samples.

This is intentionally lightweight: no retraining, only output-side cleanup.
It applies a Butterworth bandpass, a soft spectral gate, short fades, and RMS
normalization, then writes an audit bundle mirroring the input sample layout.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal


def render_mel(audio: np.ndarray, sample_rate: int, path: Path, title: str) -> None:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    mel = librosa.feature.melspectrogram(
        y=mono,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=0,
        fmax=min(sample_rate / 2, 11025),
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=min(sample_rate / 2, 11025),
        cmap="magma",
        ax=ax,
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def bandpass(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = sample_rate / 2
    high_hz = min(high_hz, nyquist * 0.98)
    sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)


def soft_spectral_gate(
    audio: np.ndarray,
    sample_rate: int,
    strength: float,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Suppress stationary noise without hard-zeroing the target call."""
    if strength <= 0:
        return audio

    channels = [audio] if audio.ndim == 1 else [audio[:, i] for i in range(audio.shape[1])]
    cleaned = []
    for channel in channels:
        stft = librosa.stft(channel, n_fft=n_fft, hop_length=hop_length, center=True)
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))

        noise_floor = np.percentile(mag, 25, axis=1, keepdims=True)
        threshold = noise_floor * (1.0 + 2.0 * strength)
        ratio = mag / (threshold + 1e-8)
        mask = np.clip((ratio - 0.6) / 0.8, 0.0, 1.0)

        min_gain = max(0.12, 1.0 - strength)
        mask = min_gain + (1.0 - min_gain) * mask
        smoothed = signal.medfilt2d(mask, kernel_size=(3, 5))
        out = librosa.istft(mag * smoothed * phase, hop_length=hop_length, length=len(channel))
        cleaned.append(out)

    if audio.ndim == 1:
        return cleaned[0]
    return np.stack(cleaned, axis=1)


def apply_fade(audio: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    fade_len = int(sample_rate * fade_ms / 1000.0)
    if fade_len <= 1 or len(audio) < fade_len * 2:
        return audio
    ramp_in = np.linspace(0.0, 1.0, fade_len)
    ramp_out = np.linspace(1.0, 0.0, fade_len)
    out = audio.copy()
    if out.ndim == 1:
        out[:fade_len] *= ramp_in
        out[-fade_len:] *= ramp_out
    else:
        out[:fade_len, :] *= ramp_in[:, None]
        out[-fade_len:, :] *= ramp_out[:, None]
    return out


def normalize_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms <= 1e-8:
        return audio
    out = audio * (target_rms / rms)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    return out


def process_one(
    wav_path: Path,
    out_wav: Path,
    out_png: Path,
    *,
    low_hz: float,
    high_hz: float,
    gate_strength: float,
    target_rms: float,
    fade_ms: float,
    title: str,
) -> dict[str, object]:
    audio, sample_rate = sf.read(wav_path, always_2d=False)
    audio = audio.astype(np.float32)

    processed = bandpass(audio, sample_rate, low_hz, high_hz)
    processed = soft_spectral_gate(processed, sample_rate, gate_strength)
    processed = apply_fade(processed, sample_rate, fade_ms)
    processed = normalize_rms(processed, target_rms)
    processed = np.clip(processed, -1.0, 1.0)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, processed, sample_rate, subtype="PCM_16")
    render_mel(processed, sample_rate, out_png, title)
    return {
        "input_audio_path": str(wav_path),
        "audio_path": str(out_wav),
        "spectrogram_path": str(out_png),
        "sample_rate": sample_rate,
        "low_hz": low_hz,
        "high_hz": high_hz,
        "gate_strength": gate_strength,
        "target_rms": target_rms,
        "fade_ms": fade_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["gentle", "tight"], required=True)
    parser.add_argument("--low-hz", type=float, default=None)
    parser.add_argument("--high-hz", type=float, default=None)
    parser.add_argument("--gate-strength", type=float, default=None)
    parser.add_argument("--target-rms", type=float, default=None)
    parser.add_argument("--fade-ms", type=float, default=None)
    args = parser.parse_args()

    presets = {
        "gentle": {
            "low_hz": 1800.0,
            "high_hz": 4800.0,
            "gate_strength": 0.25,
            "target_rms": 0.03,
            "fade_ms": 80.0,
        },
        "tight": {
            "low_hz": 2100.0,
            "high_hz": 4100.0,
            "gate_strength": 0.40,
            "target_rms": 0.03,
            "fade_ms": 100.0,
        },
    }
    preset = presets[args.mode]
    if args.low_hz is not None:
        preset["low_hz"] = args.low_hz
    if args.high_hz is not None:
        preset["high_hz"] = args.high_hz
    if args.gate_strength is not None:
        preset["gate_strength"] = args.gate_strength
    if args.target_rms is not None:
        preset["target_rms"] = args.target_rms
    if args.fade_ms is not None:
        preset["fade_ms"] = args.fade_ms
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    wavs = sorted(input_dir.rglob("generated_event.wav"))
    if not wavs:
        raise SystemExit(f"No generated_event.wav files found under {input_dir}")

    rows = []
    for wav_path in wavs:
        rel = wav_path.parent.relative_to(input_dir)
        item_dir = out_dir / rel
        row = process_one(
            wav_path,
            item_dir / "generated_event_s3a.wav",
            item_dir / "generated_event_s3a_spectrogram.png",
            title=f"SA3 LoRA S3a {args.mode} post-process",
            **preset,
        )
        metadata_path = item_dir / "generated_event_s3a_metadata.json"
        row["metadata_path"] = str(metadata_path)
        row["mode"] = args.mode
        row["manual_verdict"] = ""
        row["notes"] = ""
        metadata_path.write_text(json.dumps(row, indent=2))
        rows.append(row)

    audit_csv = out_dir / "sample_audit.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "input_audio_path",
        "audio_path",
        "spectrogram_path",
        "metadata_path",
        "sample_rate",
        "low_hz",
        "high_hz",
        "gate_strength",
        "target_rms",
        "fade_ms",
        "manual_verdict",
        "notes",
    ]
    with audit_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    playlist = out_dir / "sample_audit_absolute.m3u"
    playlist.write_text("\n".join(str(Path(row["audio_path"]).resolve()) for row in rows) + "\n")
    print(f"Wrote {len(rows)} S3a {args.mode} samples to {out_dir}")


if __name__ == "__main__":
    main()
