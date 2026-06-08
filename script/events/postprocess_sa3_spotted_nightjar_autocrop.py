#!/usr/bin/env python3
"""Post-process Spotted Nightjar SA3 samples with automatic time cropping.

The crop detector is intentionally based on target-band structure, not just
energy. A noisy low-frequency tail can be loud but spectrally broad; the target
call should remain comparatively concentrated inside the 550-950 Hz band.
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
        mask = signal.medfilt2d(mask, kernel_size=(3, 5))
        out = librosa.istft(mag * mask * phase, hop_length=hop_length, length=len(channel))
        cleaned.append(out)
    if audio.ndim == 1:
        return cleaned[0]
    return np.stack(cleaned, axis=1)


def apply_fade(audio: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    fade_len = int(sample_rate * fade_ms / 1000.0)
    if fade_len <= 1 or len(audio) < fade_len * 2:
        return audio
    out = audio.copy()
    ramp_in = np.linspace(0.0, 1.0, fade_len)
    ramp_out = np.linspace(1.0, 0.0, fade_len)
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


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size == 0:
        return values
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same")


def detect_crop_bounds(
    audio: np.ndarray,
    sample_rate: int,
    *,
    low_hz: float,
    high_hz: float,
    tail_buffer_s: float,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[int, int, dict[str, float]]:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    n_fft = 2048
    hop = 256
    stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop, center=True)
    mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    band_mag = mag[band, :]
    if band_mag.size == 0:
        return 0, min(len(audio), int(max_duration_s * sample_rate)), {"fallback": 1.0}

    energy = np.mean(np.square(band_mag), axis=0)
    peakiness = np.max(band_mag, axis=0) / (np.mean(band_mag, axis=0) + 1e-8)
    energy_s = moving_average(energy, 7)
    peak_s = moving_average(peakiness, 7)

    energy_floor = np.percentile(energy_s, 20)
    energy_thr = max(energy_floor * 1.6, np.percentile(energy_s, 45))
    peak_thr = max(1.4, np.percentile(peak_s, 45))
    active = (energy_s >= energy_thr) & (peak_s >= peak_thr)

    # Smooth out single-frame gaps and reject tiny islands.
    active_f = active.astype(np.float32)
    active_s = moving_average(active_f, 9) >= 0.35
    idx = np.flatnonzero(active_s)
    if idx.size == 0:
        end_s = min(max_duration_s, len(audio) / sample_rate)
        return 0, int(end_s * sample_rate), {
            "fallback": 1.0,
            "energy_threshold": float(energy_thr),
            "peakiness_threshold": float(peak_thr),
        }

    start_frame = int(idx[0])
    end_frame = int(idx[-1])
    start_s = max(0.0, (start_frame * hop / sample_rate) - 0.05)
    end_s = min(len(audio) / sample_rate, (end_frame * hop / sample_rate) + tail_buffer_s)

    if end_s - start_s < min_duration_s:
        end_s = min(len(audio) / sample_rate, start_s + min_duration_s)
    if end_s - start_s > max_duration_s:
        end_s = start_s + max_duration_s

    return int(start_s * sample_rate), int(end_s * sample_rate), {
        "fallback": 0.0,
        "energy_threshold": float(energy_thr),
        "peakiness_threshold": float(peak_thr),
        "detected_start_s": float(start_s),
        "detected_end_s": float(end_s),
    }


def process_one(wav_path: Path, input_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    audio, sample_rate = sf.read(wav_path, always_2d=False)
    audio = audio.astype(np.float32)
    processed = bandpass(audio, sample_rate, args.low_hz, args.high_hz)
    processed = soft_spectral_gate(processed, sample_rate, args.gate_strength)
    start, end, crop_meta = detect_crop_bounds(
        processed,
        sample_rate,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        tail_buffer_s=args.tail_buffer_s,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
    )
    if args.keep_start:
        start = 0
    if args.max_end_s is not None:
        end = min(end, int(args.max_end_s * sample_rate))
    if end <= start:
        end = min(len(processed), start + int(args.min_duration_s * sample_rate))
    cropped = processed[start:end]
    cropped = apply_fade(cropped, sample_rate, args.fade_ms)
    cropped = normalize_rms(cropped, args.target_rms)
    cropped = np.clip(cropped, -1.0, 1.0)

    rel = wav_path.parent.relative_to(input_dir)
    item_dir = out_dir / rel
    item_dir.mkdir(parents=True, exist_ok=True)
    out_wav = item_dir / "generated_event_s3a_autocrop.wav"
    out_png = item_dir / "generated_event_s3a_autocrop_spectrogram.png"
    metadata_path = item_dir / "generated_event_s3a_autocrop_metadata.json"
    sf.write(out_wav, cropped, sample_rate, subtype="PCM_16")
    render_mel(cropped, sample_rate, out_png, "SA3 LoRA S3a tight + auto time crop")

    row: dict[str, object] = {
        "input_audio_path": str(wav_path),
        "audio_path": str(out_wav),
        "spectrogram_path": str(out_png),
        "metadata_path": str(metadata_path),
        "sample_rate": sample_rate,
        "low_hz": args.low_hz,
        "high_hz": args.high_hz,
        "gate_strength": args.gate_strength,
        "target_rms": args.target_rms,
        "fade_ms": args.fade_ms,
        "tail_buffer_s": args.tail_buffer_s,
        "crop_start_s": round(start / sample_rate, 4),
        "crop_end_s": round(end / sample_rate, 4),
        "crop_duration_s": round((end - start) / sample_rate, 4),
        "manual_verdict": "",
        "notes": "",
    }
    row.update(crop_meta)
    metadata_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--low-hz", type=float, default=550.0)
    parser.add_argument("--high-hz", type=float, default=950.0)
    parser.add_argument("--gate-strength", type=float, default=0.30)
    parser.add_argument("--target-rms", type=float, default=0.035)
    parser.add_argument("--fade-ms", type=float, default=120.0)
    parser.add_argument("--tail-buffer-s", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=1.5)
    parser.add_argument("--max-duration-s", type=float, default=4.2)
    parser.add_argument("--keep-start", action="store_true")
    parser.add_argument("--max-end-s", type=float, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    wavs = sorted(input_dir.rglob("generated_event.wav"))
    if not wavs:
        raise SystemExit(f"No generated_event.wav files found under {input_dir}")

    rows = [process_one(wav, input_dir, out_dir, args) for wav in wavs]
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = out_dir / "sample_audit.csv"
    fieldnames = [
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
        "tail_buffer_s",
        "crop_start_s",
        "crop_end_s",
        "crop_duration_s",
        "fallback",
        "energy_threshold",
        "peakiness_threshold",
        "manual_verdict",
        "notes",
    ]
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    playlist = out_dir / "sample_audit_absolute.m3u"
    playlist.write_text("\n".join(str(Path(row["audio_path"]).resolve()) for row in rows) + "\n")
    print(f"Wrote {len(rows)} auto-cropped samples to {out_dir}")


if __name__ == "__main__":
    main()
