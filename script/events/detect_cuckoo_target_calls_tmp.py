#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Detection:
    start_s: float
    end_s: float
    score: float
    ridge_score: float
    contrast_score: float


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _smooth_1d(x: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1:
        return x
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(x, kernel, mode="same")


def detect_calls(
    audio: np.ndarray,
    sr: int,
    *,
    low_hz: float = 2100.0,
    high_hz: float = 4100.0,
    min_window_s: float = 0.36,
    max_window_s: float = 0.74,
    hop_s: float = 0.035,
    max_detections: int = 3,
    min_score: float = 0.68,
    min_contrast_score: float = 0.70,
    min_start_hz: float = 3450.0,
) -> list[Detection]:
    y = _mono(audio).astype(np.float32)
    n_fft = 2048
    hop_length = 256
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length)

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    band = mag[band_mask]
    band_freqs = freqs[band_mask]
    if band.size == 0:
        return []

    db = librosa.amplitude_to_db(band, ref=np.max, top_db=80)
    # Normalize so 1.0 is bright target energy and 0.0 is floor.
    norm = np.clip((db + 80.0) / 80.0, 0.0, 1.0)
    band_energy = _smooth_1d(norm.mean(axis=0), 7)

    duration_s = len(y) / sr
    candidates: list[Detection] = []
    for win_s in np.arange(min_window_s, max_window_s + 1e-6, 0.07):
        n_frames = max(3, int(round(win_s / (hop_length / sr))))
        for start_s in np.arange(0.0, max(0.0, duration_s - win_s), hop_s):
            start_frame = int(np.searchsorted(times, start_s))
            end_frame = min(start_frame + n_frames, norm.shape[1])
            if end_frame - start_frame < 3:
                continue
            win = norm[:, start_frame:end_frame]
            local_energy = band_energy[start_frame:end_frame]
            if float(np.mean(local_energy)) < 0.18:
                continue

            ridge_idx = np.argmax(win, axis=0)
            ridge_hz = band_freqs[ridge_idx]
            ridge_hz = _smooth_1d(ridge_hz.astype(np.float32), 5)
            start_hz = float(np.median(ridge_hz[: max(1, len(ridge_hz) // 4)]))
            end_hz = float(np.median(ridge_hz[-max(1, len(ridge_hz) // 4) :]))
            drop_hz = start_hz - end_hz

            # Cuckoo motif should descend, but keep tolerance for noisy generated ridges.
            descend_score = np.clip((drop_hz - 350.0) / 900.0, 0.0, 1.0)
            target_start_score = np.exp(-((start_hz - 3700.0) / 900.0) ** 2)
            target_end_score = np.exp(-((end_hz - 2800.0) / 900.0) ** 2)
            ridge_brightness = float(np.mean(np.max(win, axis=0)))
            ridge_score = float(0.45 * descend_score + 0.25 * target_start_score + 0.20 * target_end_score + 0.10 * ridge_brightness)

            # Penalize flat horizontal bed: target windows should have a ridge brighter than average band fill.
            bright = np.percentile(win, 92, axis=0)
            fill = np.mean(win, axis=0)
            contrast = float(np.mean(bright - fill))
            contrast_score = float(np.clip(contrast / 0.28, 0.0, 1.0))

            score = float(0.70 * ridge_score + 0.30 * contrast_score)
            if score >= min_score and contrast_score >= min_contrast_score and start_hz >= min_start_hz:
                candidates.append(
                    Detection(
                        start_s=float(start_s),
                        end_s=float(start_s + win_s),
                        score=score,
                        ridge_score=ridge_score,
                        contrast_score=contrast_score,
                    )
                )

    candidates.sort(key=lambda d: d.score, reverse=True)
    selected: list[Detection] = []
    for cand in candidates:
        overlaps = False
        for prev in selected:
            inter = max(0.0, min(cand.end_s, prev.end_s) - max(cand.start_s, prev.start_s))
            union = max(cand.end_s, prev.end_s) - min(cand.start_s, prev.start_s)
            if union > 0 and inter / union > 0.25:
                overlaps = True
                break
        if overlaps:
            continue
        selected.append(cand)
        if len(selected) >= max_detections:
            break

    selected.sort(key=lambda d: d.start_s)
    return selected


def fade(audio: np.ndarray, sr: int, fade_ms: float = 70.0) -> np.ndarray:
    n = min(int(sr * fade_ms / 1000.0), len(audio) // 2)
    out = audio.copy()
    if n > 1:
        out[:n] *= np.linspace(0, 1, n, dtype=np.float32)
        out[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return out


def render_spectrogram(
    audio: np.ndarray,
    sr: int,
    path: Path,
    title: str,
    detections: list[Detection] | None = None,
) -> None:
    y = _mono(audio)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=0,
        fmax=min(sr / 2, 11025),
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=min(sr / 2, 11025),
        cmap="magma",
        ax=ax,
    )
    if detections:
        for det in detections:
            rect = patches.Rectangle(
                (det.start_s, 1700),
                det.end_s - det.start_s,
                3200,
                linewidth=2.5,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(det.start_s, 5150, f"{det.score:.2f}", color="red", fontsize=8)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def process_seed(
    seed_dir: Path,
    out_root: Path,
    *,
    pre_s: float,
    post_s: float,
    min_score: float,
    min_contrast_score: float,
) -> dict[str, object]:
    wav = seed_dir / "generated_event_s3a.wav"
    audio, sr = sf.read(wav, always_2d=False)
    audio = _mono(audio).astype(np.float32)
    detections = detect_calls(audio, sr, min_score=min_score, min_contrast_score=min_contrast_score)

    out_dir = out_root / seed_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline = np.zeros_like(audio)
    pieces = []
    piece_offsets = []
    cursor = 0
    exported = []
    for idx, det in enumerate(detections, start=1):
        start = max(0, int((det.start_s - pre_s) * sr))
        end = min(len(audio), int((det.end_s + post_s) * sr))
        piece = fade(audio[start:end], sr)
        timeline[start:end] += piece
        pieces.append(piece)
        piece_offsets.append(cursor / sr)
        cursor += len(piece) + int(0.14 * sr)
        sf.write(out_dir / f"call_{idx:02d}.wav", piece, sr, subtype="PCM_16")
        exported.append(
            {
                "index": idx,
                "start_s": round(start / sr, 3),
                "end_s": round(end / sr, 3),
                "duration_s": round((end - start) / sr, 3),
                "score": round(det.score, 4),
                "ridge_score": round(det.ridge_score, 4),
                "contrast_score": round(det.contrast_score, 4),
                "audio_path": f"call_{idx:02d}.wav",
            }
        )

    if pieces:
        silence = np.zeros(int(0.14 * sr), dtype=np.float32)
        concat = np.concatenate([p for piece in pieces for p in (piece, silence)])[:- len(silence)]
    else:
        concat = np.zeros(int(0.5 * sr), dtype=np.float32)

    timeline = np.clip(timeline, -1.0, 1.0)
    concat = np.clip(concat, -1.0, 1.0)
    sf.write(out_dir / "target_timeline.wav", timeline, sr, subtype="PCM_16")
    sf.write(out_dir / "target_concat.wav", concat, sr, subtype="PCM_16")
    render_spectrogram(audio, sr, out_dir / "target_detection.png", "Cuckoo target-call detection", detections)
    render_spectrogram(timeline, sr, out_dir / "target_timeline_spectrogram.png", "Cuckoo target calls in original timeline")
    render_spectrogram(concat, sr, out_dir / "target_concat_spectrogram.png", "Cuckoo target calls concatenated")

    metadata = {
        "seed": seed_dir.name,
        "input_audio_path": str(wav.relative_to(ROOT)),
        "sample_rate": sr,
        "method": "cuckoo_descending_ridge_detector_v1",
        "frequency_band_hz": [2100, 4100],
        "pre_buffer_s": pre_s,
        "post_buffer_s": post_s,
        "min_score": min_score,
        "min_contrast_score": min_contrast_score,
        "detections": exported,
        "outputs": {
            "target_timeline_wav": "target_timeline.wav",
            "target_concat_wav": "target_concat.wav",
            "target_detection_png": "target_detection.png",
            "target_timeline_png": "target_timeline_spectrogram.png",
            "target_concat_png": "target_concat_spectrogram.png",
        },
    }
    (out_dir / "target_call_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "seed": seed_dir.name,
        "num_detections": len(detections),
        "detections": exported,
        "out_dir": str(out_dir.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", default=["seed_0045", "seed_0047", "seed_0049", "seed_0061", "seed_0073"])
    parser.add_argument("--pre-s", type=float, default=0.14)
    parser.add_argument("--post-s", type=float, default=0.18)
    parser.add_argument("--min-score", type=float, default=0.68)
    parser.add_argument("--min-contrast-score", type=float, default=0.70)
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seeds = sorted(p.name for p in input_dir.glob("seed_*")) if args.seeds == ["all"] else args.seeds
    for seed in seeds:
        rows.append(
            process_seed(
                input_dir / seed,
                out_dir,
                pre_s=args.pre_s,
                post_s=args.post_s,
                min_score=args.min_score,
                min_contrast_score=args.min_contrast_score,
            )
        )
    (out_dir / "target_call_detection_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "target_concat_audit_absolute.m3u").write_text(
        "\n".join(str((out_dir / row["seed"] / "target_concat.wav").resolve()) for row in rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "target_timeline_audit_absolute.m3u").write_text(
        "\n".join(str((out_dir / row["seed"] / "target_timeline.wav").resolve()) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} seeds to {out_dir.relative_to(ROOT)}")
    for row in rows:
        print(row["seed"], row["num_detections"], row["detections"])


if __name__ == "__main__":
    main()
