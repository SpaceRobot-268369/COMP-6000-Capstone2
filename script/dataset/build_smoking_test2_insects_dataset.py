"""
build_smoking_test2_insects_dataset.py

Selects candidate cicada/insect-heavy ambient clips for Layer A smoke test 2.
This does not modify the existing spring-night smoke test dataset.

The source pool is the cleaned Layer A ambient segment index. By default this
script targets the summer/afternoon cell because project audit notes identify it
as the most likely cicada-rich condition. Candidates are scored for stable
high-frequency energy and low low-frequency energy, then exported as 16 kHz mono
WAV clips with captions and metadata for human audit before training.

Inputs:
    acoustic_ai/data/ambient/ambient_index.csv
    acoustic_ai/data/ambient/ambient_segments/*.wav
    resources/site_257_bowra-dry-a/site_257_env_data.csv

Outputs:
    resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/clips/<clip_id>/
        audio.wav
        caption.txt
        meta.json

    resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/manifest.csv

Usage:
    python3 script/dataset/build_smoking_test2_insects_dataset.py --dry-run
    python3 script/dataset/build_smoking_test2_insects_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "acoustic_ai"))

AMBIENT_INDEX = REPO_ROOT / "acoustic_ai/data/ambient/ambient_index.csv"
AMBIENT_SEGMENTS = REPO_ROOT / "acoustic_ai/data/ambient/ambient_segments"
ENV_DATA = REPO_ROOT / "resources/site_257_bowra-dry-a/site_257_env_data.csv"
TRAINING_MANIFEST = REPO_ROOT / "resources/site_257_bowra-dry-a/site_257_training_manifest.csv"
ANNOTATIONS_DIR = REPO_ROOT / "resources/site_257_bowra-dry-a/downloaded_annotations"
OUT_DIR = REPO_ROOT / "resources/site_257_bowra-dry-a/smoking_test2_insects_dataset"
MANIFEST_PATH = OUT_DIR / "manifest.csv"
TARGET_SR = 16000
SPECTROGRAM_DURATION_S = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--season", default="summer")
    parser.add_argument("--diel", default="afternoon")
    parser.add_argument(
        "--month",
        type=int,
        action="append",
        default=None,
        help="Optional month filter. Repeat for multiple months.",
    )
    parser.add_argument("--min-duration-s", type=float, default=10.0)
    parser.add_argument(
        "--max-wind-speed-ms",
        type=float,
        default=4.5,
        help="Exclude rows at or above this mean wind speed. Matches the existing 'strong wind' caption threshold.",
    )
    parser.add_argument(
        "--max-wind-max-ms",
        type=float,
        default=8.0,
        help="Exclude rows at or above this daily max wind speed.",
    )
    parser.add_argument("--max-per-date", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_env_map(path: Path) -> dict[str, dict]:
    env_map = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            env_map[row["recording_id"]] = row
    return env_map


def load_clip_time_map(path: Path) -> dict[str, tuple[float, float]]:
    clip_times = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            clip_times[row["clip_path"]] = (
                float(row["clip_start_seconds"]),
                float(row["clip_end_seconds"]),
            )
    return clip_times


def load_annotation_intervals(annotations_dir: Path) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in annotations_dir.glob("annotations_*.csv"):
        recording_id = path.stem.replace("annotations_", "")
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    start = float(row["event_start_seconds"])
                    end = float(row["event_end_seconds"])
                except (KeyError, TypeError, ValueError):
                    continue
                if end > start:
                    intervals[recording_id].append((start, end))

    for recording_id in intervals:
        intervals[recording_id].sort()
    return intervals


def rec_id_from_source_clip(source_clip: str) -> str:
    folder = source_clip.split("/")[-2]
    return folder.replace("site_257_item_", "")


def overlaps_annotation(
    recording_id: str,
    segment_start_s: float,
    segment_end_s: float,
    annotation_intervals: dict[str, list[tuple[float, float]]],
) -> bool:
    for event_start, event_end in annotation_intervals.get(recording_id, []):
        if event_start < segment_end_s and event_end > segment_start_s:
            return True
    return False


def is_strong_wind(env: dict, max_wind_speed_ms: float, max_wind_max_ms: float) -> bool:
    try:
        wind_speed = float(env.get("wind_speed_ms", "nan"))
    except (TypeError, ValueError):
        wind_speed = float("nan")
    try:
        wind_max = float(env.get("wind_max_ms", "nan"))
    except (TypeError, ValueError):
        wind_max = float("nan")

    return (
        np.isfinite(wind_speed)
        and wind_speed >= max_wind_speed_ms
    ) or (
        np.isfinite(wind_max)
        and wind_max >= max_wind_max_ms
    )


def load_candidates(
    args: argparse.Namespace,
    env_map: dict[str, dict],
    clip_time_map: dict[str, tuple[float, float]],
    annotation_intervals: dict[str, list[tuple[float, float]]],
) -> tuple[list[dict], dict[str, int]]:
    months = set(args.month or [])
    rows = []
    stats = {
        "condition_mismatch": 0,
        "too_short": 0,
        "month_mismatch": 0,
        "strong_wind": 0,
        "annotated_event_overlap": 0,
        "missing_segment_wav": 0,
        "missing_clip_time": 0,
        "kept": 0,
    }

    with open(AMBIENT_INDEX) as f:
        for row in csv.DictReader(f):
            if row["season"] != args.season or row["diel_bin"] != args.diel:
                stats["condition_mismatch"] += 1
                continue
            if float(row["duration_s"]) < args.min_duration_s:
                stats["too_short"] += 1
                continue

            rec_id = rec_id_from_source_clip(row["source_clip"])
            env = env_map.get(rec_id, {})
            if months and int(env.get("month", 0)) not in months:
                stats["month_mismatch"] += 1
                continue
            if is_strong_wind(env, args.max_wind_speed_ms, args.max_wind_max_ms):
                stats["strong_wind"] += 1
                continue

            clip_times = clip_time_map.get(row["source_clip"])
            if clip_times is None:
                stats["missing_clip_time"] += 1
                continue

            clip_start_s, _clip_end_s = clip_times
            segment_abs_start_s = clip_start_s + float(row["t_start"])
            segment_abs_end_s = clip_start_s + float(row["t_end"])
            if overlaps_annotation(rec_id, segment_abs_start_s, segment_abs_end_s, annotation_intervals):
                stats["annotated_event_overlap"] += 1
                continue

            segment_wav = AMBIENT_SEGMENTS / f"{row['segment_id']}.wav"
            if not segment_wav.exists():
                stats["missing_segment_wav"] += 1
                continue

            row["_rec_id"] = rec_id
            row["_env"] = env
            row["_date"] = env.get("sample_local_date", "unknown")
            row["_segment_wav"] = segment_wav
            row["_segment_abs_start_s"] = segment_abs_start_s
            row["_segment_abs_end_s"] = segment_abs_end_s
            rows.append(row)
            stats["kept"] += 1

    return rows, stats


def score_insect_texture(path: Path) -> dict:
    """Score stable high-frequency insect/cicada-like texture.

    This is intentionally a filter, not a classifier. Human audit remains the
    authority before any training run.
    """
    import librosa

    audio, sr = librosa.load(path, sr=None, mono=True)
    if audio.size == 0:
        return {"score": 0.0, "reason": "empty"}

    n_fft = 2048
    hop = 512
    spectrum = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total = np.maximum(spectrum.sum(axis=0), 1e-12)

    high_mask = (freqs >= 3000) & (freqs <= min(9500, sr / 2))
    low_mask = freqs <= 500
    mid_mask = (freqs >= 1000) & (freqs < 3000)

    high = spectrum[high_mask].sum(axis=0)
    low = spectrum[low_mask].sum(axis=0)
    mid = spectrum[mid_mask].sum(axis=0)

    high_ratio = float(np.mean(high / total))
    low_ratio = float(np.mean(low / total))
    mid_ratio = float(np.mean(mid / total))
    high_cv = float(np.std(high) / (np.mean(high) + 1e-12))
    rms = float(np.sqrt(np.mean(np.square(audio))))

    stable_high = max(0.0, 1.0 - min(high_cv, 2.0) / 2.0)
    low_penalty = max(0.0, 1.0 - min(low_ratio, 0.75) / 0.75)
    rms_weight = min(1.0, rms / 0.01)

    score = (
        high_ratio
        * (0.65 + 0.35 * stable_high)
        * (0.70 + 0.30 * low_penalty)
        * rms_weight
    ) + (0.05 * mid_ratio * rms_weight)

    return {
        "score": round(float(score), 6),
        "high_ratio": round(high_ratio, 6),
        "mid_ratio": round(mid_ratio, 6),
        "low_ratio": round(low_ratio, 6),
        "high_cv": round(high_cv, 6),
        "rms": round(rms, 8),
    }


def build_caption(env: dict, diel: str, season: str) -> str:
    parts = [
        f"{season} {diel}",
        "insect-rich ambient soundscape",
        "cicada and insect texture",
        "Bowra dry woodland, Australia",
    ]

    try:
        temp = float(env.get("temperature_c", ""))
        if temp >= 30:
            parts.append(f"hot ({temp:.0f}C)")
        elif temp >= 24:
            parts.append(f"warm ({temp:.0f}C)")
        else:
            parts.append(f"mild ({temp:.0f}C)")
    except (TypeError, ValueError):
        pass

    try:
        hum = float(env.get("humidity_pct", ""))
        if hum < 35:
            parts.append("dry air")
        elif hum > 75:
            parts.append("humid air")
    except (TypeError, ValueError):
        pass

    try:
        wind = float(env.get("wind_speed_ms", ""))
        if wind < 0.5:
            parts.append("still")
        elif wind < 2.0:
            parts.append("light breeze")
        elif wind < 4.5:
            parts.append("moderate wind")
    except (TypeError, ValueError):
        pass

    date_str = env.get("sample_local_date", "")
    if date_str:
        parts.append(f"recorded {date_str}")

    parts.extend(["no music", "no machinery"])
    return ", ".join(parts)


def rank_candidates(candidates: list[dict]) -> list[dict]:
    ranked = []
    for row in candidates:
        metrics = score_insect_texture(row["_segment_wav"])
        row["_score"] = metrics["score"]
        row["_metrics"] = metrics
        ranked.append(row)

    return sorted(ranked, key=lambda r: r["_score"], reverse=True)


def select_for_audit(ranked: list[dict], target: int, max_per_date: int) -> list[dict]:
    selected = []
    per_date = defaultdict(int)

    for row in ranked:
        if len(selected) >= target:
            break
        date = row["_date"]
        if per_date[date] >= max_per_date:
            continue
        selected.append(row)
        per_date[date] += 1

    if len(selected) < target:
        selected_ids = {row["segment_id"] for row in selected}
        for row in ranked:
            if len(selected) >= target:
                break
            if row["segment_id"] not in selected_ids:
                selected.append(row)

    return selected


def convert_to_training_wav(src: Path, dst: Path) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def write_mel_spectrogram(audio_path: Path, npy_path: Path, png_path: Path) -> None:
    import soundfile as sf

    from modules.ambient.diffusion.layer_a_visualization import (
        render_layer_a_mel_png_bytes,
        waveform_to_layer_a_mel_db,
    )

    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    target_samples = int(round(SPECTROGRAM_DURATION_S * sample_rate))
    if waveform.shape[0] > target_samples:
        waveform = waveform[:target_samples]
    elif waveform.shape[0] < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.shape[0]))

    mel_db = waveform_to_layer_a_mel_db(waveform, sample_rate).astype(np.float32)
    np.save(npy_path, mel_db)

    png_bytes = render_layer_a_mel_png_bytes(mel_db, SPECTROGRAM_DURATION_S)
    png_path.write_bytes(png_bytes)


def write_dataset(selected: list[dict], out_dir: Path, overwrite: bool) -> tuple[list[dict], list[str]]:
    clips_dir = out_dir / "clips"
    if overwrite and clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    errors = []

    for i, row in enumerate(selected, 1):
        clip_id = f"{i:03d}_{row['segment_id']}"
        clip_dir = clips_dir / clip_id
        if clip_dir.exists() and overwrite:
            shutil.rmtree(clip_dir)
        clip_dir.mkdir(exist_ok=True)

        caption = build_caption(row["_env"], row["diel_bin"], row["season"])
        audio_path = clip_dir / "audio.wav"
        if overwrite or not audio_path.exists():
            ok = convert_to_training_wav(row["_segment_wav"], audio_path)
        else:
            ok = True

        status = "ok" if ok else "ffmpeg_error"
        if not ok:
            errors.append(clip_id)
        else:
            try:
                mel_npy = clip_dir / "mel_spectrogram.npy"
                mel_png = clip_dir / "mel_spectrogram.png"
                if overwrite or not (mel_npy.exists() and mel_png.exists()):
                    write_mel_spectrogram(audio_path, mel_npy, mel_png)
            except Exception as exc:
                status = "spectrogram_error"
                errors.append(clip_id)
                print(f"  [spectrogram error] {clip_id}: {exc}")

        meta = {
            "clip_id": clip_id,
            "segment_id": row["segment_id"],
            "source_clip": row["source_clip"],
            "source_segment_wav": str(row["_segment_wav"].relative_to(REPO_ROOT)),
            "t_start": float(row["t_start"]),
            "t_end": float(row["t_end"]),
            "duration_s": float(row["duration_s"]),
            "segment_abs_start_s": row["_segment_abs_start_s"],
            "segment_abs_end_s": row["_segment_abs_end_s"],
            "diel_bin": row["diel_bin"],
            "season": row["season"],
            "recording_date": row["_date"],
            "caption": caption,
            "insect_filter_metrics": row["_metrics"],
            "filter_rules": {
                "excluded_annotated_event_overlap": True,
                "max_wind_speed_ms": "strictly below 4.5 by default",
                "max_wind_max_ms": "strictly below 8.0 by default",
            },
            "env": {k: v for k, v in row["_env"].items() if not k.startswith("_")},
        }

        (clip_dir / "caption.txt").write_text(caption + "\n")
        (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        manifest_rows.append({
            "clip_id": clip_id,
            "segment_id": row["segment_id"],
            "recording_date": row["_date"],
            "diel_bin": row["diel_bin"],
            "season": row["season"],
            "duration_s": row["duration_s"],
            "audio_path": str(audio_path.relative_to(REPO_ROOT)),
            "caption": caption,
            "insect_score": row["_metrics"]["score"],
            "high_ratio": row["_metrics"]["high_ratio"],
            "low_ratio": row["_metrics"]["low_ratio"],
            "high_cv": row["_metrics"]["high_cv"],
            "rms": row["_metrics"]["rms"],
            "status": status,
        })

        mark = "ok" if status == "ok" else "fail"
        print(f"  {mark:4s} {i:02d}/{len(selected)} {clip_id} score={row['_score']:.4f}")

    return manifest_rows, errors


def write_manifest(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    env_map = load_env_map(ENV_DATA)
    clip_time_map = load_clip_time_map(TRAINING_MANIFEST)
    annotation_intervals = load_annotation_intervals(ANNOTATIONS_DIR)
    candidates, filter_stats = load_candidates(args, env_map, clip_time_map, annotation_intervals)

    if not candidates:
        sys.exit(
            f"No candidates found for season={args.season} "
            f"diel={args.diel} months={args.month or 'all'}"
        )

    print(f"Candidates after condition filter: {len(candidates)}")
    print("Filter stats:")
    for key, value in filter_stats.items():
        if value:
            print(f"  {key}: {value}")
    ranked = rank_candidates(candidates)
    selected = select_for_audit(ranked, args.target, args.max_per_date)
    print(f"Selected for audit: {len(selected)}")
    print(f"Unique recording dates: {len(set(row['_date'] for row in selected))}")

    print("\nTop selected candidates:")
    for i, row in enumerate(selected[: min(20, len(selected))], 1):
        metrics = row["_metrics"]
        print(
            f"{i:02d} {row['segment_id']} date={row['_date']} "
            f"score={metrics['score']:.4f} high={metrics['high_ratio']:.3f} "
            f"low={metrics['low_ratio']:.3f} cv={metrics['high_cv']:.3f} "
            f"rms={metrics['rms']:.5f}"
        )

    if args.dry_run:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows, errors = write_dataset(selected, OUT_DIR, args.overwrite)
    write_manifest(manifest_rows, MANIFEST_PATH)

    print(f"\nDone. {len(manifest_rows) - len(errors)}/{len(manifest_rows)} clips exported.")
    print(f"Manifest: {MANIFEST_PATH}")
    if errors:
        print(f"Errors ({len(errors)}): {errors}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
