"""
build_mvp1_all_conditioned_dataset.py

Builds the Layer A MVP-1 dataset: all clean ambient segments from site 257,
fully conditioned via captions on (season, diel_bin, temperature, humidity,
wind, date).

Differs from `build_smoking_test2_insects_dataset.py`:
- No (season, diel) filter — keeps every cell.
- No insect-texture scoring — hygiene filters only (wind, precip, duration,
  event overlap, missing files).
- Caption is a uniform conditioned template, not insect-specific.
- Optional per-(season,diel) cap and per-date cap to balance the dataset.

Inputs (same as smoke_2):
    acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_index.csv
    acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_segments/*.wav
    resources/site_257_bowra-dry-a/site_257_env_data.csv
    resources/site_257_bowra-dry-a/site_257_training_manifest.csv
    resources/site_257_bowra-dry-a/downloaded_annotations/

Outputs:
    resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/clips/<clip_id>/
        audio.wav
        caption.txt
        meta.json
    resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest.csv

Usage:
    python3 script/dataset/build_mvp1_all_conditioned_dataset.py --dry-run
    python3 script/dataset/build_mvp1_all_conditioned_dataset.py --overwrite
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

REPO_ROOT = Path(__file__).resolve().parents[2]

AMBIENT_INDEX = REPO_ROOT / "acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_index.csv"
AMBIENT_SEGMENTS = REPO_ROOT / "acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_segments"
ENV_DATA = REPO_ROOT / "resources/site_257_bowra-dry-a/site_257_env_data.csv"
TRAINING_MANIFEST = REPO_ROOT / "resources/site_257_bowra-dry-a/site_257_training_manifest.csv"
ANNOTATIONS_DIR = REPO_ROOT / "resources/site_257_bowra-dry-a/downloaded_annotations"
OUT_DIR = REPO_ROOT / "resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset"
MANIFEST_PATH = OUT_DIR / "manifest.csv"

TARGET_SR = 16000
SPECTROGRAM_DURATION_S = 10.0


# ---------- args ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--min-duration-s", type=float, default=10.0)
    p.add_argument("--max-wind-speed-ms", type=float, default=4.5,
                   help="Exclude rows at or above this mean wind speed.")
    p.add_argument("--max-wind-max-ms", type=float, default=8.0,
                   help="Exclude rows at or above this daily max wind speed.")
    p.add_argument("--max-precipitation-mm", type=float, default=0.1,
                   help="Exclude rows at or above this hourly precipitation.")
    p.add_argument("--per-cell-cap", type=int, default=0,
                   help="Max clips per (season, diel) cell. 0 = unlimited.")
    p.add_argument("--max-per-date", type=int, default=0,
                   help="Max clips per recording_date. 0 = unlimited.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of selected clips held out as validation, "
                        "stratified per (season, diel) cell. 0 = no val split.")
    p.add_argument("--split-seed", type=int, default=42,
                   help="RNG seed for deterministic per-cell train/val shuffle.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--render-spectrograms", action="store_true",
                   help="Render mel spectrogram NPY+PNG per clip. Slow; off by default for MVP build.")
    p.add_argument("--no-content-filters", action="store_true",
                   help="Skip §6.1 per-clip audio-content filters (RMS / DC / crest / spectral flatness).")
    p.add_argument("--min-rms", type=float, default=0.0005)
    p.add_argument("--max-rms", type=float, default=0.3)
    p.add_argument("--max-dc-offset", type=float, default=0.05)
    p.add_argument("--max-crest-factor", type=float, default=30.0)
    p.add_argument("--max-low-band-flat-frac", type=float, default=0.60,
                   help="Drop if >FRAC of frames have flatness <0.05 in the 500-2000 Hz band (motor bleed).")
    p.add_argument("--audit-samples-per-cell", type=int, default=5,
                   help="Number of clips to copy into audit_samples/<season>_<diel>/. 0 disables.")
    p.add_argument("--audit-seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output dataset directory (relative to repo root or absolute). "
                        "Defaults to the MVP-1 dataset dir. Use a separate dir to keep "
                        "the MVP-1 dataset frozen for rollback.")
    return p.parse_args()


# ---------- loaders ----------

def load_env_map(path: Path) -> dict[str, dict]:
    with open(path) as f:
        return {row["recording_id"]: row for row in csv.DictReader(f)}


def load_clip_time_map(path: Path) -> dict[str, tuple[str, float, float]]:
    """clip_path -> (recording_id, clip_start_seconds, clip_end_seconds)."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["clip_path"]] = (
                row["recording_id"],
                float(row["clip_start_seconds"]),
                float(row["clip_end_seconds"]),
            )
    return out


def load_annotation_intervals(annotations_dir: Path) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in annotations_dir.glob("annotations_*.csv"):
        rec_id = path.stem.replace("annotations_", "")
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    s = float(row["event_start_seconds"])
                    e = float(row["event_end_seconds"])
                except (KeyError, TypeError, ValueError):
                    continue
                if e > s:
                    intervals[rec_id].append((s, e))
    for rid in intervals:
        intervals[rid].sort()
    return intervals


# ---------- predicates ----------

def is_strong_wind(env: dict, max_wind_speed_ms: float, max_wind_max_ms: float) -> bool:
    try:
        w = float(env.get("wind_speed_ms", ""))
        if w >= max_wind_speed_ms:
            return True
    except (TypeError, ValueError):
        pass
    try:
        wm = float(env.get("wind_max_ms", ""))
        if wm >= max_wind_max_ms:
            return True
    except (TypeError, ValueError):
        pass
    return False


def is_rainy(env: dict, max_precip_mm: float) -> bool:
    try:
        p = float(env.get("precipitation_mm", ""))
        return p >= max_precip_mm
    except (TypeError, ValueError):
        return False


def overlaps_annotation(rec_id: str, seg_start_s: float, seg_end_s: float,
                        intervals: dict[str, list[tuple[float, float]]]) -> bool:
    for a, b in intervals.get(rec_id, []):
        if a < seg_end_s and b > seg_start_s:
            return True
    return False


# ---------- §6.1 content stats ----------

def compute_content_stats(wav_path: Path) -> dict | None:
    """Cheap per-clip audio-content stats. None if read fails."""
    import numpy as np
    import soundfile as sf
    try:
        y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception:
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size == 0:
        return None
    rms = float(np.sqrt(np.mean(y * y) + 1e-12))
    peak = float(np.max(np.abs(y)))
    dc = float(np.mean(y))
    crest = peak / (rms + 1e-12)

    # Spectral flatness, 500-2000 Hz band, fraction of frames below 0.05.
    n_fft, hop = 1024, 512
    if y.shape[0] < n_fft + hop:
        flat_low_frac = 0.0
    else:
        usable = y.shape[0] - n_fft
        n_frames = 1 + usable // hop
        win = np.hanning(n_fft).astype(np.float32)
        idx = np.arange(n_fft)[None, :] + (np.arange(n_frames) * hop)[:, None]
        frames = y[idx] * win
        spec = np.abs(np.fft.rfft(frames, axis=1)) + 1e-12
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        band = (freqs >= 500.0) & (freqs <= 2000.0)
        if int(band.sum()) < 2:
            flat_low_frac = 0.0
        else:
            band_spec = spec[:, band]
            geo = np.exp(np.mean(np.log(band_spec), axis=1))
            arith = np.mean(band_spec, axis=1)
            flat = geo / (arith + 1e-12)
            flat_low_frac = float(np.mean(flat < 0.05))

    return {"rms": rms, "peak": peak, "dc_offset": abs(dc),
            "crest_factor": crest, "flat_low_band_frac": flat_low_frac}


def content_drop_reason(stats: dict, args) -> str | None:
    if stats["rms"] < args.min_rms:
        return "content_near_silent"
    if stats["rms"] > args.max_rms:
        return "content_clipping"
    if stats["dc_offset"] > args.max_dc_offset:
        return "content_dc_offset"
    if stats["crest_factor"] > args.max_crest_factor:
        return "content_high_crest"
    if stats["flat_low_band_frac"] > args.max_low_band_flat_frac:
        return "content_motor_tonal"
    return None


# ---------- candidate gather ----------

def load_candidates(args, env_map, clip_time_map, ann_intervals):
    stats = defaultdict(int)
    rows = []

    with open(AMBIENT_INDEX) as f:
        for row in csv.DictReader(f):
            if float(row["duration_s"]) < args.min_duration_s:
                stats["too_short"] += 1
                continue

            clip_info = clip_time_map.get(row["source_clip"])
            if clip_info is None:
                stats["missing_clip_time"] += 1
                continue
            rec_id, clip_start_s, _ = clip_info

            env = env_map.get(rec_id, {})
            if not env:
                stats["missing_env"] += 1
                continue

            if is_strong_wind(env, args.max_wind_speed_ms, args.max_wind_max_ms):
                stats["strong_wind"] += 1
                continue
            if is_rainy(env, args.max_precipitation_mm):
                stats["rainy"] += 1
                continue

            seg_abs_start = clip_start_s + float(row["t_start"])
            seg_abs_end = clip_start_s + float(row["t_end"])
            if overlaps_annotation(rec_id, seg_abs_start, seg_abs_end, ann_intervals):
                stats["annotated_event_overlap"] += 1
                continue

            seg_wav = AMBIENT_SEGMENTS / f"{row['segment_id']}.wav"
            if not seg_wav.exists():
                stats["missing_segment_wav"] += 1
                continue

            if not args.no_content_filters:
                cs = compute_content_stats(seg_wav)
                if cs is None:
                    stats["content_read_error"] += 1
                    continue
                reason = content_drop_reason(cs, args)
                if reason:
                    stats[reason] += 1
                    continue
                row["_content_stats"] = cs

            row["_rec_id"] = rec_id
            row["_env"] = env
            row["_date"] = env.get("sample_local_date", "unknown")
            row["_segment_wav"] = seg_wav
            row["_seg_abs_start"] = seg_abs_start
            row["_seg_abs_end"] = seg_abs_end
            rows.append(row)
            stats["kept"] += 1

    return rows, dict(stats)


# ---------- balance ----------

def balance(rows: list[dict], per_cell_cap: int, max_per_date: int) -> list[dict]:
    """Cap per (season, diel) cell and per date; deterministic stable order."""
    per_cell = defaultdict(int)
    per_date = defaultdict(int)
    selected = []

    rows_sorted = sorted(rows, key=lambda r: (r["season"], r["diel_bin"], r["_date"], r["segment_id"]))

    for r in rows_sorted:
        cell = (r["season"], r["diel_bin"])
        if per_cell_cap and per_cell[cell] >= per_cell_cap:
            continue
        if max_per_date and per_date[r["_date"]] >= max_per_date:
            continue
        selected.append(r)
        per_cell[cell] += 1
        per_date[r["_date"]] += 1

    return selected


# ---------- split ----------

def assign_split(rows: list[dict], val_fraction: float, seed: int) -> None:
    """Stratified per (season, diel) train/val split.

    Each cell gets `round(N_cell * val_fraction)` clips assigned to 'val'
    (minimum 1 for cells with >=2 clips). Selection within a cell is a
    deterministic shuffle seeded by (split_seed, season, diel_bin) so the
    same cell always produces the same val set across builds.

    Mutates rows in-place to add row['_split'] in {'train', 'val'}.
    """
    import random

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["season"], r["diel_bin"])].append(r)

    for cell, cell_rows in by_cell.items():
        if val_fraction <= 0 or len(cell_rows) < 2:
            for r in cell_rows:
                r["_split"] = "train"
            continue

        n_val = max(1, round(len(cell_rows) * val_fraction))
        n_val = min(n_val, len(cell_rows) - 1)  # always keep >=1 in train

        # Deterministic shuffle per cell — same dataset build gives same split.
        rng = random.Random(f"{seed}:{cell[0]}:{cell[1]}")
        shuffled = list(cell_rows)
        rng.shuffle(shuffled)

        val_ids = {id(r) for r in shuffled[:n_val]}
        for r in cell_rows:
            r["_split"] = "val" if id(r) in val_ids else "train"


# ---------- caption ----------

def temp_bucket(env: dict) -> str | None:
    try:
        t = float(env["temperature_c"])
    except (KeyError, TypeError, ValueError):
        return None
    if t < 15: label = "cold"
    elif t < 25: label = "mild"
    elif t < 32: label = "warm"
    elif t < 40: label = "hot"
    else: label = "very hot"
    return f"{label} ({t:.0f}C)"


def humidity_bucket(env: dict) -> str | None:
    try:
        h = float(env["humidity_pct"])
    except (KeyError, TypeError, ValueError):
        return None
    if h < 40: return "dry air"
    if h < 70: return "moderate humidity"
    return "humid air"


def wind_bucket(env: dict) -> str | None:
    try:
        w = float(env["wind_speed_ms"])
    except (KeyError, TypeError, ValueError):
        return None
    if w < 0.5: return "still"
    if w < 2.0: return "light breeze"
    return "moderate wind"


def build_caption(row: dict) -> str:
    env = row["_env"]
    parts = [
        f"{row['diel_bin']} {row['season']} ambient soundscape",
        "Bowra dry woodland, Australia",
    ]
    for fn in (temp_bucket, humidity_bucket, wind_bucket):
        v = fn(env)
        if v:
            parts.append(v)
    date_str = env.get("sample_local_date", "")
    if date_str:
        parts.append(f"recorded {date_str}")
    parts.extend(["no music", "no machinery"])
    return ", ".join(parts)


# ---------- write ----------

def convert_to_training_wav(src: Path, dst: Path) -> bool:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", str(TARGET_SR), "-ac", "1", "-sample_fmt", "s16", str(dst)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def write_dataset(selected: list[dict], out_dir: Path, overwrite: bool, render_spec: bool):
    clips_dir = out_dir / "clips"
    if overwrite and clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    if render_spec:
        sys.path.insert(0, str(REPO_ROOT / "acoustic_ai"))
        import numpy as np
        import soundfile as sf
        from modules.ambient.diffusion.layer_a_visualization import (
            render_layer_a_mel_png_bytes, waveform_to_layer_a_mel_db,
        )

    manifest_rows = []
    errors = []
    n = len(selected)
    for i, row in enumerate(selected, 1):
        clip_id = f"{i:04d}_{row['segment_id']}"
        row["_clip_id"] = clip_id
        clip_dir = clips_dir / clip_id
        clip_dir.mkdir(exist_ok=True)
        caption = build_caption(row)

        audio_path = clip_dir / "audio.wav"
        ok = True if (audio_path.exists() and not overwrite) else convert_to_training_wav(row["_segment_wav"], audio_path)
        status = "ok" if ok else "ffmpeg_error"
        if not ok:
            errors.append(clip_id)

        if ok and render_spec:
            try:
                waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                target_samples = int(round(SPECTROGRAM_DURATION_S * sr))
                if waveform.shape[0] > target_samples:
                    waveform = waveform[:target_samples]
                elif waveform.shape[0] < target_samples:
                    waveform = np.pad(waveform, (0, target_samples - waveform.shape[0]))
                mel_db = waveform_to_layer_a_mel_db(waveform, sr).astype(np.float32)
                np.save(clip_dir / "mel_spectrogram.npy", mel_db)
                (clip_dir / "mel_spectrogram.png").write_bytes(
                    render_layer_a_mel_png_bytes(mel_db, SPECTROGRAM_DURATION_S))
            except Exception as exc:
                status = "spectrogram_error"
                errors.append(clip_id)
                print(f"  [spectrogram error] {clip_id}: {exc}")

        meta = {
            "clip_id": clip_id,
            "segment_id": row["segment_id"],
            "source_clip": row["source_clip"],
            "source_segment_wav": str(row["_segment_wav"].relative_to(REPO_ROOT)),
            "recording_id": row["_rec_id"],
            "recording_date": row["_date"],
            "t_start": float(row["t_start"]),
            "t_end": float(row["t_end"]),
            "duration_s": float(row["duration_s"]),
            "segment_abs_start_s": row["_seg_abs_start"],
            "segment_abs_end_s": row["_seg_abs_end"],
            "diel_bin": row["diel_bin"],
            "season": row["season"],
            "split": row.get("_split", "train"),
            "caption": caption,
            "env": {k: v for k, v in row["_env"].items() if not k.startswith("_")},
        }
        (clip_dir / "caption.txt").write_text(caption + "\n")
        (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        manifest_rows.append({
            "clip_id": clip_id,
            "segment_id": row["segment_id"],
            "recording_id": row["_rec_id"],
            "recording_date": row["_date"],
            "diel_bin": row["diel_bin"],
            "season": row["season"],
            "split": row.get("_split", "train"),
            "duration_s": float(row["duration_s"]),
            "audio_path": str(audio_path.relative_to(REPO_ROOT)),
            "caption": caption,
            "temperature_c": row["_env"].get("temperature_c", ""),
            "humidity_pct": row["_env"].get("humidity_pct", ""),
            "wind_speed_ms": row["_env"].get("wind_speed_ms", ""),
            "status": status,
        })

        if i % 50 == 0 or i == n:
            print(f"  exported {i}/{n}")

    return manifest_rows, errors


def write_audit_samples(selected: list[dict], out_dir: Path, n_per_cell: int, seed: int) -> int:
    """Copy n_per_cell random audio.wav files per (season, diel) into audit_samples/."""
    import random
    if n_per_cell <= 0:
        return 0
    audit_dir = out_dir / "audit_samples"
    if audit_dir.exists():
        shutil.rmtree(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in selected:
        if r.get("_clip_id"):
            by_cell[(r["season"], r["diel_bin"])].append(r)

    copied = 0
    for (season, diel), rows in by_cell.items():
        rng = random.Random(f"{seed}:audit:{season}:{diel}")
        sample = rng.sample(rows, min(n_per_cell, len(rows)))
        cell_dir = audit_dir / f"{season}_{diel}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        for r in sample:
            src = clips_dir / r["_clip_id"] / "audio.wav"
            if src.exists():
                shutil.copy2(src, cell_dir / f"{r['_clip_id']}.wav")
                copied += 1
    return copied


def write_manifest(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------- main ----------

def main() -> int:
    args = parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    else:
        out_dir = OUT_DIR
    manifest_path = out_dir / "manifest.csv"

    env_map = load_env_map(ENV_DATA)
    clip_time_map = load_clip_time_map(TRAINING_MANIFEST)
    ann_intervals = load_annotation_intervals(ANNOTATIONS_DIR)

    candidates, filter_stats = load_candidates(args, env_map, clip_time_map, ann_intervals)

    print("Filter stats:")
    for k, v in filter_stats.items():
        print(f"  {k}: {v}")
    print(f"Clean candidates: {len(candidates)}")

    print("\nPer (season, diel) — clean before balancing:")
    cell_counts = defaultdict(int)
    for r in candidates:
        cell_counts[(r["season"], r["diel_bin"])] += 1
    for cell, n in sorted(cell_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cell[0]:8s} {cell[1]:10s} {n}")

    selected = balance(candidates, args.per_cell_cap, args.max_per_date)
    print(f"\nAfter balance (per_cell_cap={args.per_cell_cap}, max_per_date={args.max_per_date}): {len(selected)}")

    assign_split(selected, args.val_fraction, args.split_seed)
    n_train = sum(1 for r in selected if r["_split"] == "train")
    n_val = sum(1 for r in selected if r["_split"] == "val")
    print(f"Train/val split (val_fraction={args.val_fraction}, seed={args.split_seed}): "
          f"train={n_train}, val={n_val}")

    print("\nPer (season, diel) — split breakdown:")
    print(f"  {'cell':<22s} {'train':>6s} {'val':>5s}")
    cell_split = defaultdict(lambda: {"train": 0, "val": 0})
    for r in selected:
        cell_split[(r["season"], r["diel_bin"])][r["_split"]] += 1
    for cell in sorted(cell_split.keys(), key=lambda c: -(cell_split[c]["train"] + cell_split[c]["val"])):
        s = cell_split[cell]
        print(f"  {cell[0]:8s} {cell[1]:10s}    {s['train']:>4d}  {s['val']:>4d}")

    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows, errors = write_dataset(selected, out_dir, args.overwrite, args.render_spectrograms)
    write_manifest(manifest_rows, manifest_path)

    n_audit = write_audit_samples(selected, out_dir, args.audit_samples_per_cell, args.audit_seed)
    if n_audit:
        print(f"\nAudit samples: {n_audit} clips copied into {out_dir / 'audit_samples'}")

    # Gitignore the audit copies — they are derivative listening fodder.
    gi = out_dir / ".gitignore"
    if not gi.exists():
        gi.write_text("audit_samples/\n")

    print(f"\nDone. {len(manifest_rows) - len(errors)}/{len(manifest_rows)} clips exported.")
    print(f"Manifest: {manifest_path}")
    if errors:
        print(f"Errors ({len(errors)}): {errors[:20]}{'...' if len(errors) > 20 else ''}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
