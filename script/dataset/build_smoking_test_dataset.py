"""
build_smoking_test_dataset.py

Selects 50 ambient clips from the spring/night/September cluster (the densest
homogeneous condition in ambient_index.csv) and extracts them as 16 kHz mono
WAV files for the diffusion model smoke test.

Inputs:
    acoustic_ai/data/ambient/ambient_index.csv          filtered ambient segments
    resources/site_257_bowra-dry-a/site_257_env_data.csv  per-recording env data

Outputs (per clip):
    resources/site_257_bowra-dry-a/smoking_test_dataset/clips/<clip_id>/
        audio.wav       16 kHz mono, exactly the ambient segment duration
        caption.txt     one-line text description from env data
        meta.json       full environmental metadata + source provenance

    resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv
        index of all 50 clips with paths + captions

Usage:
    python3 script/dataset/build_smoking_test_dataset.py
    python3 script/dataset/build_smoking_test_dataset.py --dry-run   # print manifest, no extraction
    python3 script/dataset/build_smoking_test_dataset.py --target 50 --season spring --diel night --month 9
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
AMBIENT_INDEX = REPO_ROOT / "acoustic_ai/data/ambient/ambient_index.csv"
ENV_DATA = REPO_ROOT / "resources/site_257_bowra-dry-a/site_257_env_data.csv"
OUT_DIR = REPO_ROOT / "resources/site_257_bowra-dry-a/smoking_test_dataset"
MANIFEST_PATH = OUT_DIR / "manifest.csv"
TARGET_SR = 16000


def load_env_map(path: Path) -> dict:
    env_map = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            env_map[r["recording_id"]] = r
    return env_map


def rec_id_from_clip_path(clip_path: str) -> str:
    folder = clip_path.split("/")[-2]
    return folder.replace("site_257_item_", "")


def build_caption(env: dict, diel: str, season: str) -> str:
    """Generate a one-line text caption from environmental metadata."""
    parts = []

    # Time-of-day descriptor
    diel_labels = {
        "dawn": "dawn",
        "morning": "morning",
        "afternoon": "afternoon",
        "night": "night",
    }
    parts.append(f"{season} {diel_labels.get(diel, diel)}")

    # Acoustic environment
    parts.append("ambient soundscape, Bowra dry woodland, Australia")

    # Temperature
    try:
        temp = float(env.get("temperature_c", ""))
        if temp < 8:
            parts.append(f"cold ({temp:.0f}°C)")
        elif temp < 16:
            parts.append(f"cool ({temp:.0f}°C)")
        elif temp < 24:
            parts.append(f"mild ({temp:.0f}°C)")
        else:
            parts.append(f"warm ({temp:.0f}°C)")
    except (ValueError, TypeError):
        pass

    # Humidity
    try:
        hum = float(env.get("humidity_pct", ""))
        if hum > 80:
            parts.append("humid")
        elif hum < 35:
            parts.append("dry air")
    except (ValueError, TypeError):
        pass

    # Wind
    try:
        wind = float(env.get("wind_speed_ms", ""))
        if wind < 0.5:
            parts.append("still")
        elif wind < 2.0:
            parts.append("light breeze")
        elif wind < 4.5:
            parts.append("moderate wind")
        else:
            parts.append("strong wind")
    except (ValueError, TypeError):
        pass

    # Precipitation context
    try:
        precip = float(env.get("precipitation_mm", "0"))
        days_dry = float(env.get("days_since_rain", "999"))
        if precip > 0.5:
            parts.append("light rain")
        elif days_dry < 2:
            parts.append("post-rain")
        elif days_dry > 20:
            parts.append("extended dry spell")
    except (ValueError, TypeError):
        pass

    # Date context
    date_str = env.get("sample_local_date", "")
    if date_str:
        parts.append(f"recorded {date_str}")

    return ", ".join(parts)


def select_clips(
    ambient_index: Path,
    env_map: dict,
    season: str,
    diel: str,
    month: int,
    target: int,
) -> list[dict]:
    """
    Load ambient_index, filter to (season, diel, month), then select `target` clips
    spread across as many distinct recording dates as possible.
    """
    with open(ambient_index) as f:
        all_segments = list(csv.DictReader(f))

    # Filter to target condition
    candidates = []
    for r in all_segments:
        if r["season"] != season or r["diel_bin"] != diel:
            continue
        rec_id = rec_id_from_clip_path(r["source_clip"])
        env = env_map.get(rec_id, {})
        if env and int(env.get("month", 0)) == month:
            r["_rec_id"] = rec_id
            r["_env"] = env
            r["_date"] = env.get("sample_local_date", "unknown")
            candidates.append(r)

    if not candidates:
        sys.exit(f"No segments found for season={season} diel={diel} month={month}")

    print(f"Candidates after filtering: {len(candidates)} segments")

    # Group by date, then spread selection across dates
    by_date = defaultdict(list)
    for r in candidates:
        by_date[r["_date"]].append(r)

    dates = sorted(by_date.keys())
    print(f"Unique recording dates: {len(dates)}")

    # Round-robin pick across dates to maximise date diversity
    selected = []
    date_iters = {d: iter(segs) for d, segs in by_date.items()}
    active_dates = list(dates)

    while len(selected) < target and active_dates:
        exhausted = []
        for d in list(active_dates):
            if len(selected) >= target:
                break
            try:
                seg = next(date_iters[d])
                selected.append(seg)
            except StopIteration:
                exhausted.append(d)
        for d in exhausted:
            active_dates.remove(d)

    print(f"Selected: {len(selected)} clips from {len(set(s['_date'] for s in selected))} dates")
    return selected


def extract_segment(src_wav: Path, t_start: float, t_end: float, out_wav: Path) -> bool:
    duration = t_end - t_start
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_wav),
        "-ss", str(t_start),
        "-t", str(duration),
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--season", default="spring")
    parser.add_argument("--diel", default="night")
    parser.add_argument("--month", type=int, default=9)
    args = parser.parse_args()

    env_map = load_env_map(ENV_DATA)
    selected = select_clips(AMBIENT_INDEX, env_map, args.season, args.diel, args.month, args.target)

    if args.dry_run:
        print("\n--- DRY RUN MANIFEST ---")
        for i, seg in enumerate(selected, 1):
            caption = build_caption(seg["_env"], seg["diel_bin"], seg["season"])
            print(f"{i:02d}  {seg['segment_id']}  [{seg['_date']}]  {caption}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips_dir = OUT_DIR / "clips"
    clips_dir.mkdir(exist_ok=True)

    manifest_rows = []
    errors = []

    for i, seg in enumerate(selected, 1):
        clip_id = f"{i:03d}_{seg['segment_id']}"
        clip_dir = clips_dir / clip_id
        clip_dir.mkdir(exist_ok=True)

        # Source wav (pre-converted by method-1 pipeline)
        src_clip = seg["source_clip"]
        src_wav = REPO_ROOT / (src_clip + ".wav")
        out_wav = clip_dir / "audio.wav"
        t_start = float(seg["t_start"])
        t_end = float(seg["t_end"])

        caption = build_caption(seg["_env"], seg["diel_bin"], seg["season"])

        # Write caption
        (clip_dir / "caption.txt").write_text(caption + "\n")

        # Write meta
        meta = {
            "clip_id": clip_id,
            "segment_id": seg["segment_id"],
            "source_clip": src_clip,
            "t_start": t_start,
            "t_end": t_end,
            "duration_s": float(seg["duration_s"]),
            "diel_bin": seg["diel_bin"],
            "season": seg["season"],
            "recording_date": seg["_date"],
            "caption": caption,
            "env": {k: v for k, v in seg["_env"].items() if not k.startswith("_")},
        }
        (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        # Extract audio
        if src_wav.exists():
            ok = extract_segment(src_wav, t_start, t_end, out_wav)
            status = "ok" if ok else "ffmpeg_error"
            if not ok:
                errors.append(clip_id)
        else:
            status = "source_missing"
            errors.append(clip_id)
            print(f"  [SKIP] {clip_id} — source wav not found: {src_wav}")

        manifest_rows.append({
            "clip_id": clip_id,
            "segment_id": seg["segment_id"],
            "recording_date": seg["_date"],
            "diel_bin": seg["diel_bin"],
            "season": seg["season"],
            "duration_s": seg["duration_s"],
            "audio_path": str(out_wav.relative_to(REPO_ROOT)),
            "caption": caption,
            "status": status,
        })

        mark = "✓" if status == "ok" else "✗"
        print(f"  {mark} {i:02d}/{len(selected)}  {clip_id}  [{seg['_date']}]")

    # Write manifest
    fieldnames = list(manifest_rows[0].keys())
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\nDone. {len(manifest_rows) - len(errors)}/{len(manifest_rows)} clips extracted.")
    print(f"Manifest: {MANIFEST_PATH}")
    if errors:
        print(f"Errors ({len(errors)}): {errors}")


if __name__ == "__main__":
    main()
