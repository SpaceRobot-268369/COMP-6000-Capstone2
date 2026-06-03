"""Build local E-C positive clips from downloaded species recordings.

This script is intentionally dependency-light: it uses Python stdlib plus
ffmpeg/ffprobe. It writes local training data under local_data/ by default;
do not commit generated WAVs.
"""

from __future__ import annotations

import argparse
import audioop
import csv
import math
import re
import subprocess
from pathlib import Path


SAMPLE_RATE = 16_000
FRAME_S = 0.25
SOURCE_RE = re.compile(r"(XC\d+)", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--target-count", type=int, default=220)
    parser.add_argument("--max-per-source", type=int, default=10)
    parser.add_argument("--clip-duration-s", type=float, default=5.0)
    parser.add_argument("--min-spacing-s", type=float, default=2.5)
    parser.add_argument("--train-sources", type=float, default=0.70)
    parser.add_argument("--val-sources", type=float, default=0.15)
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise FileNotFoundError(args.raw_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    sources = sorted(
        p for p in args.raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    )
    if not sources:
        raise RuntimeError(f"No audio files found in {args.raw_dir}")

    split_by_source = assign_splits(sources, args.train_sources, args.val_sources)
    candidates: list[dict[str, object]] = []

    for source in sources:
        source_id = source_xc_id(source)
        duration_s = probe_duration(source)
        starts = select_clip_starts(
            source,
            duration_s=duration_s,
            clip_duration_s=args.clip_duration_s,
            max_clips=args.max_per_source,
            min_spacing_s=args.min_spacing_s,
        )

        for start_s in starts:
            candidates.append({
                "source": source,
                "source_id": source_id,
                "start_s": start_s,
                "split": split_by_source[source.name],
            })

    selected = select_by_split(candidates, args.target_count, args.train_sources, args.val_sources)
    source_clip_counts: dict[str, int] = {}
    rows: list[dict[str, str]] = []

    for item in selected:
        source = item["source"]
        assert isinstance(source, Path)
        source_id = str(item["source_id"])
        start_s = float(item["start_s"])
        split = str(item["split"])
        source_clip_counts[source.name] = source_clip_counts.get(source.name, 0) + 1
        idx = source_clip_counts[source.name]
        end_s = start_s + args.clip_duration_s
        start_ms = int(round(start_s * 1000))
        end_ms = int(round(end_s * 1000))
        clip_id = (
            f"{args.label}__{source_id}__"
            f"s{start_ms:06d}_e{end_ms:06d}__clip{idx:03d}"
        )
        out_path = args.output_dir / f"{clip_id}.wav"
        transcode_clip(source, out_path, start_s, args.clip_duration_s)
        rows.append({
            "clip_id": clip_id,
            "label": args.label,
            "source_file": source.name,
            "source_xc_id": source_id,
            "start_s": fmt_float(start_s),
            "end_s": fmt_float(end_s),
            "duration_s": fmt_float(args.clip_duration_s),
            "split": split,
            "audio_path": to_manifest_path(out_path),
            "notes": "auto_selected",
        })

    write_manifest(args.manifest, rows)
    print(f"wrote {len(rows)} clips")
    print(f"output: {args.output_dir}")
    print(f"manifest: {args.manifest}")
    print("split counts:", split_counts(rows))
    return 0


def select_by_split(
    candidates: list[dict[str, object]],
    target_count: int,
    train_ratio: float,
    val_ratio: float,
) -> list[dict[str, object]]:
    split_targets = {
        "train": int(round(target_count * train_ratio)),
        "val": int(round(target_count * val_ratio)),
    }
    split_targets["test"] = target_count - split_targets["train"] - split_targets["val"]

    by_split: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for item in candidates:
        by_split[str(item["split"])].append(item)

    selected: list[dict[str, object]] = []
    leftovers: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        items = by_split[split]
        take = min(split_targets[split], len(items))
        selected.extend(items[:take])
        leftovers.extend(items[take:])

    shortfall = target_count - len(selected)
    if shortfall > 0:
        selected.extend(leftovers[:shortfall])

    return selected


def assign_splits(sources: list[Path], train_ratio: float, val_ratio: float) -> dict[str, str]:
    count = len(sources)
    train_n = max(1, int(round(count * train_ratio)))
    val_n = max(1, int(round(count * val_ratio)))
    if train_n + val_n >= count:
        val_n = max(1, count - train_n - 1)

    split_by_source: dict[str, str] = {}
    for idx, source in enumerate(sources):
        if idx < train_n:
            split = "train"
        elif idx < train_n + val_n:
            split = "val"
        else:
            split = "test"
        split_by_source[source.name] = split
    return split_by_source


def source_xc_id(path: Path) -> str:
    match = SOURCE_RE.search(path.name)
    return match.group(1).upper() if match else path.stem.replace(" ", "_")


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return float(proc.stdout.strip())


def select_clip_starts(
    path: Path,
    *,
    duration_s: float,
    clip_duration_s: float,
    max_clips: int,
    min_spacing_s: float,
) -> list[float]:
    if duration_s < clip_duration_s:
        return []

    energies = decode_frame_rms(path)
    if not energies:
        return []

    median_energy = median(energies)
    floor = max(median_energy, 1.0)
    candidates: list[tuple[float, float]] = []
    half_clip = clip_duration_s / 2.0

    for frame_idx, rms in enumerate(energies):
        t = frame_idx * FRAME_S
        if t < 0.25 or t > duration_s - 0.25:
            continue
        left = energies[max(0, frame_idx - 2):frame_idx]
        right = energies[frame_idx + 1:frame_idx + 3]
        local_max = all(rms >= v for v in left + right)
        if not local_max:
            continue

        score = rms / floor
        if score < 1.35:
            continue

        start = min(max(t - half_clip, 0.0), max(duration_s - clip_duration_s, 0.0))
        candidates.append((score, start))

    if not candidates:
        # Fallback: take the loudest frames conservatively.
        for frame_idx, rms in sorted(enumerate(energies), key=lambda item: item[1], reverse=True)[:max_clips * 3]:
            t = frame_idx * FRAME_S
            start = min(max(t - half_clip, 0.0), max(duration_s - clip_duration_s, 0.0))
            candidates.append((rms / floor, start))

    selected: list[float] = []
    for _score, start in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(abs(start - existing) >= min_spacing_s for existing in selected):
            selected.append(round(start, 3))
        if len(selected) >= max_clips:
            break

    return sorted(selected)


def decode_frame_rms(path: Path) -> list[float]:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-f", "s16le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    frame_bytes = int(SAMPLE_RATE * FRAME_S) * 2
    data = proc.stdout
    return [
        float(audioop.rms(data[i:i + frame_bytes], 2))
        for i in range(0, len(data) - frame_bytes + 1, frame_bytes)
    ]


def transcode_clip(source: Path, out_path: Path, start_s: float, duration_s: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y",
            "-ss", fmt_float(start_s),
            "-i", str(source),
            "-t", fmt_float(duration_s),
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            str(out_path),
        ],
        check=True,
    )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "clip_id", "label", "source_file", "source_xc_id", "start_s", "end_s",
        "duration_s", "split", "audio_path", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    return counts


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def fmt_float(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def to_manifest_path(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
