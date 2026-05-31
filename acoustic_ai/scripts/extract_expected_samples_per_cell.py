"""Extract per-cell real-audio "expected" samples for Layer A bank attempts.

Bank attempts (``uses_cells: true``) host one LoRA per (season, diel) cell;
the Dev UI's Expected Results panel needs 2-3 real-audio exemplars per cell
so the user can compare each scene against ground truth.

Per attempt this writes, for each of the 16 cells:

    acoustic_ai/layers/layer_a/attempts/<attempt>/expected/
        <cell>/
            real_<clip_id>/
                audio.wav            # DVC-tracked
                spectrogram.png      # git-tracked
                metadata.json        # git-tracked  (carries "cell": "<cell>")

Selection heuristic (deterministic; same source CSV always yields the same picks):

    1. Filter the cell's source CSV to split == "train" AND status == "ok".
    2. Group rows by recording_id; rank recording_ids by row count desc, then
       by recording_id asc (tie-break).
    3. Take the first row from each of the top N recording_ids (default 3),
       which guarantees N different field recordings whenever the cell has
       at least N distinct recordings.
    4. Fallback: when the cell has < N distinct recordings, top up with the
       earliest remaining rows in CSV order.

The source-clip WAVs live under ``resources/<site>/mvp2_per_cell_dataset/clips/``
which is DVC-tracked — run ``dvc pull`` for the dataset before running this.

Run from the project root with the project venv:

    ./acoustic_ai/.venv/bin/python acoustic_ai/scripts/extract_expected_samples_per_cell.py
    ./acoustic_ai/.venv/bin/python acoustic_ai/scripts/extract_expected_samples_per_cell.py --force
    ./acoustic_ai/.venv/bin/python acoustic_ai/scripts/extract_expected_samples_per_cell.py \
        --attempt lucas__prod_1__per_cell_loras --samples-per-cell 2
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "acoustic_ai"))

from layers.layer_a.attempts.lucas__smoke_1__audioldm2_spring_night.code.layer_a_visualization import (  # noqa: E402
    build_expected_overlay,
    build_expected_png_text,
    render_layer_a_mel_png_bytes,
    waveform_to_layer_a_mel_db,
)

# ---------------------------------------------------------------------------
# Attempt → source-dataset mapping. Both attempts ship the same per-cell bank,
# so they pull from the same per-cell CSVs. Add new bank attempts here.
# ---------------------------------------------------------------------------

DATASET_ROOT = _PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "mvp2_per_cell_dataset"

ATTEMPTS: list[dict] = [
    {"layer": "layer_a", "attempt": "lucas__mvp_2__per_cell_loras",  "dataset_root": DATASET_ROOT},
    {"layer": "layer_a", "attempt": "lucas__prod_1__per_cell_loras", "dataset_root": DATASET_ROOT},
]

CELLS = [
    "spring_dawn",      "spring_morning",   "spring_afternoon",   "spring_night",
    "summer_dawn",      "summer_morning",   "summer_afternoon",   "summer_night",
    "autumn_dawn",      "autumn_morning",   "autumn_afternoon",   "autumn_night",
    "winter_dawn",      "winter_morning",   "winter_afternoon",   "winter_night",
]


# ---------------------------------------------------------------------------
# CSV pick logic
# ---------------------------------------------------------------------------


def _read_cell_csv(csv_path: Path) -> list[dict]:
    """Read a cell CSV preserving file order — order matters for tie-breaks."""
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def pick_rows(rows: list[dict], n: int) -> list[dict]:
    """Deterministic top-N pick: one row from each of the most-represented
    recording_ids; if fewer than N distinct recordings exist, top up with the
    earliest remaining rows."""
    usable = [r for r in rows if r.get("split") == "train" and r.get("status") == "ok"]
    if not usable:
        return []

    by_rec: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        by_rec[r["recording_id"]].append(r)

    # Sort recording_ids by (row count desc, recording_id asc) for stability.
    ranked = sorted(by_rec.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    picks: list[dict] = [rows_for_rec[0] for _, rows_for_rec in ranked[:n]]

    # Top-up if we ran short on distinct recordings.
    if len(picks) < n:
        already = {(p["recording_id"], p["clip_id"]) for p in picks}
        for r in usable:
            key = (r["recording_id"], r["clip_id"])
            if key not in already:
                picks.append(r)
                already.add(key)
                if len(picks) >= n:
                    break

    return picks[:n]


# ---------------------------------------------------------------------------
# Per-pick extraction
# ---------------------------------------------------------------------------


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _audio_stats(audio: np.ndarray, sr: int) -> dict:
    return {
        "sample_rate": sr,
        "duration_s":  float(audio.shape[0] / sr),
        "rms":         float(np.sqrt(np.mean(np.square(audio)))),
        "peak":        float(np.max(np.abs(audio))),
    }


def _source_metadata(row: dict) -> dict:
    """Project the CSV row into the `source_metadata` block in metadata.json."""
    def _f(key: str) -> float | None:
        v = row.get(key, "")
        try:
            return float(v) if v != "" else None
        except ValueError:
            return None
    return {
        "caption":        row.get("caption", ""),
        "clip_id":        row["clip_id"],
        "segment_id":     row.get("segment_id", ""),
        "recording_id":   row.get("recording_id", ""),
        "recording_date": row.get("recording_date", ""),
        "diel_bin":       row.get("diel_bin", ""),
        "season":         row.get("season", ""),
        "duration_s":     _f("duration_s"),
        "source_clip":    row.get("audio_path", ""),
        "env": {
            "temperature_c": _f("temperature_c"),
            "humidity_pct":  _f("humidity_pct"),
            "wind_speed_ms": _f("wind_speed_ms"),
        },
    }


def _extract_one(row: dict, *, layer: str, attempt: str, cell: str,
                 attempt_root: Path, dataset_root: Path, force: bool) -> str:
    """Materialise one expected sample. Returns a short status tag."""
    clip_id = row["clip_id"]
    stem = f"real_{clip_id}"
    case = attempt_root / "expected" / cell / stem
    dst_wav  = case / "audio.wav"
    dst_png  = case / "spectrogram.png"
    dst_json = case / "metadata.json"

    if not force and dst_wav.is_file() and dst_png.is_file() and dst_json.is_file():
        return "skip (already present)"

    src_wav = _PROJECT_ROOT / row["audio_path"]
    if not src_wav.is_file():
        raise FileNotFoundError(
            f"source WAV missing: {src_wav.relative_to(_PROJECT_ROOT)} "
            f"(run `dvc pull {dataset_root.relative_to(_PROJECT_ROOT)}/clips.dvc`)"
        )

    case.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_wav, dst_wav)

    audio, sr = _load_wav_mono(dst_wav)
    duration_s = float(audio.shape[0] / sr)
    mel_db = waveform_to_layer_a_mel_db(audio, sr)

    src_meta = _source_metadata(row)
    metadata = {
        "tier":             "expected",
        "cell":             cell,
        "source":           "real_audio",
        "source_kind":      "clip_dir",
        "source_clip_id":   clip_id,
        "source_manifest":  str(
            (dataset_root / f"cell_{cell}.csv").relative_to(_PROJECT_ROOT)
        ),
        "selection_reason": (
            f"top-recording_id pick for cell {cell} "
            f"(recording_id={row.get('recording_id', '?')})"
        ),
        "audio":            _audio_stats(audio, sr),
        "source_metadata":  src_meta,
    }

    png_bytes = render_layer_a_mel_png_bytes(
        mel_db, duration_s,
        overlay=build_expected_overlay(metadata),
        png_text=build_expected_png_text(metadata),
    )
    dst_png.write_bytes(png_bytes)
    dst_json.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return "wrote"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _iter_targets(only_attempt: str | None) -> Iterator[dict]:
    for spec in ATTEMPTS:
        if only_attempt and spec["attempt"] != only_attempt:
            continue
        yield spec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attempt", help="limit to a single attempt id")
    ap.add_argument("--samples-per-cell", type=int, default=3,
                    help="number of expected samples per cell (default 3)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing case dirs")
    args = ap.parse_args()

    exit_code = 0
    for spec in _iter_targets(args.attempt):
        layer = spec["layer"]
        attempt = spec["attempt"]
        dataset_root = spec["dataset_root"]
        attempt_root = _PROJECT_ROOT / "acoustic_ai" / "layers" / layer / "attempts" / attempt
        if not attempt_root.is_dir():
            print(f"[skip] {layer}/{attempt}: attempt dir missing")
            continue

        print(f"\n=== {layer}/{attempt} (source: {dataset_root.relative_to(_PROJECT_ROOT)}) ===")
        for cell in CELLS:
            csv_path = dataset_root / f"cell_{cell}.csv"
            if not csv_path.is_file():
                print(f"  [skip] {cell}: cell CSV missing ({csv_path.name})")
                continue

            rows = _read_cell_csv(csv_path)
            picks = pick_rows(rows, args.samples_per_cell)
            if not picks:
                print(f"  [warn] {cell}: no usable rows (train + ok)")
                continue

            for row in picks:
                try:
                    tag = _extract_one(
                        row, layer=layer, attempt=attempt, cell=cell,
                        attempt_root=attempt_root, dataset_root=dataset_root,
                        force=args.force,
                    )
                    print(f"  [{tag:>4}] {cell}/real_{row['clip_id']}")
                except Exception as exc:  # noqa: BLE001
                    exit_code = 1
                    print(f"  [FAIL] {cell}/real_{row.get('clip_id','?')}: "
                          f"{type(exc).__name__}: {exc}")

    if exit_code == 0:
        print("\nNext: track WAVs with DVC and commit the PNG/JSON sidecars, e.g.")
        print("  ATT=acoustic_ai/layers/layer_a/attempts/lucas__prod_1__per_cell_loras")
        print("  dvc add $ATT/expected/*/*/audio.wav")
        print("  git add $ATT/expected/")
        print("  git commit -m 'data: per-cell expected samples for layer_a bank'")
        print("  git push && dvc push")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
