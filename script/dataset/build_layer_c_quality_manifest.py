#!/usr/bin/env python3
"""Build quality-ranked Layer C training manifests from prepared event clips.

The first Layer C filter is metadata-only: species label, BirdNET score, event
duration, diel bin, and source-clip boundaries. This script adds an audio-level
second pass for training data:

- join prepared segment metadata with manual audit verdicts
- crop each event to a shorter foreground window
- compute simple audio quality features
- write per-species training manifests sorted by quality

The goal is not to replace manual audit. It is to stop training on long,
background-heavy clips when Layer C should learn foreground bird-call events.
For larger recovery pools, the script can also score unaudited clips first so
manual review starts from the strongest candidates.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = (
    REPO_ROOT / "resources" / "site_257_bowra-dry-a" / "layer_c_smoke_2_3_species"
)
DEFAULT_TARGET_SR = 16_000

SPECIES_PROMPTS = {
    "splendid_fairywren": "Splendid Fairywren bird call, isolated foreground call, Bowra dry woodland",
    "chestnut_rumped_thornbill": "Chestnut-rumped Thornbill sharp bird call, isolated foreground call, Bowra dry woodland",
    "boobook": "Southern Boobook two-note mopoke owl call at night, isolated foreground call, Bowra dry woodland",
    "red_capped_robin": "Red-capped Robin bird call, isolated foreground call, Bowra dry woodland",
    "crested_bellbird": "Crested Bellbird ringing bird call, isolated foreground call, Bowra dry woodland",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create audio-quality-ranked Layer C v2 training manifests."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--prepared-manifest", type=Path, default=None)
    parser.add_argument("--audit-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-verdict",
        action="append",
        default=["Pass"],
        help="Manual audit verdict to include. Repeatable. Default: Pass.",
    )
    parser.add_argument(
        "--include-unaudited",
        action="store_true",
        help=(
            "Also score prepared clips that do not have a manual audit row yet. "
            "Use this for large candidate pools before manual review."
        ),
    )
    parser.add_argument("--crop-buffer-s", type=float, default=0.5)
    parser.add_argument("--target-sr", type=int, default=DEFAULT_TARGET_SR)
    parser.add_argument("--top-n-per-species", type=int, default=80)
    parser.add_argument("--min-event-bg-db", type=float, default=3.0)
    parser.add_argument("--min-active-ratio", type=float, default=0.03)
    parser.add_argument("--max-silence-ratio", type=float, default=0.95)
    parser.add_argument("--overwrite-crops", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float32)))))


def db_ratio(numerator: float, denominator: float) -> float:
    return float(20.0 * math.log10((numerator + 1e-8) / (denominator + 1e-8)))


def frame_rms(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(1, int(round(0.05 * sample_rate)))
    hop = max(1, int(round(0.025 * sample_rate)))
    if waveform.size < frame:
        return np.asarray([rms(waveform)], dtype=np.float32)
    values = []
    for start in range(0, waveform.size - frame + 1, hop):
        values.append(rms(waveform[start : start + frame]))
    return np.asarray(values, dtype=np.float32)


def spectral_centroid_hz(waveform: np.ndarray, sample_rate: int) -> float:
    if waveform.size < 16:
        return 0.0
    window = np.hanning(waveform.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(waveform.astype(np.float32) * window))
    freqs = np.fft.rfftfreq(waveform.size, d=1.0 / sample_rate)
    total = float(np.sum(spectrum))
    if total <= 1e-8:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def high_freq_ratio(waveform: np.ndarray, sample_rate: int, cutoff_hz: float = 3000.0) -> float:
    if waveform.size < 16:
        return 0.0
    spectrum = np.square(np.abs(np.fft.rfft(waveform.astype(np.float32))))
    freqs = np.fft.rfftfreq(waveform.size, d=1.0 / sample_rate)
    total = float(np.sum(spectrum))
    if total <= 1e-8:
        return 0.0
    return float(np.sum(spectrum[freqs >= cutoff_hz]) / total)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def quality_features(
    waveform: np.ndarray,
    sample_rate: int,
    event_start_s: float,
    event_end_s: float,
    crop_start_s: float,
    crop_end_s: float,
) -> dict[str, float]:
    event_start = max(0, int(round(event_start_s * sample_rate)))
    event_end = min(waveform.size, int(round(event_end_s * sample_rate)))
    crop_start = max(0, int(round(crop_start_s * sample_rate)))
    crop_end = min(waveform.size, int(round(crop_end_s * sample_rate)))

    event_audio = waveform[event_start:event_end]
    crop_audio = waveform[crop_start:crop_end]
    bg_audio = np.concatenate([waveform[:event_start], waveform[event_end:]])

    event_rms = rms(event_audio)
    bg_rms = rms(bg_audio)
    crop_rms = rms(crop_audio)
    peak = float(np.max(np.abs(crop_audio))) if crop_audio.size else 0.0
    frames = frame_rms(crop_audio, sample_rate)
    active_threshold = max(bg_rms * 1.5, crop_rms * 0.2, 1e-5)
    active_ratio = float(np.mean(frames >= active_threshold)) if frames.size else 0.0
    silence_ratio = float(np.mean(frames < active_threshold)) if frames.size else 1.0
    event_bg_db = db_ratio(event_rms, bg_rms)

    # Reward clear foreground energy and some activity, but avoid scoring pure
    # wall-to-wall noise as ideal.
    quality_score = (
        event_bg_db
        + 8.0 * min(active_ratio, 0.5)
        - 4.0 * max(0.0, silence_ratio - 0.85)
        - 2.0 * float(peak >= 0.98)
    )

    return {
        "event_rms": event_rms,
        "background_rms": bg_rms,
        "crop_rms": crop_rms,
        "peak": peak,
        "event_to_background_db": event_bg_db,
        "active_ratio": active_ratio,
        "silence_ratio": silence_ratio,
        "spectral_centroid_hz": spectral_centroid_hz(event_audio, sample_rate),
        "high_freq_ratio_3khz": high_freq_ratio(event_audio, sample_rate),
        "quality_score": quality_score,
    }


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else REPO_ROOT / args.dataset_dir
    prepared_path = args.prepared_manifest or dataset_dir / "prepared_manifest.csv"
    audit_path = args.audit_manifest or dataset_dir / "manual_audit_grouped.csv"
    output_dir = args.output_dir or dataset_dir / "quality_v2"
    output_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir

    prepared_rows = read_csv(prepared_path)
    if audit_path.exists():
        audit_rows = read_csv(audit_path)
    elif args.include_unaudited:
        audit_rows = []
    else:
        raise FileNotFoundError(
            f"audit manifest not found: {audit_path}. "
            "Use --include-unaudited to quality-rank clips before manual audit."
        )
    audit_by_event_id = {row["audio_event_id"]: row for row in audit_rows}
    include_verdicts = {value.lower() for value in args.include_verdict}

    scored_rows: list[dict[str, str]] = []
    train_rows_by_species: dict[str, list[dict[str, str]]] = {}
    crop_dir = output_dir / "crops"

    for row in prepared_rows:
        event_id = row["audio_event_id"]
        audit = audit_by_event_id.get(event_id)
        verdict = (audit or {}).get("verdict", "")
        if verdict:
            if verdict.lower() not in include_verdicts:
                continue
        elif not args.include_unaudited:
            continue

        audio_path = resolve_path(row["audio_path"])
        if not audio_path.exists():
            continue

        waveform, sample_rate = load_audio(audio_path)
        if sample_rate != args.target_sr:
            raise ValueError(
                f"expected {args.target_sr} Hz prepared audio, got {sample_rate}: {audio_path}"
            )

        buffer_seconds = float(row["buffer_seconds"])
        event_duration = float(row["event_duration_seconds"])
        event_start_s = buffer_seconds
        event_end_s = buffer_seconds + event_duration
        crop_start_s = max(0.0, event_start_s - args.crop_buffer_s)
        crop_end_s = min(float(waveform.size / sample_rate), event_end_s + args.crop_buffer_s)
        crop_start = int(round(crop_start_s * sample_rate))
        crop_end = int(round(crop_end_s * sample_rate))
        crop_audio = waveform[crop_start:crop_end]

        species_key = row["event_type"]
        crop_path = (
            crop_dir
            / species_key
            / f"audioevent_{event_id}_crop_{args.crop_buffer_s:.2f}s.wav"
        )
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite_crops or not crop_path.exists():
            sf.write(crop_path, crop_audio, sample_rate, subtype="PCM_16")

        features = quality_features(
            waveform,
            sample_rate,
            event_start_s,
            event_end_s,
            crop_start_s,
            crop_end_s,
        )
        passes_quality_gate = (
            features["event_to_background_db"] >= args.min_event_bg_db
            and features["active_ratio"] >= args.min_active_ratio
            and features["silence_ratio"] <= args.max_silence_ratio
        )

        caption = SPECIES_PROMPTS.get(species_key, row["caption"])
        scored = {
            **row,
            "manual_verdict": verdict,
            "manual_notes": (audit or {}).get("notes", ""),
            "crop_audio_path": display_path(crop_path),
            "crop_start_seconds": f"{crop_start_s:.3f}",
            "crop_end_seconds": f"{crop_end_s:.3f}",
            "crop_duration_seconds": f"{(crop_end_s - crop_start_s):.3f}",
            **{key: f"{value:.6f}" for key, value in features.items()},
            "quality_gate": "Pass" if passes_quality_gate else "Fail",
            "v2_caption": caption,
        }
        scored_rows.append(scored)

        if passes_quality_gate:
            train_rows_by_species.setdefault(species_key, []).append(
                {
                    "audit_index": (audit or {}).get("audit_index", row["smoke_event_index"]),
                    "species_index": row["species_index"],
                    "event_type": species_key,
                    "species_common_name": row["species_common_name"],
                    "audio_event_id": event_id,
                    "score": row["score"],
                    "diel_bin": row["diel_bin"],
                    "audio_path": display_path(crop_path),
                    "mel_spectrogram_png_path": row["mel_spectrogram_png_path"],
                    "caption": caption,
                    "verdict": verdict,
                    "notes": (audit or {}).get("notes", ""),
                    "quality_score": f"{features['quality_score']:.6f}",
                    "event_to_background_db": f"{features['event_to_background_db']:.6f}",
                    "active_ratio": f"{features['active_ratio']:.6f}",
                    "silence_ratio": f"{features['silence_ratio']:.6f}",
                    "crop_duration_seconds": f"{(crop_end_s - crop_start_s):.3f}",
                }
            )

    scored_rows.sort(
        key=lambda row: (
            row["event_type"],
            -float(row["quality_score"]),
            -float(row["score"]),
        )
    )
    write_csv(output_dir / "quality_scored_manifest.csv", scored_rows)

    audit_manifest_report = (
        f"`{display_path(audit_path)}`"
        if audit_path.exists()
        else "`not found; unaudited clips included`"
    )
    report_lines = [
        "# Layer C Quality V2 Report",
        "",
        f"- Prepared manifest: `{display_path(prepared_path)}`",
        f"- Audit manifest: {audit_manifest_report}",
        f"- Included verdicts: `{', '.join(sorted(include_verdicts))}`",
        f"- Include unaudited: `{args.include_unaudited}`",
        f"- Crop buffer: `{args.crop_buffer_s}` seconds",
        f"- Minimum event/background dB: `{args.min_event_bg_db}`",
        f"- Minimum active ratio: `{args.min_active_ratio}`",
        f"- Maximum silence ratio: `{args.max_silence_ratio}`",
        "",
        "| Species | Scored | Quality pass | Manifest |",
        "|---|---:|---:|---|",
    ]

    species_keys = sorted({row["event_type"] for row in scored_rows})
    for species_key in species_keys:
        species_rows = [row for row in scored_rows if row["event_type"] == species_key]
        train_rows = sorted(
            train_rows_by_species.get(species_key, []),
            key=lambda row: -float(row["quality_score"]),
        )[: args.top_n_per_species]
        manifest_path = output_dir / f"train_manifest_{species_key}_quality_v2.csv"
        write_csv(manifest_path, train_rows)
        report_lines.append(
            f"| {species_key} | {len(species_rows)} | {len(train_rows)} | "
            f"`{display_path(manifest_path)}` |"
        )

    (output_dir / "quality_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'quality_scored_manifest.csv'}")
    print(f"wrote {output_dir / 'quality_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
