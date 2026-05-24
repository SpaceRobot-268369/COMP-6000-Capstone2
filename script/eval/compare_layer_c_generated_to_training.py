#!/usr/bin/env python3
"""Compare generated Layer C samples against manually approved training clips.

The output is a diagnostic CSV. It checks whether generated samples sit near
the acoustic distribution of the approved source clips, using robust z-scores
over simple audio metrics. Manual audit remains the source of truth for species
correctness.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from auto_eval_layer_c_generated import evaluate_audio  # noqa: E402


METRIC_COLUMNS = [
    "rms_dbfs",
    "peak_dbfs",
    "active_ratio_db_gt_minus45",
    "active_duration_s",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff85_hz",
    "zero_crossing_rate",
    "band_0_1khz_ratio",
    "band_1_8khz_ratio",
    "band_8khz_plus_ratio",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def metrics_for_rows(rows: list[dict[str, str]], audio_column: str) -> list[dict[str, str | float]]:
    output = []
    for row in rows:
        audio_path = Path(row[audio_column])
        if not audio_path.exists():
            output.append({**row, "dist_flags": "missing_audio", "dist_verdict": "review"})
            continue
        output.append({**row, **evaluate_audio(audio_path)})
    return output


def reference_stats(reference_metrics: list[dict[str, str | float]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in METRIC_COLUMNS:
        values = np.array([float(row[column]) for row in reference_metrics if column in row], dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        q1 = float(np.quantile(values, 0.25))
        q3 = float(np.quantile(values, 0.75))
        iqr = q3 - q1
        robust_scale = max(1.4826 * mad, iqr / 1.349 if iqr > 0 else 0.0, 1e-9)
        stats[column] = {
            "median": median,
            "mad": mad,
            "q1": q1,
            "q3": q3,
            "robust_scale": robust_scale,
        }
    return stats


def add_distribution_flags(
    generated_metrics: list[dict[str, str | float]],
    stats: dict[str, dict[str, float]],
    threshold: float,
) -> list[dict[str, str | float]]:
    output = []
    for row in generated_metrics:
        flags = []
        max_abs_z = 0.0
        for column in METRIC_COLUMNS:
            if column not in row:
                continue
            value = float(row[column])
            median = stats[column]["median"]
            scale = stats[column]["robust_scale"]
            z = (value - median) / scale
            max_abs_z = max(max_abs_z, abs(z))
            row[f"{column}_robust_z"] = round(z, 3)
            if abs(z) > threshold:
                direction = "high" if z > 0 else "low"
                flags.append(f"{column}_{direction}")

        inherited = str(row.get("auto_flags", "")).strip()
        if inherited:
            flags.insert(0, f"auto:{inherited}")
        row["distribution_flags"] = ";".join(flags)
        row["distribution_max_abs_z"] = round(max_abs_z, 3)
        row["distribution_verdict"] = "review" if flags else "in_distribution"
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stats(path: Path, stats: dict[str, dict[str, float]], reference_count: int) -> None:
    rows = []
    for metric, metric_stats in stats.items():
        rows.append({"metric": metric, "reference_count": reference_count, **metric_stats})
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_csv", required=True, type=Path)
    parser.add_argument("--generated_csv", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--reference_stats_csv", required=True, type=Path)
    parser.add_argument("--training_audio_column", default="audio_path")
    parser.add_argument("--generated_audio_column", default="audio_path")
    parser.add_argument("--threshold", type=float, default=3.5)
    args = parser.parse_args()

    training_rows = read_rows(args.training_csv)
    generated_rows = read_rows(args.generated_csv)
    training_metrics = metrics_for_rows(training_rows, args.training_audio_column)
    generated_metrics = metrics_for_rows(generated_rows, args.generated_audio_column)

    stats = reference_stats(training_metrics)
    compared = add_distribution_flags(generated_metrics, stats, args.threshold)

    write_csv(args.output_csv, compared)
    write_stats(args.reference_stats_csv, stats, len(training_metrics))

    counts: dict[str, int] = {}
    for row in compared:
        verdict = str(row["distribution_verdict"])
        counts[verdict] = counts.get(verdict, 0) + 1

    print(f"Wrote {len(compared)} generated rows to {args.output_csv}")
    print(f"Wrote reference stats to {args.reference_stats_csv}")
    print("Distribution verdict counts:", counts)


if __name__ == "__main__":
    main()
