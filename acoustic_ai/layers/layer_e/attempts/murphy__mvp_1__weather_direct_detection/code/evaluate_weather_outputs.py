"""Evaluate Layer E-B weather analysis JSON outputs against a small manifest.

This helper is intentionally lightweight: it does not run models, download
audio, or depend on pandas. It only compares existing analysis JSON files
against human/fixture labels so gate changes can be checked quickly.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABEL_ORDER = [
    "none",
    "rain",
    "wind",
    "thunder",
    "rain+wind",
    "rain+thunder",
    "wind+thunder",
    "rain+thunder+wind",
]

ID_COLUMNS = ("audio_id", "id", "clip_id", "sample_id", "stem")
EXPECTED_COLUMNS = ("expected_label", "expected", "label", "human_label")
RESULT_COLUMNS = ("result_json", "json_path", "output_json")


@dataclass(frozen=True)
class ExpectedRow:
    audio_id: str
    expected_label: str
    result_json: Path | None = None


def _first_present(row: dict[str, str], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column, "").strip()
        if value:
            return value
    return ""


def load_manifest(path: Path) -> list[ExpectedRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[ExpectedRow] = []
        for line_number, row in enumerate(reader, start=2):
            audio_id = _first_present(row, ID_COLUMNS)
            expected_label = _first_present(row, EXPECTED_COLUMNS)
            result_value = _first_present(row, RESULT_COLUMNS)
            if not audio_id or not expected_label:
                raise ValueError(
                    f"{path}:{line_number} needs one id column {ID_COLUMNS} "
                    f"and one expected label column {EXPECTED_COLUMNS}"
                )
            rows.append(
                ExpectedRow(
                    audio_id=audio_id,
                    expected_label=expected_label,
                    result_json=Path(result_value) if result_value else None,
                )
            )
    return rows


def resolve_result_path(
    row: ExpectedRow,
    manifest_path: Path,
    results_dir: Path | None,
) -> Path:
    if row.result_json is not None:
        path = row.result_json
        if not path.is_absolute():
            path = manifest_path.parent / path
        return path

    if results_dir is None:
        raise ValueError(
            f"No result_json column for {row.audio_id}; pass --results-dir instead."
        )

    candidates = [
        results_dir / f"{row.audio_id}.json",
        results_dir / f"{row.audio_id}_analysis.json",
        results_dir / f"{row.audio_id}_weather.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    glob_matches = sorted(results_dir.glob(f"*{row.audio_id}*.json"))
    if len(glob_matches) == 1:
        return glob_matches[0]
    if len(glob_matches) > 1:
        raise ValueError(
            f"Multiple result JSON files match {row.audio_id}: "
            + ", ".join(str(path) for path in glob_matches)
        )
    return candidates[0]


def load_prediction(path: Path) -> tuple[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    label = str(data.get("weather", {}).get("overall_label", "")).strip()
    if not label:
        raise ValueError(f"{path} does not contain weather.overall_label")
    return label, data


def element_presence(label: str) -> set[str]:
    if label == "none" or not label:
        return set()
    return set(label.split("+"))


def summarize(rows: list[ExpectedRow], manifest_path: Path, results_dir: Path | None) -> dict[str, Any]:
    total = 0
    exact = 0
    missing: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []
    by_expected: dict[str, Counter[str]] = defaultdict(Counter)
    element_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        result_path = resolve_result_path(row, manifest_path, results_dir)
        total += 1
        if not result_path.exists():
            missing.append(
                {
                    "audio_id": row.audio_id,
                    "expected": row.expected_label,
                    "result_json": str(result_path),
                }
            )
            by_expected[row.expected_label]["missing"] += 1
            continue

        predicted, data = load_prediction(result_path)
        by_expected[row.expected_label][predicted] += 1
        if predicted == row.expected_label:
            exact += 1
        else:
            mismatches.append(
                {
                    "audio_id": row.audio_id,
                    "expected": row.expected_label,
                    "predicted": predicted,
                    "result_json": str(result_path),
                    "warnings": ",".join(data.get("weather", {}).get("warnings", [])),
                }
            )

        expected_elements = element_presence(row.expected_label)
        predicted_elements = element_presence(predicted)
        for element in ("rain", "wind", "thunder"):
            if element in expected_elements and element in predicted_elements:
                element_counts[element]["true_positive"] += 1
            elif element in expected_elements and element not in predicted_elements:
                element_counts[element]["false_negative"] += 1
            elif element not in expected_elements and element in predicted_elements:
                element_counts[element]["false_positive"] += 1
            else:
                element_counts[element]["true_negative"] += 1

    evaluated = total - len(missing)
    return {
        "total": total,
        "evaluated": evaluated,
        "missing": len(missing),
        "exact": exact,
        "exact_rate": round(exact / evaluated, 6) if evaluated else 0.0,
        "by_expected": {
            label: dict(by_expected[label])
            for label in LABEL_ORDER
            if label in by_expected
        },
        "element_counts": {
            element: dict(counts)
            for element, counts in element_counts.items()
        },
        "mismatches": mismatches,
        "missing_results": missing,
    }


def print_text_summary(summary: dict[str, Any]) -> None:
    print(
        f"Exact: {summary['exact']}/{summary['evaluated']} "
        f"({summary['exact_rate']:.3f}); missing {summary['missing']} of {summary['total']}"
    )
    print("\nBy expected label:")
    for label, counts in summary["by_expected"].items():
        rendered = ", ".join(f"{pred}={count}" for pred, count in sorted(counts.items()))
        print(f"  {label}: {rendered}")
    print("\nElement counts:")
    for element, counts in summary["element_counts"].items():
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        print(f"  {element}: {rendered}")
    if summary["mismatches"]:
        print("\nMismatches:")
        for row in summary["mismatches"]:
            print(
                f"  {row['audio_id']}: expected {row['expected']} -> "
                f"{row['predicted']} ({row['warnings']})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate E-B weather JSON outputs against expected labels."
    )
    parser.add_argument("manifest", type=Path, help="CSV with id and expected label columns.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing result JSON files when manifest has no result_json column.",
    )
    parser.add_argument("--out", type=Path, help="Optional path for summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    summary = summarize(rows, args.manifest, args.results_dir)
    print_text_summary(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
