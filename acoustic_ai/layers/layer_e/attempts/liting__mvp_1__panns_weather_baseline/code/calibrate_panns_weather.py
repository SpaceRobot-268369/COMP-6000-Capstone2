"""Export PANNs calibration evidence for Layer E-B MVP-1.

This script does not train a model. It records PANNs zero-shot scores against
Murphy/Liting's site257 promoted Layer B weather assets so the next iteration can
calibrate thresholds instead of guessing them.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from layers.layer_e.attempts.liting__mvp_1__panns_weather_baseline.code import weather_detector  # noqa: E402
from tests.e_b_weather_mvp_test import (  # noqa: E402
    DEFAULT_MAIN_INDEX,
    DEFAULT_SITE_PROMOTED_MANIFEST,
    classify_policy_case,
    compare_label,
    evaluate_case,
    load_assets,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / "debug" / "e_b_weather_mvp" / "panns_calibration"
COMPONENTS = ("rain", "wind", "thunder")
SWEEP_THRESHOLDS = [round(i / 100, 2) for i in range(0, 61)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export E-B PANNs calibration evidence.")
    parser.add_argument("--asset-index", type=Path, default=DEFAULT_MAIN_INDEX)
    parser.add_argument("--site-promoted-manifest", type=Path, default=DEFAULT_SITE_PROMOTED_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Optional max case count.")
    args = parser.parse_args()

    assets, source_note = load_assets(args.site_promoted_manifest, args.asset_index, legacy_root=None)
    if args.limit > 0:
        assets = assets[: args.limit]
    assets = [asset for asset in assets if asset.audio_path.exists()]
    if not assets:
        print("FAIL: no materialised weather assets found.")
        return 1

    rows = []
    for asset in assets:
        spectral = weather_detector.smoke.analyse_weather(asset.audio_path, calibration_assets=assets)
        panns = weather_detector.score_with_panns(asset.audio_path)
        result = (
            weather_detector._fuse_panns_with_spectral(spectral, panns)
            if panns.available
            else dict(spectral)
        )
        policy_class = classify_policy_case(asset)
        status, component_status = evaluate_case(asset, result, policy_class)

        row = {
            "asset_id": asset.asset_id,
            "audio_path": str(asset.audio_path.relative_to(PROJECT_ROOT)),
            "policy_class": policy_class,
            "expected_rain": asset.labels["rain"],
            "expected_wind": asset.labels["wind"],
            "expected_thunder": asset.labels.get("thunder", "none"),
            "spectral_rain": spectral["rain_intensity"],
            "spectral_wind": spectral["wind_intensity"],
            "spectral_thunder": spectral["thunder_intensity"],
            "fused_rain": result["rain_intensity"],
            "fused_wind": result["wind_intensity"],
            "fused_thunder": result["thunder_intensity"],
            "fused_status": status,
            "rain_status": component_status["rain"],
            "wind_status": component_status["wind"],
            "panns_available": panns.available,
            "panns_status": panns.reason,
        }
        for component in COMPONENTS:
            row[f"panns_{component}_score"] = round(panns.scores.get(component, 0.0), 6)
            for label, score in panns.labels.get(component, {}).items():
                key = f"panns_label_{normalise_key(label)}"
                row[key] = round(score, 6)
        rows.append(row)
        print(
            f"[calibrate] {asset.asset_id}: policy={policy_class} "
            f"expected rain={row['expected_rain']} wind={row['expected_wind']} | "
            f"panns rain={row['panns_rain_score']:.4f} wind={row['panns_wind_score']:.4f} | "
            f"fused rain={row['fused_rain']} wind={row['fused_wind']} -> {status}"
        )

    summary = build_summary(rows, source_note)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "panns_weather_scores.csv", rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Scores written to: {args.out_dir / 'panns_weather_scores.csv'}")
    print(f"Summary written to: {args.out_dir / 'summary.json'}")
    for component, data in summary["threshold_sweep"].items():
        best = data["best"]
        print(
            f"{component}: best_presence_threshold={best['threshold']:.2f}, "
            f"accuracy={best['accuracy']:.2f}, cases={best['case_count']}"
        )
    return 0


def build_summary(rows: list[dict], source_note: str) -> dict:
    primary_rows = [row for row in rows if row["policy_class"] in {"rain_primary", "wind_primary"}]
    return {
        "component": "E-B",
        "attempt": "liting__mvp_1__panns_weather_baseline",
        "asset_source": source_note,
        "case_count": len(rows),
        "primary_case_count": len(primary_rows),
        "panns_available_count": sum(1 for row in rows if row["panns_available"]),
        "status_counts": count_values(row["fused_status"] for row in rows),
        "policy_counts": count_values(row["policy_class"] for row in rows),
        "score_stats": {
            component: score_stats_by_expected(primary_rows, component)
            for component in ("rain", "wind")
        },
        "threshold_sweep": {
            component: sweep_presence_threshold(primary_rows, component)
            for component in ("rain", "wind")
        },
        "recommendation": [
            "Use primary rain/wind assets for threshold calibration; keep rain_wind_mixed as boundary validation.",
            "Do not fine-tune yet. First compare calibrated PANNs thresholds against the spectral fallback.",
            "Add no-weather negatives before treating PANNs confidence as a reliable absolute probability.",
        ],
    }


def sweep_presence_threshold(rows: list[dict], component: str) -> dict:
    cases = [row for row in rows if row[f"expected_{component}"] != "unclear"]
    results = []
    for threshold in SWEEP_THRESHOLDS:
        correct = 0
        false_positive = 0
        false_negative = 0
        for row in cases:
            expected_present = row[f"expected_{component}"] != "none"
            observed_present = float(row[f"panns_{component}_score"]) >= threshold
            if observed_present == expected_present:
                correct += 1
            elif observed_present:
                false_positive += 1
            else:
                false_negative += 1
        accuracy = correct / len(cases) if cases else 0.0
        results.append(
            {
                "threshold": threshold,
                "accuracy": round(accuracy, 3),
                "false_positive": false_positive,
                "false_negative": false_negative,
                "case_count": len(cases),
            }
        )
    best = max(results, key=lambda item: (item["accuracy"], -item["false_positive"], -item["false_negative"]))
    return {"best": best, "candidates": results}


def score_stats_by_expected(rows: list[dict], component: str) -> dict:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        label = row[f"expected_{component}"]
        grouped.setdefault(label, []).append(float(row[f"panns_{component}_score"]))
    return {label: describe_scores(scores) for label, scores in sorted(grouped.items())}


def describe_scores(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0}
    values = sorted(scores)
    return {
        "count": len(values),
        "min": round(values[0], 6),
        "median": round(percentile(values, 0.5), 6),
        "mean": round(sum(values) / len(values), 6),
        "max": round(values[-1], 6),
    }


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalise_key(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


if __name__ == "__main__":
    raise SystemExit(main())
