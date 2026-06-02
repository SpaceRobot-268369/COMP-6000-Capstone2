#!/usr/bin/env python3
"""Build a Layer D site-weather candidate pool from CLAP retrieval manifests.

This runs after CLAP retrieval. It does not create new audio windows; it applies
the manual-audit-calibrated policy to scored windows and writes a curated pool
manifest for later Layer D retrieval/mixing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


POLICY_VERSION = "site_weather_candidate_pool_v0.3"
LAYER_D_TARGET_SAMPLE_RATE_HZ = 22050
CLIPPING_REJECT_THRESHOLD = 0.005


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except ValueError:
        return default


def intensity_hint(row: dict[str, str], label: str) -> str:
    rain = parse_float(row, "precipitation_mm")
    wind = parse_float(row, "wind_speed_ms")
    wind_max = parse_float(row, "wind_max_ms")
    if label == "rain":
        if rain >= 5:
            return "heavy"
        if rain >= 1:
            return "medium"
        return "light"
    if label == "wind":
        wind_signal = max(wind, wind_max)
        if wind_signal >= 6:
            return "heavy"
        if wind_signal >= 3:
            return "medium"
        return "light"
    if label == "rain+wind":
        return "mixed"
    return ""


def classify_row(row: dict[str, str]) -> dict[str, str]:
    target = row.get("retrieval_target", "")
    gate = row.get("gate_status", "")
    clap_label = row.get("clap_weather_label", "")
    contamination_label = row.get("contamination_label", "")
    contamination_score = parse_float(row, "contamination_score")
    target_score = parse_float(row, "target_clap_score", parse_float(row, "clap_weather_score"))
    rain_score = parse_float(row, "clap_rain_score")
    wind_score = parse_float(row, "clap_wind_score")
    thunder_score = parse_float(row, "clap_thunder_score")
    weather_margin = parse_float(row, "weather_margin")
    target_vs_other = parse_float(row, "target_vs_other_weather_margin")
    env_prior = parse_float(row, "target_env_prior", parse_float(row, "env_prior_for_clap_label"))
    clipping_ratio = parse_float(row, "analysis_clipping_ratio")

    result = {
        "pool_decision": "reject",
        "pool_category": "reject",
        "pool_label": "",
        "pool_intensity_hint": "",
        "pool_reason": "",
        "manual_review_priority": "low",
        "layer_d_ready": "false",
        "layer_d_export_action": "do_not_export",
        "layer_d_target_sample_rate_hz": str(LAYER_D_TARGET_SAMPLE_RATE_HZ),
        "layer_d_target_channels": "1",
        "layer_d_recommended_format": "wav",
    }

    bio_close = contamination_label == "bird_or_insect" and contamination_score >= 0.32
    bio_dominant = contamination_label == "bird_or_insect" and weather_margin < 0.05

    if target == "thunder":
        result["pool_reason"] = "site_thunder_not_reliable_in_mvp"
        if max(rain_score, wind_score) >= 0.35 and weather_margin >= -0.05:
            result.update(
                {
                    "pool_decision": "backup",
                    "pool_category": "storm_rain_wind_backup",
                    "pool_label": "rain+wind" if rain_score >= 0.30 else "wind",
                    "manual_review_priority": "medium",
                }
            )
        result["pool_intensity_hint"] = intensity_hint(row, result["pool_label"])
        return result

    if target == "rain":
        if (
            rain_score >= 0.35
            and target_vs_other >= 0.10
            and weather_margin >= 0.14
            and not bio_close
        ):
            result.update(
                {
                    "pool_decision": "accept",
                    "pool_category": "rain_primary",
                    "pool_label": "rain",
                    "pool_reason": "clear_rain_target_score",
                    "manual_review_priority": "sample",
                }
            )
        elif (
            rain_score >= 0.36
            and target_vs_other >= -0.08
            and weather_margin >= 0.08
            and wind_score >= rain_score
            and not bio_close
        ):
            result.update(
                {
                    "pool_decision": "accept",
                    "pool_category": "rain_wind_mixed",
                    "pool_label": "rain+wind",
                    "pool_reason": "site_rain_wind_mixed_texture_not_pure_rain",
                    "manual_review_priority": "sample",
                }
            )
        elif rain_score >= 0.32 and weather_margin >= 0.00 and not bio_dominant:
            result.update(
                {
                    "pool_decision": "backup",
                    "pool_category": "rain_backup_maybe",
                    "pool_label": "rain",
                    "pool_reason": "rain_possible_but_needs_review_or_backup_use",
                    "manual_review_priority": "medium",
                }
            )
        else:
            result["pool_reason"] = "rain_too_weak_or_contaminated"

    elif target == "wind":
        if (
            wind_score >= 0.43
            and target_vs_other >= 0.05
            and weather_margin >= 0.12
            and not bio_close
        ):
            result.update(
                {
                    "pool_decision": "accept",
                    "pool_category": "wind_primary",
                    "pool_label": "wind",
                    "pool_reason": "clear_wind_target_score",
                    "manual_review_priority": "sample",
                }
            )
        elif wind_score >= 0.40 and weather_margin >= 0.05:
            category = "wind_with_bio_backup" if bio_close else "wind_backup_maybe"
            result.update(
                {
                    "pool_decision": "backup",
                    "pool_category": category,
                    "pool_label": "wind",
                    "pool_reason": "wind_present_but_not_clean",
                    "manual_review_priority": "medium",
                }
            )
        elif wind_score >= 0.34 and weather_margin >= 0.03 and env_prior >= 0.55:
            result.update(
                {
                    "pool_decision": "backup",
                    "pool_category": "wind_weak_backup",
                    "pool_label": "wind",
                    "pool_reason": "weak_wind_possible",
                    "manual_review_priority": "low",
                }
            )
        else:
            result["pool_reason"] = "wind_too_weak_or_contaminated"

    else:
        if gate == "candidate" and clap_label in {"rain", "wind"}:
            result.update(
                {
                    "pool_decision": "backup",
                    "pool_category": f"{clap_label}_unbalanced_backup",
                    "pool_label": clap_label,
                    "pool_reason": "legacy_unbalanced_candidate",
                    "manual_review_priority": "low",
                }
            )
        else:
            result["pool_reason"] = "unsupported_target_or_rejected_gate"

    if result["pool_decision"] in {"accept", "backup"} and clipping_ratio >= CLIPPING_REJECT_THRESHOLD:
        result.update(
            {
                "pool_decision": "reject",
                "pool_category": "reject",
                "pool_label": "",
                "pool_reason": "clipping_or_overload_risk",
                "manual_review_priority": "low",
            }
        )

    if result["pool_decision"] in {"accept", "backup"}:
        result["layer_d_export_action"] = "export_22050hz_wav_after_spot_check"

    result["pool_intensity_hint"] = intensity_hint(row, result["pool_label"])
    return result


def add_policy_fields(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        classified = classify_row(row)
        output.append({**row, **classified, "candidate_pool_policy_version": POLICY_VERSION})
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    summary: dict[str, object] = {
        "policy_version": POLICY_VERSION,
        "total_rows": len(rows),
        "pool_decision_counts": {},
        "pool_category_counts": {},
        "pool_label_counts": {},
        "manual_review_priority_counts": {},
    }
    for row in rows:
        for field, key in [
            ("pool_decision", "pool_decision_counts"),
            ("pool_category", "pool_category_counts"),
            ("pool_label", "pool_label_counts"),
            ("manual_review_priority", "manual_review_priority_counts"),
        ]:
            value = row.get(field, "")
            counts = summary[key]  # type: ignore[index]
            counts[value] = counts.get(value, 0) + 1
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def sample_for_review(rows: list[dict[str, str]], per_category: int) -> list[dict[str, str]]:
    sampled: list[dict[str, str]] = []
    categories = sorted({row["pool_category"] for row in rows})
    for category in categories:
        category_rows = [row for row in rows if row["pool_category"] == category]
        category_rows.sort(
            key=lambda row: (
                row.get("manual_review_priority") != "sample",
                -parse_float(row, "final_score"),
            )
        )
        sampled.extend(category_rows[:per_category])
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-sample-per-category", type=int, default=20)
    args = parser.parse_args()

    rows = add_policy_fields(read_csv(args.retrieval_manifest))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_pool_manifest.csv", rows)
    write_summary(args.output_dir / "summary.json", rows)
    write_csv(
        args.output_dir / "manual_review_sample.csv",
        sample_for_review(rows, args.review_sample_per_category),
    )
    (args.output_dir / "policy_version.txt").write_text(POLICY_VERSION + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} candidate rows to {args.output_dir}")


if __name__ == "__main__":
    main()
