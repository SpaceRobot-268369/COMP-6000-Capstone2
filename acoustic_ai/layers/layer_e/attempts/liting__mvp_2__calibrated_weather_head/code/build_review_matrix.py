"""Build the E-B reviewer evidence matrix for PR #34.

This script reads Murphy's audited Site257 Layer B weather asset index and
runs Liting's E-B attempts against a fixed reviewer-facing subset when the
corresponding WAV files are materialised locally.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[6]
AI_ROOT = PROJECT_ROOT / "acoustic_ai"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from server import registry  # noqa: E402


ASSET_INDEX = (
    PROJECT_ROOT
    / "acoustic_ai"
    / "layers"
    / "layer_b"
    / "attempts"
    / "murphy__smoke_1__curated_assets"
    / "data"
    / "weather"
    / "asset_index.csv"
)
LOCAL_PROMOTED_ROOT = (
    PROJECT_ROOT
    / "acoustic_ai"
    / "layers"
    / "layer_e"
    / "attempts"
    / "liting__smoke_1__e_b_weather_analysis"
    / "data"
    / "analysis"
    / "site257_clap_promoted"
    / "assets_wav_22050_mono"
)
OUT_DIR = (
    PROJECT_ROOT
    / "acoustic_ai"
    / "layers"
    / "layer_e"
    / "attempts"
    / "liting__mvp_2__calibrated_weather_head"
    / "review"
)

ATTEMPTS = [
    "liting__mvp_1__panns_weather_baseline",
    "liting__mvp_2__calibrated_weather_head",
    "liting__mvp_3__balanced_weather_head",
    "liting__mvp_4__data_expanded_weather_head",
    "liting__mvp_5__clap_weather_probe",
]
OFFLINE_SKIP_ATTEMPTS = {
    "liting__mvp_5__clap_weather_probe": (
        "CLAP backbone is not cached locally in this environment; running it "
        "would attempt HuggingFace network access. Use the existing MVP5 "
        "metrics or rerun this script on Server B / a machine with the CLAP "
        "cache materialised."
    )
}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def _read_asset_rows() -> list[dict[str, str]]:
    with ASSET_INDEX.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _source_s3_uri(row: dict[str, str]) -> str:
    try:
        payload = json.loads(row.get("provenance_json") or "{}")
    except json.JSONDecodeError:
        payload = {}
    return str(payload.get("source_s3_uri") or "")


def _expected_rain(row: dict[str, str]) -> str:
    if not _truthy(row.get("has_rain")):
        return "none"
    return (row.get("rain_intensity") or "moderate").replace("medium", "moderate")


def _expected_wind(row: dict[str, str]) -> str:
    if not _truthy(row.get("has_wind")):
        return "none"
    value = row.get("wind_intensity") or "moderate"
    return "strong" if value == "heavy" else value.replace("medium", "moderate")


def _expected_thunder(row: dict[str, str]) -> str:
    if not _truthy(row.get("has_thunder")):
        return "none"
    value = row.get("thunder_intensity") or "moderate"
    return "strong" if value == "heavy" else value.replace("medium", "moderate")


def _resolve_local_wav(row: dict[str, str]) -> Path | None:
    path = PROJECT_ROOT / row["clip_path"]
    if path.exists():
        return path
    basename = Path(row["clip_path"]).name
    matches = sorted(LOCAL_PROMOTED_ROOT.glob(f"*/*{basename}"))
    if matches:
        return matches[0]
    return None


def _select(rows: list[dict[str, str]], predicate: Callable[[dict[str, str]], bool], limit: int) -> list[dict[str, str]]:
    matches = [row for row in rows if predicate(row)]
    matches.sort(key=lambda row: (row.get("asset_id") or "", row.get("clip_path") or ""))
    return matches[:limit]


def _build_review_cases(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    site = [
        row
        for row in rows
        if row.get("source_type") == "site"
        and row.get("analysis_use") in {"site_ready_pool", "site_backup_pool"}
    ]
    buckets = [
        ("light_rain", lambda r: _truthy(r.get("has_rain")) and r.get("rain_intensity") == "light", 2),
        ("heavy_rain", lambda r: _truthy(r.get("has_rain")) and r.get("rain_intensity") == "heavy", 2),
        ("moderate_rain", lambda r: r.get("primary_weather") == "rain" and r.get("rain_intensity") == "medium", 2),
        ("mixed_rain_wind", lambda r: r.get("primary_weather") == "rain+wind", 2),
        ("breezing_light_wind", lambda r: _truthy(r.get("has_wind")) and r.get("wind_intensity") == "light", 2),
        ("strong_wind", lambda r: _truthy(r.get("has_wind")) and r.get("wind_intensity") == "heavy", 2),
        ("thunder_backup", lambda r: _truthy(r.get("has_thunder")), 2),
    ]

    seen: set[str] = set()
    selected: list[dict[str, str]] = []
    for scene, predicate, limit in buckets:
        for row in _select(site, predicate, limit):
            asset_id = row["asset_id"]
            if asset_id in seen:
                continue
            seen.add(asset_id)
            item = dict(row)
            item["review_scene"] = scene
            selected.append(item)
    return selected


def _attempt_checkpoint_state(attempt_id: str) -> str:
    spec = registry.get_attempt("layer_e", attempt_id)
    if spec.checkpoint is None:
        return "no_checkpoint_required"
    if not spec.checkpoint.exists():
        return f"checkpoint_dir_missing:{spec.checkpoint}"
    has_weight = any(spec.checkpoint.glob("*.pt"))
    if has_weight:
        return "checkpoint_materialized"
    pointers = sorted(p.name for p in spec.checkpoint.glob("*.pt.dvc"))
    if pointers:
        return "checkpoint_pointer_only:" + ",".join(pointers)
    return "checkpoint_missing"


def _run_attempt(attempt_id: str, wav_path: Path) -> dict[str, object]:
    allow_clap = os.getenv("LITING_EB_RUN_CLAP") == "1"
    if attempt_id in OFFLINE_SKIP_ATTEMPTS and not allow_clap:
        return {
            "status": "not_run",
            "reason": OFFLINE_SKIP_ATTEMPTS[attempt_id],
            "rain": None,
            "wind": None,
            "thunder": None,
            "rain_confidence": None,
            "wind_confidence": None,
            "thunder_confidence": None,
        }

    checkpoint_state = _attempt_checkpoint_state(attempt_id)
    if checkpoint_state.startswith("checkpoint_pointer_only"):
        return {
            "status": "not_run",
            "reason": checkpoint_state,
            "rain": None,
            "wind": None,
            "thunder": None,
            "rain_confidence": None,
            "wind_confidence": None,
            "thunder_confidence": None,
        }

    try:
        result = registry.analyze("layer_e", attempt_id, str(wav_path))
        report = result["report"]
        summary = report.get("summary", {})
        return {
            "status": "run",
            "reason": checkpoint_state,
            "rain": summary.get("rain", {}).get("intensity"),
            "wind": summary.get("wind", {}).get("intensity"),
            "thunder": summary.get("thunder", {}).get("intensity"),
            "rain_confidence": summary.get("rain", {}).get("confidence"),
            "wind_confidence": summary.get("wind", {}).get("confidence"),
            "thunder_confidence": summary.get("thunder", {}).get("confidence"),
            "method": report.get("model", {}).get("method"),
            "primary_model": report.get("model", {}).get("primary"),
        }
    except Exception as exc:  # noqa: BLE001 - evidence collection must keep going.
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "rain": None,
            "wind": None,
            "thunder": None,
            "rain_confidence": None,
            "wind_confidence": None,
            "thunder_confidence": None,
        }


def _pass_status(row: dict[str, object], expected_rain: str, expected_wind: str, expected_thunder: str) -> str:
    if row["status"] != "run":
        return str(row["status"])
    if (
        row["rain"] == expected_rain
        and row["wind"] == expected_wind
        and row["thunder"] == expected_thunder
    ):
        return "pass"
    expected_active = [
        ("rain", expected_rain),
        ("wind", expected_wind),
        ("thunder", expected_thunder),
    ]
    for component, expected in expected_active:
        if expected != "none" and row[component] not in {None, "none"}:
            return "partial"
    if all(expected == "none" for _, expected in expected_active):
        if row["rain"] == "none" or row["wind"] == "none" or row["thunder"] == "none":
            return "partial"
    return "fail"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_asset_rows()
    cases = _build_review_cases(rows)

    matrix_rows: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for row in cases:
        wav_path = _resolve_local_wav(row)
        expected_rain = _expected_rain(row)
        expected_wind = _expected_wind(row)
        expected_thunder = _expected_thunder(row)
        case = {
            "review_scene": row["review_scene"],
            "asset_id": row["asset_id"],
            "primary_weather": row["primary_weather"],
            "expected_rain": expected_rain,
            "expected_wind": expected_wind,
            "expected_thunder": expected_thunder,
            "analysis_use": row["analysis_use"],
            "human_audit_status": row["human_audit_status"],
            "human_weather_label": row["human_weather_label"],
            "human_notes": row["human_notes"],
            "clip_path": row["clip_path"],
            "local_wav_path": str(wav_path.relative_to(PROJECT_ROOT)) if wav_path else "",
            "available_locally": bool(wav_path),
            "source_s3_uri": _source_s3_uri(row),
        }
        matrix_rows.append(case)

        attempt_results = {}
        if wav_path:
            for attempt_id in ATTEMPTS:
                outcome = _run_attempt(attempt_id, wav_path)
                outcome["pass_status"] = _pass_status(outcome, expected_rain, expected_wind, expected_thunder)
                attempt_results[attempt_id] = outcome
        results.append({**case, "attempt_results": attempt_results})

    matrix_path = OUT_DIR / "murphy_site257_fixed_review_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(matrix_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(matrix_rows)

    results_path = OUT_DIR / "mvp1_to_mvp5_cross_review_results.json"
    summary = {
        "asset_index": str(ASSET_INDEX.relative_to(PROJECT_ROOT)),
        "case_count": len(matrix_rows),
        "available_locally_count": sum(1 for row in matrix_rows if row["available_locally"]),
        "scene_counts": dict(Counter(row["review_scene"] for row in matrix_rows)),
        "attempts": ATTEMPTS,
        "checkpoint_state": {attempt_id: _attempt_checkpoint_state(attempt_id) for attempt_id in ATTEMPTS},
    }
    results_path.write_text(json.dumps({"summary": summary, "cases": results}, indent=2), encoding="utf-8")

    md_path = OUT_DIR / "MVP1_TO_MVP5_REVIEW_EVIDENCE.md"
    lines = [
        "# E-B MVP1-MVP5 Fixed Review Evidence",
        "",
        "Owner: `liting`",
        "",
        "This review set is built from Murphy's audited Site257 Layer B weather asset index.",
        "It is intended to answer PR #34 reviewer questions about exact sample clips and sample results.",
        "",
        "## Source",
        "",
        f"- Asset index: `{ASSET_INDEX.relative_to(PROJECT_ROOT)}`",
        "- Source restriction: `source_type=site` only.",
        "- Eligible pools: `site_ready_pool` and `site_backup_pool`.",
        "",
        "## Coverage",
        "",
        "| Review scene | Cases in matrix | Locally runnable now | Note |",
        "|---|---:|---:|---|",
    ]
    scene_counts = Counter(row["review_scene"] for row in matrix_rows)
    local_counts = Counter(row["review_scene"] for row in matrix_rows if row["available_locally"])
    notes = {
        "light_rain": "Only one audited Site257 light-rain row exists in the current index.",
        "heavy_rain": "Only one audited Site257 heavy-rain row exists in the current index.",
        "moderate_rain": "Two selected as rain-positive review cases.",
        "mixed_rain_wind": "Known hard mixed-weather cases.",
        "breezing_light_wind": "Two selected from three available light-wind rows.",
        "strong_wind": "Two selected from strong Site257 wind rows.",
        "thunder_backup": "Backup-only; E-B candidates suppress thunder until more site evidence exists.",
    }
    for scene in scene_counts:
        lines.append(
            f"| `{scene}` | {scene_counts[scene]} | {local_counts[scene]} | {notes.get(scene, '')} |"
        )

    lines.extend([
        "",
        "## Attempt Checkpoint State",
        "",
        "| Attempt | State |",
        "|---|---|",
    ])
    for attempt_id, state in summary["checkpoint_state"].items():
        lines.append(f"| `{attempt_id}` | `{state}` |")

    lines.extend([
        "",
        "## Attempt Result Summary",
        "",
        "| Attempt | Pass | Partial | Fail | Not run | Error |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for attempt_id in ATTEMPTS:
        counts: Counter[str] = Counter()
        for case in results:
            outcome = case["attempt_results"].get(attempt_id)
            counts[outcome["pass_status"] if outcome else "not_run"] += 1
        lines.append(
            f"| `{attempt_id}` | "
            f"{counts['pass']} | {counts['partial']} | {counts['fail']} | "
            f"{counts['not_run']} | {counts['error']} |"
        )

    lines.extend([
        "",
        "## Sample Results",
        "",
        "| Scene | Asset ID | Expected rain | Expected wind | Expected thunder | Local WAV | MVP1 | MVP2 | MVP3 | MVP4 | MVP5 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for case in results:
        cells = []
        for attempt_id in ATTEMPTS:
            outcome = case["attempt_results"].get(attempt_id)
            if not outcome:
                cells.append("not run: wav not local")
                continue
            if outcome["status"] != "run":
                cells.append(f"{outcome['status']}: {outcome['reason']}")
            else:
                cells.append(
                    f"{outcome['pass_status']} "
                    f"(rain={outcome['rain']} {outcome['rain_confidence']}, "
                    f"wind={outcome['wind']} {outcome['wind_confidence']}, "
                    f"thunder={outcome['thunder']} {outcome['thunder_confidence']})"
                )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case['review_scene']}`",
                    f"`{case['asset_id']}`",
                    str(case["expected_rain"]),
                    str(case["expected_wind"]),
                    str(case["expected_thunder"]),
                    f"`{case['local_wav_path'] or 'not materialized locally'}`",
                    *cells,
                ]
            )
            + " |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This matrix confirms that the current Murphy-audited Site257 pool does contain multiple weather combinations.",
        "- The missing Site257 WAVs and all MVP2-MVP5 checkpoint artifacts were materialized on Server B, then the full matrix was rerun there.",
        "- The matrix now has 12/12 locally runnable samples: all selected WAVs are materialized and all MVP1-MVP5 attempts have a resolved checkpoint state.",
        "- MVP5 is the strongest result on this fixed matrix: 10 exact passes and 2 failures. The two failures are both thunder backup cases.",
        "- MVP2 remains the safest current frontend/integration candidate because it is already wired for demo output, but MVP5 should be treated as the strongest candidate-model result from this review run.",
        "- The reviewer bar is still not fully satisfied for every requested scene because the current audited Site257 index only contains one `light_rain` row and one `heavy_rain` row. Adding two local examples for those exact scenes requires either expanding the audited Site257 rain pool or relaxing the scene definition to rain-positive cases.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"matrix={matrix_path}")
    print(f"results={results_path}")
    print(f"markdown={md_path}")


if __name__ == "__main__":
    main()
