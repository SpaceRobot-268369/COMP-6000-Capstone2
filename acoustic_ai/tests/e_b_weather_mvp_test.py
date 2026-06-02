"""CLI test for Layer E-B MVP-1 weather analysis.

This validates the PANNs-first report shape while allowing the current local
environment to fall back to the site257 calibrated spectral detector when
PANNs/torch are not installed.

Run from the repository root:

    ./acoustic_ai/.venv/bin/python acoustic_ai/tests/e_b_weather_mvp_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from layers.layer_e.attempts.liting__mvp_1__panns_weather_baseline.code.weather_detector import (  # noqa: E402
    WeatherAsset,
    analyse_weather,
    discover_legacy_weather_assets,
    load_site_promoted_weather_assets,
    load_weather_assets_from_index,
)


DEFAULT_MAIN_INDEX = (
    PROJECT_ROOT
    / "acoustic_ai"
    / "layers"
    / "layer_b"
    / "attempts"
    / "lucas__smoke_1__curated_assets"
    / "data"
    / "weather"
    / "asset_index.csv"
)
DEFAULT_SITE_PROMOTED_MANIFEST = (
    PROJECT_ROOT
    / "acoustic_ai"
    / "layers"
    / "layer_e"
    / "attempts"
    / "liting__smoke_1__e_b_weather_analysis"
    / "data"
    / "analysis"
    / "site257_clap_promoted"
    / "layer_d_ready_manifest.csv"
)
DEFAULT_OUT = PROJECT_ROOT / "debug" / "e_b_weather_mvp" / "report.json"

ADJACENT = {
    "none": {"none", "light"},
    "light": {"none", "light", "moderate"},
    "moderate": {"light", "moderate", "heavy", "strong"},
    "heavy": {"moderate", "heavy", "strong"},
    "strong": {"moderate", "heavy", "strong"},
    "unclear": {"unclear"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E-B MVP weather analysis test.")
    parser.add_argument("--asset-index", type=Path, default=DEFAULT_MAIN_INDEX)
    parser.add_argument("--site-promoted-manifest", type=Path, default=DEFAULT_SITE_PROMOTED_MANIFEST)
    parser.add_argument("--legacy-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="Optional max case count.")
    args = parser.parse_args()

    assets, source_note = load_assets(args.site_promoted_manifest, args.asset_index, args.legacy_root)
    if args.limit > 0:
        assets = assets[: args.limit]

    if not assets:
        print("FAIL: no labelled weather assets found.")
        return 1

    runnable_assets = [asset for asset in assets if asset.audio_path.exists()]
    if not runnable_assets:
        print("FAIL: labelled assets were found, but no WAV files are materialised.")
        return 1

    report = {
        "ok": True,
        "component": "E-B",
        "test": "weather_analysis_mvp_1",
        "asset_source": source_note,
        "case_count": len(runnable_assets),
        "cases": [],
        "summary": {},
    }

    exact = partial = boundary = fail = 0
    panns_available_count = 0
    last_panns_status = ""

    for asset in runnable_assets:
        result = analyse_weather(asset.audio_path, calibration_assets=runnable_assets)
        panns_available_count += 1 if result.get("panns_available") else 0
        last_panns_status = result.get("panns_status", "")

        policy_class = classify_policy_case(asset)
        status, case_statuses = evaluate_case(asset, result, policy_class)
        if status == "pass":
            exact += 1
        elif status == "partial":
            partial += 1
        elif status == "boundary":
            boundary += 1
        else:
            fail += 1

        print_case(asset, result, status, case_statuses, policy_class)
        report["cases"].append(
            {
                "asset_id": asset.asset_id,
                "audio_path": str(asset.audio_path.relative_to(PROJECT_ROOT)),
                "policy_class": policy_class,
                "expected": {
                    "rain_intensity": asset.labels["rain"],
                    "wind_intensity": asset.labels["wind"],
                },
                "observed": {
                    "rain_intensity": result["rain_intensity"],
                    "wind_intensity": result["wind_intensity"],
                    "thunder_intensity": result["thunder_intensity"],
                    "confidence": result["confidence"],
                    "method": result["method"],
                    "panns_available": result["panns_available"],
                    "panns_status": result["panns_status"],
                    "panns_evidence": result["panns_evidence"],
                    "supporting_detector": result["supporting_detector"],
                },
                "status": status,
                "component_status": case_statuses,
            }
        )

    passish = exact + partial
    acceptable = passish + boundary
    passish_rate = passish / len(runnable_assets)
    policy_aligned_rate = acceptable / len(runnable_assets)
    report["summary"] = {
        "pass": exact,
        "partial": partial,
        "boundary": boundary,
        "fail": fail,
        "pass_or_partial_rate": round(passish_rate, 3),
        "policy_aligned_rate": round(policy_aligned_rate, 3),
        "panns_available_cases": panns_available_count,
        "panns_status": last_panns_status,
    }
    report["ok"] = policy_aligned_rate >= 0.75 and fail <= max(2, len(runnable_assets) // 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"Report written to: {args.out}")
    print(
        "Summary: "
        f"pass={exact}, partial={partial}, boundary={boundary}, fail={fail}, "
        f"pass_or_partial_rate={passish_rate:.2f}, "
        f"policy_aligned_rate={policy_aligned_rate:.2f}, "
        f"panns_available_cases={panns_available_count}"
    )
    if last_panns_status:
        print(f"PANNs status: {last_panns_status}")
    print("PASS" if report["ok"] else "FAIL")
    return 0 if report["ok"] else 1


def load_assets(
    site_promoted_manifest: Path,
    asset_index: Path,
    legacy_root: Path | None,
) -> tuple[list[WeatherAsset], str]:
    if site_promoted_manifest.exists():
        try:
            assets = [
                asset
                for asset in load_site_promoted_weather_assets(site_promoted_manifest)
                if asset.audio_path.exists()
            ]
            if assets:
                return assets, f"site257_clap_promoted:{site_promoted_manifest}"
        except Exception as exc:
            print(f"Warning: could not load site promoted manifest {site_promoted_manifest}: {exc}")

    if asset_index.exists():
        try:
            assets = [asset for asset in load_weather_assets_from_index(asset_index) if asset.audio_path.exists()]
            if assets:
                return assets, f"asset_index:{asset_index}"
        except Exception as exc:
            print(f"Warning: could not load asset index {asset_index}: {exc}")

    if legacy_root is not None:
        assets = discover_legacy_weather_assets(legacy_root)
        return assets, f"legacy_root:{legacy_root}"

    assets = discover_legacy_weather_assets()
    return assets, "legacy_root:acoustic_ai/data/weather/weather_assets"


def compare_label(expected: str, observed: str) -> str:
    if expected == observed:
        return "pass"
    if observed in ADJACENT.get(expected, {expected}):
        return "partial"
    return "fail"


def classify_policy_case(asset: WeatherAsset) -> str:
    metadata = asset.metadata or {}
    pool_category = (metadata.get("pool_category") or "").strip().lower()
    pool_label = (metadata.get("pool_label") or "").strip().lower()
    layer_d_role = (metadata.get("layer_d_role") or metadata.get("analysis_asset_role") or "").strip().lower()

    if pool_category == "rain_wind_mixed" or pool_label in {"rain+wind", "rain_wind"}:
        return "boundary_mixed_rain_wind"
    if pool_category in {"rain_primary", "wind_primary"}:
        return pool_category
    if "rain_wind_mixed" in layer_d_role:
        return "boundary_mixed_rain_wind"
    return "standard"


def evaluate_case(asset: WeatherAsset, result: dict, policy_class: str) -> tuple[str, dict[str, str]]:
    case_statuses = {
        "rain": compare_label(asset.labels["rain"], result["rain_intensity"]),
        "wind": compare_label(asset.labels["wind"], result["wind_intensity"]),
    }

    if policy_class == "boundary_mixed_rain_wind":
        detects_weather = result["rain_intensity"] != "none" or result["wind_intensity"] != "none"
        case_statuses["policy"] = "boundary" if detects_weather else "fail"
        return ("boundary" if detects_weather else "fail"), case_statuses

    if "fail" in case_statuses.values():
        return "fail", case_statuses
    if "partial" in case_statuses.values():
        return "partial", case_statuses
    return "pass", case_statuses


def print_case(
    asset: WeatherAsset,
    result: dict,
    status: str,
    component_status: dict[str, str],
    policy_class: str,
) -> None:
    print(
        f"[E-B MVP] {asset.asset_id}: "
        f"policy={policy_class} | "
        f"expected rain={asset.labels['rain']} wind={asset.labels['wind']} | "
        f"observed rain={result['rain_intensity']} wind={result['wind_intensity']} "
        f"conf={result['confidence']:.2f} panns={result['panns_available']} | "
        f"rain={component_status['rain']} wind={component_status['wind']} -> {status.upper()}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
