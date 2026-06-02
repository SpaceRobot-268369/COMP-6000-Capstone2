"""CLI smoke test for Layer E-B weather analysis.

This is intentionally lightweight: it validates the E-B contract with labelled
Layer B weather assets, without training a new classifier or requiring Server B.

Run from the repository root:

    ./acoustic_ai/.venv/bin/python acoustic_ai/tests/e_b_weather_smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from layers.layer_e.attempts.liting__smoke_1__e_b_weather_analysis.code.weather_detector import (  # noqa: E402
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
DEFAULT_OUT = PROJECT_ROOT / "debug" / "e_b_weather_smoke" / "report.json"

ADJACENT = {
    "none": {"none", "light"},
    "light": {"none", "light", "moderate"},
    "moderate": {"light", "moderate", "heavy", "strong"},
    "heavy": {"moderate", "heavy", "strong"},
    "strong": {"moderate", "heavy", "strong"},
    "unclear": {"unclear"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E-B weather analysis smoke test.")
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
        print(f"Tried asset index: {args.asset_index}")
        print("Fallback legacy root: acoustic_ai/data/weather/weather_assets")
        return 1

    runnable_assets = [asset for asset in assets if asset.audio_path.exists()]
    if not runnable_assets:
        print("FAIL: labelled assets were found, but no WAV files are materialised.")
        print("Run dvc pull for the Layer B weather assets, or use the legacy local assets.")
        return 1

    report = {
        "ok": True,
        "component": "E-B",
        "test": "weather_analysis_smoke",
        "asset_source": source_note,
        "case_count": len(runnable_assets),
        "calibration_note": (
            "Smoke baseline uses labelled Layer B / site-weather assets as calibration "
            "references; this is not classifier training."
        ),
        "cases": [],
        "summary": {},
    }

    exact = partial = fail = 0
    for asset in runnable_assets:
        result = analyse_weather(asset.audio_path, calibration_assets=runnable_assets)
        case_statuses = {
            "rain": compare_label(asset.labels["rain"], result["rain_intensity"]),
            "wind": compare_label(asset.labels["wind"], result["wind_intensity"]),
        }

        if "fail" in case_statuses.values():
            status = "fail"
            fail += 1
        elif "partial" in case_statuses.values():
            status = "partial"
            partial += 1
        else:
            status = "pass"
            exact += 1

        print_case(asset, result, status, case_statuses)
        report["cases"].append(
            {
                "asset_id": asset.asset_id,
                "audio_path": str(asset.audio_path.relative_to(PROJECT_ROOT)),
                "expected": {
                    "rain_intensity": asset.labels["rain"],
                    "wind_intensity": asset.labels["wind"],
                },
                "observed": {
                    "rain_intensity": result["rain_intensity"],
                    "wind_intensity": result["wind_intensity"],
                    "thunder_intensity": result["thunder_intensity"],
                    "confidence": result["confidence"],
                    "component_confidence": result["component_confidence"],
                },
                "status": status,
                "component_status": case_statuses,
                "features": result["features"],
                "limitations": result["limitations"],
            }
        )

    passish = exact + partial
    passish_rate = passish / len(runnable_assets)
    report["summary"] = {
        "pass": exact,
        "partial": partial,
        "fail": fail,
        "pass_or_partial_rate": round(passish_rate, 3),
    }
    report["ok"] = passish_rate >= 0.75 and fail <= max(2, len(runnable_assets) // 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"Report written to: {args.out}")
    print(
        "Summary: "
        f"pass={exact}, partial={partial}, fail={fail}, "
        f"pass_or_partial_rate={passish_rate:.2f}"
    )
    print("PASS" if report["ok"] else "FAIL")
    return 0 if report["ok"] else 1


def load_assets(
    site_promoted_manifest: Path,
    asset_index: Path,
    legacy_root: Path | None,
) -> tuple[list[WeatherAsset], str]:
    assets: list[WeatherAsset] = []
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


def print_case(asset: WeatherAsset, result: dict, status: str, component_status: dict[str, str]) -> None:
    print(
        f"[E-B smoke] {asset.asset_id}: "
        f"expected rain={asset.labels['rain']} wind={asset.labels['wind']} | "
        f"observed rain={result['rain_intensity']} wind={result['wind_intensity']} "
        f"conf={result['confidence']:.2f} | "
        f"rain={component_status['rain']} wind={component_status['wind']} -> {status.upper()}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
