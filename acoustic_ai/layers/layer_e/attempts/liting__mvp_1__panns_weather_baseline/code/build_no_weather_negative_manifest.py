"""Build local no-weather negative proxy manifest for E-B MVP validation.

The full Site257 ambient clip DVC payload is not always materialised locally.
This script therefore looks for already-materialised Site257 event/reference
clips and keeps only files that the current E-B detector reports as no rain and
no wind. These are proxy negatives, not final ambient holdout data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from layers.layer_e.attempts.liting__mvp_1__panns_weather_baseline.code.weather_detector import (  # noqa: E402
    analyse_weather,
)
from tests.e_b_weather_mvp_test import (  # noqa: E402
    DEFAULT_MAIN_INDEX,
    DEFAULT_SITE_PROMOTED_MANIFEST,
    load_assets,
)


ATTEMPT_DIR = PROJECT_ROOT / "acoustic_ai" / "layers" / "layer_e" / "attempts" / "liting__mvp_1__panns_weather_baseline"
DEFAULT_OUT = ATTEMPT_DIR / "data" / "no_weather_negative_manifest.csv"
DEFAULT_CANDIDATE_GLOBS = (
    "resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/**/*.wav",
    "acoustic_ai/data/events/layer_c_sa3_horsfields_bronze_cuckoo_core6_smoke/**/*.wav",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build E-B no-weather negative proxy manifest.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--candidate-glob", action="append", default=[])
    args = parser.parse_args()

    positive_assets, _source_note = load_assets(DEFAULT_SITE_PROMOTED_MANIFEST, DEFAULT_MAIN_INDEX, legacy_root=None)
    positive_assets = [asset for asset in positive_assets if asset.audio_path.exists()]
    if not positive_assets:
        print("FAIL: positive calibration assets are not materialised.")
        return 1

    globs = args.candidate_glob or list(DEFAULT_CANDIDATE_GLOBS)
    candidates = discover_candidates(globs)
    if not candidates:
        print("FAIL: no local no-weather proxy candidates found.")
        return 1

    accepted = []
    for path in candidates:
        result = analyse_weather(path, calibration_assets=positive_assets)
        panns = result.get("panns_evidence", {}).get("component_scores", {})
        no_weather = result["rain_intensity"] == "none" and result["wind_intensity"] == "none"
        print(
            f"[negative-candidate] {path.name}: "
            f"rain={result['rain_intensity']} wind={result['wind_intensity']} "
            f"panns_rain={panns.get('rain', 0.0):.4f} panns_wind={panns.get('wind', 0.0):.4f} "
            f"-> {'KEEP' if no_weather else 'reject'}"
        )
        if not no_weather:
            continue
        accepted.append(
            {
                "asset_id": f"no_weather_proxy__{path.stem}",
                "audio_path": str(path.relative_to(PROJECT_ROOT)),
                "expected_rain": "none",
                "expected_wind": "none",
                "policy_class": "no_weather_negative",
                "source_note": "local Site257 Layer C/event proxy; replace with ambient no-weather holdout when DVC clips are materialised",
                "panns_rain_score": panns.get("rain", 0.0),
                "panns_wind_score": panns.get("wind", 0.0),
            }
        )
        if len(accepted) >= args.limit:
            break

    if not accepted:
        print("FAIL: candidates were found, but none passed the no-weather filter.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(accepted[0].keys()))
        writer.writeheader()
        writer.writerows(accepted)

    print()
    print(f"Negative manifest written to: {args.out}")
    print(f"Accepted negatives: {len(accepted)}")
    return 0


def discover_candidates(globs: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for pattern in globs:
        candidates.extend(PROJECT_ROOT.glob(pattern))

    by_stem: dict[str, Path] = {}
    for path in sorted({path for path in candidates if path.is_file()}, key=candidate_sort_key):
        by_stem.setdefault(path.stem, path)
    return list(by_stem.values())


def candidate_sort_key(path: Path) -> tuple[int, str]:
    rel = str(path.relative_to(PROJECT_ROOT))
    return (0 if rel.startswith("resources/site_257_bowra-dry-a/") else 1, rel)


if __name__ == "__main__":
    raise SystemExit(main())
