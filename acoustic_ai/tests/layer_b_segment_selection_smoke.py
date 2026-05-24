"""Smoke test for Layer B weather segment selection.

This checks the Layer B handoff contract: retrieve weather assets, select
candidate segments, validate basic audio quality, and export clips for manual
listening. It does not test Layer D mixing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "acoustic_ai"))

from modules.weather.segment_selector import select_weather_segments


CASES = [
    {
        "name": "wind",
        "query": "strong natural forest wind ambience",
        "weather_types": ["wind"],
        "wind_speed_ms": 9.0,
        "precipitation_mm": 0.0,
    },
    {
        "name": "rain",
        "query": "light drizzle under forest canopy",
        "weather_types": ["rain"],
        "wind_speed_ms": 1.0,
        "precipitation_mm": 1.5,
    },
    {
        "name": "thunder",
        "query": "distant rolling thunderstorm ambience",
        "weather_types": ["thunder"],
        "include_thunder": True,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Layer B segment selection smoke test.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT
        / "model"
        / "candidates"
        / "murphy"
        / "layer-b-segment-selection-smoke"
        / "outputs",
        help="Directory for JSON report and exported candidate clips.",
    )
    parser.add_argument("--top-assets", type=int, default=3)
    parser.add_argument("--segments-per-type", type=int, default=2)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "ok": True,
        "cases": [],
        "clip_dir": str(args.out_dir),
    }
    failures = []

    for case in CASES:
        print(f"\n[Layer B smoke] {case['name']}: {case['query']}")
        result = select_weather_segments(
            query=case["query"],
            weather_types=case["weather_types"],
            wind_speed_ms=case.get("wind_speed_ms"),
            precipitation_mm=case.get("precipitation_mm"),
            include_thunder=case.get("include_thunder", False),
            target_duration=30.0,
            top_assets=args.top_assets,
            segments_per_type=args.segments_per_type,
            window_seconds=args.window_seconds,
            overlap_seconds=args.overlap_seconds,
        )

        case_failures = validate_case(case, result, args.out_dir)
        failures.extend(case_failures)

        print_case_summary(result, case_failures)
        report["cases"].append({
            "name": case["name"],
            "request": case,
            "result": result,
            "failures": case_failures,
        })

    if failures:
        report["ok"] = False

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nReport: {report_path}")
    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPASS")
    print("Listen to the exported WAV clips in the output directory to confirm semantic quality.")
    return 0


def validate_case(case: dict, result: dict, out_dir: Path) -> list[str]:
    expected_type = case["weather_types"][0]
    failures = []

    if result.get("warnings"):
        failures.append(f"{case['name']} returned warnings: {result['warnings']}")

    segments = [
        item for item in result.get("results", [])
        if item.get("weather_type") == expected_type
    ]
    if not segments:
        return failures + [f"{case['name']} returned no {expected_type} segments."]

    for index, item in enumerate(segments):
        failures.extend(validate_segment(case["name"], expected_type, index, item))
        if item.get("validation", {}).get("asset_available") is not False:
            export_segment_clip(case["name"], index, item, out_dir)

    return failures


def validate_segment(case_name: str, expected_type: str, index: int, item: dict) -> list[str]:
    failures = []
    prefix = f"{case_name} segment {index}"

    file_path = Path(item["file"])
    resolved_path = file_path if file_path.is_absolute() else PROJECT_ROOT / file_path
    if expected_type not in resolved_path.parts:
        failures.append(f"{prefix} file is not under the expected {expected_type} folder: {item['file']}")
    if not resolved_path.exists():
        failures.append(f"{prefix} selected file does not exist: {resolved_path}")

    segment = item.get("segment", {})
    if segment.get("duration", 0) <= 0:
        failures.append(f"{prefix} has non-positive duration.")
    if segment.get("start_time", -1) < 0:
        failures.append(f"{prefix} has negative start_time.")

    validation = item.get("validation", {})
    if validation.get("asset_available") is False:
        failures.append(f"{prefix} asset was retrieved but not available for validation.")
        return failures

    silence_ratio = float(validation.get("silence_ratio", 1.0))
    clipping_ratio = float(validation.get("clipping_ratio", 1.0))
    stability = float(validation.get("stability", 0.0))

    if clipping_ratio > 0.02:
        failures.append(f"{prefix} clips too much: clipping_ratio={clipping_ratio}")

    if expected_type in {"wind", "rain"}:
        if silence_ratio > 0.35:
            failures.append(f"{prefix} has too much silence: silence_ratio={silence_ratio}")
        if stability < 0.20:
            failures.append(f"{prefix} is too unstable for a texture bed: stability={stability}")
    elif expected_type == "thunder" and silence_ratio > 0.80:
        failures.append(f"{prefix} has too much silence for thunder: silence_ratio={silence_ratio}")

    return failures


def export_segment_clip(case_name: str, index: int, item: dict, out_dir: Path) -> None:
    source_path = Path(item["file"])
    source_path = source_path if source_path.is_absolute() else PROJECT_ROOT / source_path
    if not source_path.exists():
        return

    info = sf.info(source_path)
    segment = item["segment"]
    start_frame = int(round(float(segment["start_time"]) * info.samplerate))
    frames = int(round(float(segment["duration"]) * info.samplerate))
    audio, sample_rate = sf.read(source_path, start=start_frame, frames=frames, always_2d=True)

    safe_file = f"{case_name}_{index}_{source_path.stem}_{segment['start_time']:.1f}s.wav"
    sf.write(out_dir / safe_file, audio, sample_rate)


def print_case_summary(result: dict, failures: list[str]) -> None:
    for item in result.get("results", []):
        segment = item["segment"]
        validation = item.get("validation", {})
        print(
            f"  {item['weather_type']:7} "
            f"score={item['score']:.4f} "
            f"file={Path(item['file']).name} "
            f"start={segment['start_time']}s "
            f"duration={segment['duration']}s "
            f"silence={validation.get('silence_ratio', 'n/a')} "
            f"clip={validation.get('clipping_ratio', 'n/a')}"
        )

    if failures:
        print("  case status: FAIL")
    else:
        print("  case status: PASS")


if __name__ == "__main__":
    raise SystemExit(main())
