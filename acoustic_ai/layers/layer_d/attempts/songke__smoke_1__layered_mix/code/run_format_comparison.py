"""Generate Layer D source-format comparison stems from three real inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio_format import normalize_audio_file


FORMATS = {
    "22050_mono_pcm16": {
        "sample_rate": 22_050,
        "channels": 1,
        "subtype": "PCM_16",
    },
    "44100_stereo_pcm24": {
        "sample_rate": 44_100,
        "channels": 2,
        "subtype": "PCM_24",
    },
}


def run(source_root: Path, output_root: Path) -> dict:
    inputs = {
        "ambient": source_root / "Layer_A_ambient_3min" / "spring_morning.wav",
        "weather": source_root / "Layer_B_weather_3min" / "rain_and_wind_01_3min.wav",
        "events": source_root / "Layer_C_even_3min" / "bird_mix_01.wav",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing comparison input(s): {missing}")

    report: dict = {"inputs": {key: str(value) for key, value in inputs.items()}, "formats": {}}
    for format_name, config in FORMATS.items():
        format_report = {}
        for role, source_path in inputs.items():
            output_path = output_root / format_name / f"{role}.wav"
            result = normalize_audio_file(
                source_path,
                output_path,
                target_sample_rate=config["sample_rate"],
                target_channels=config["channels"],
                subtype=config["subtype"],
            )
            format_report[role] = {
                "output_path": str(output_path),
                "operations": list(result.operations),
                "source_metrics": result.source_metrics,
                "output_metrics": result.output_metrics,
            }
        report["formats"][format_name] = format_report

    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "comparison_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_root / "listening_review.md").write_text(
        "# Format Comparison Listening Review\n\n"
        "Compare each normalized stem with its source. No gain, EQ, denoising, "
        "compression, limiting, fades, or mixing were applied.\n\n"
        "| Role | 22.05 kHz mono notes | 44.1 kHz stereo notes | Preferred |\n"
        "|---|---|---|---|\n"
        "| Ambient | | | |\n"
        "| Weather | | | |\n"
        "| Events | | | |\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_root.resolve(), args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
