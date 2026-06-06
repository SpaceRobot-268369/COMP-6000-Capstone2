"""Render the first local Layer D A+B+C listening mix."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audio_format import load_audio
from .audio_mixer import EventPlacement, LayerStem, MixRequest, export_mix_result, render_mix


DEFAULT_SOURCE_ROOT = Path.home() / "Desktop" / "test_resource"
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "dev-artifacts-self-testing"
    / "first_full_mix"
)


def run(
    source_root: Path,
    output_root: Path,
    duration_s: float = 180.0,
    *,
    event_activity_envelope: bool = True,
    event_boundary_fade_s: float = 0.15,
    event_gain_db: float = -12.0,
    event_bandpass_hz: tuple[float, float] | None = None,
) -> None:
    ambient_path = source_root / "Layer_A_ambient_3min" / "spring_morning.wav"
    weather_path = source_root / "Layer_B_weather_3min" / "rain_and_wind_01_3min.wav"
    event_path = source_root / "Layer_C_even_3min" / "bird_mix_01.wav"

    ambient = _load_stem(ambient_path, "ambient")
    weather = _load_stem(weather_path, "weather")
    event = _load_stem(event_path, "event")
    result = render_mix(
        MixRequest(
            ambient=ambient,
            weather=weather,
            events=(EventPlacement(stem=event, start_s=0.0),),
            duration_s=duration_s,
            event_activity_envelope=event_activity_envelope,
            event_boundary_fade_s=event_boundary_fade_s,
            event_gain_db=event_gain_db,
            event_bandpass_hz=event_bandpass_hz,
        )
    )
    export_mix_result(
        result,
        output_root / "audio.wav",
        output_root / "explanation.json",
    )


def _load_stem(path: Path, role: str) -> LayerStem:
    if not path.is_file():
        raise FileNotFoundError(f"missing Layer D listening input: {path}")
    audio, sample_rate = load_audio(path)
    return LayerStem(
        role=role,
        audio=audio,
        sample_rate=sample_rate,
        source_id=str(path),
        metadata={"source_path": str(path)},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument(
        "--disable-event-activity-envelope",
        action="store_true",
        help="Reproduce the v1 mix without event activity fades.",
    )
    parser.add_argument("--event-boundary-fade-s", type=float, default=0.15)
    parser.add_argument("--event-gain-db", type=float, default=-12.0)
    parser.add_argument("--event-bandpass-low-hz", type=float)
    parser.add_argument("--event-bandpass-high-hz", type=float)
    args = parser.parse_args()
    event_bandpass_hz = None
    if args.event_bandpass_low_hz is not None or args.event_bandpass_high_hz is not None:
        if args.event_bandpass_low_hz is None or args.event_bandpass_high_hz is None:
            parser.error("both event bandpass limits must be provided")
        event_bandpass_hz = (
            args.event_bandpass_low_hz,
            args.event_bandpass_high_hz,
        )
    run(
        args.source_root.resolve(),
        args.output_root.resolve(),
        args.duration_s,
        event_activity_envelope=not args.disable_event_activity_envelope,
        event_boundary_fade_s=args.event_boundary_fade_s,
        event_gain_db=args.event_gain_db,
        event_bandpass_hz=event_bandpass_hz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
