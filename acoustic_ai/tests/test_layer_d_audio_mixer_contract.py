"""Tests for the Layer D mixer input and output contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from acoustic_ai.layers.layer_d.attempts.lucas__smoke_1__layered_mix.code.audio_mixer import (
    EXPORT_SUBTYPE,
    MIX_CHANNELS,
    MIX_SAMPLE_RATE,
    PEAK_CEILING,
    WEATHER_CROSSFADE_S,
    EventPlacement,
    LAYER_GAIN_DB,
    LayerStem,
    MixRequest,
    MixResult,
    export_mix_result,
    mix_aligned_stems,
    prepare_ambient_stem,
    prepare_event_timeline,
    prepare_weather_stem,
    render_mix,
)


class AudioMixerContractTest(unittest.TestCase):
    def test_runtime_and_export_formats_are_fixed(self) -> None:
        self.assertEqual(MIX_SAMPLE_RATE, 22_050)
        self.assertEqual(MIX_CHANNELS, 1)
        self.assertEqual(EXPORT_SUBTYPE, "PCM_16")

    def test_request_supports_optional_weather_and_positioned_events(self) -> None:
        ambient = _stem("ambient")
        weather = _stem("weather")
        event = _stem("event")

        request = MixRequest(
            ambient=ambient,
            weather=weather,
            events=(EventPlacement(stem=event, start_s=4.5),),
            duration_s=10.0,
        )

        self.assertIs(request.ambient, ambient)
        self.assertIs(request.weather, weather)
        self.assertEqual(request.events[0].start_s, 4.5)
        self.assertEqual(request.duration_s, 10.0)

    def test_result_carries_audio_format_and_explanation(self) -> None:
        audio = np.zeros((22_050, 1), dtype=np.float32)
        result = MixResult(
            audio=audio,
            sample_rate=MIX_SAMPLE_RATE,
            explanation={"runtime_format": "22050_hz_mono_float32"},
        )

        self.assertIs(result.audio, audio)
        self.assertEqual(result.sample_rate, MIX_SAMPLE_RATE)
        self.assertIn("runtime_format", result.explanation)

    def test_mixes_aligned_stems_with_fixed_gains(self) -> None:
        ambient = _stem("ambient", value=0.1, frames=100)
        weather = _stem("weather", value=0.2, frames=100)
        event = _stem("event", value=0.3, frames=100)

        result = mix_aligned_stems(ambient, weather=weather, events=(event,))

        expected = (
            0.1
            + 0.2 * (10.0 ** (LAYER_GAIN_DB["weather"] / 20.0))
            + 0.3 * (10.0 ** (LAYER_GAIN_DB["event"] / 20.0))
        )
        np.testing.assert_allclose(result.audio, expected, atol=1e-7)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertEqual(len(result.explanation["layers"]), 3)
        self.assertEqual(
            result.explanation["processing"],
            ["fixed_gain_sum", "peak_ceiling"],
        )
        self.assertFalse(result.explanation["peak_protection"]["applied"])

    def test_rejects_unaligned_stems(self) -> None:
        ambient = _stem("ambient", frames=100)
        weather = _stem("weather", frames=99)

        with self.assertRaisesRegex(ValueError, "same frame count"):
            mix_aligned_stems(ambient, weather=weather)

    def test_scales_entire_mix_when_peak_exceeds_ceiling(self) -> None:
        ambient = _stem("ambient", value=0.8, frames=100)
        weather = _stem("weather", value=0.8, frames=100)
        event = _stem("event", value=0.8, frames=100)

        result = mix_aligned_stems(ambient, weather=weather, events=(event,))
        protection = result.explanation["peak_protection"]

        self.assertTrue(protection["applied"])
        self.assertLess(protection["scale"], 1.0)
        self.assertGreater(protection["input_peak"], PEAK_CEILING)
        self.assertAlmostEqual(protection["output_peak"], PEAK_CEILING, places=6)
        self.assertLessEqual(float(np.max(np.abs(result.audio))), PEAK_CEILING)

    def test_peak_ceiling_can_be_overridden(self) -> None:
        result = mix_aligned_stems(
            _stem("ambient", value=0.8, frames=100),
            peak_ceiling=0.5,
        )

        self.assertAlmostEqual(float(np.max(np.abs(result.audio))), 0.5, places=6)
        self.assertEqual(result.explanation["peak_protection"]["ceiling"], 0.5)

    def test_event_gain_can_be_overridden(self) -> None:
        ambient = _stem("ambient", value=0.0, frames=100)
        event = _stem("event", value=1.0, frames=100)

        result = mix_aligned_stems(
            ambient,
            events=(event,),
            gain_db_overrides={"event": -18.0},
        )

        expected = 10.0 ** (-18.0 / 20.0)
        np.testing.assert_allclose(result.audio, expected, atol=1e-7)
        event_row = next(row for row in result.explanation["layers"] if row["role"] == "event")
        self.assertEqual(event_row["gain_db"], -18.0)

    def test_exports_pcm16_wav_and_explanation_json(self) -> None:
        result = mix_aligned_stems(_stem("ambient", value=0.1, frames=100))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            wav_path = output_root / "mix.wav"
            explanation_path = output_root / "explanation.json"

            exported = export_mix_result(result, wav_path, explanation_path)
            info = sf.info(wav_path)
            saved_explanation = json.loads(explanation_path.read_text(encoding="utf-8"))

        self.assertEqual(info.samplerate, MIX_SAMPLE_RATE)
        self.assertEqual(info.channels, MIX_CHANNELS)
        self.assertEqual(info.subtype, EXPORT_SUBTYPE)
        self.assertEqual(exported, saved_explanation)
        self.assertEqual(saved_explanation["export"]["metrics"]["frames"], 100)

    def test_prepares_ambient_by_normalizing_and_looping_to_duration(self) -> None:
        ambient = LayerStem(
            role="ambient",
            audio=np.array([[0.1], [0.2]], dtype=np.float32),
            sample_rate=MIX_SAMPLE_RATE,
            source_id="ambient_short",
        )

        prepared = prepare_ambient_stem(ambient, duration_s=4 / MIX_SAMPLE_RATE)

        np.testing.assert_allclose(
            prepared.audio[:, 0],
            np.array([0.1, 0.2, 0.1, 0.2], dtype=np.float32),
        )
        self.assertEqual(prepared.sample_rate, MIX_SAMPLE_RATE)
        self.assertEqual(prepared.metadata["duration_operation"], "loop")

    def test_prepares_ambient_by_trimming_to_duration(self) -> None:
        ambient = LayerStem(
            role="ambient",
            audio=np.arange(6, dtype=np.float32)[:, None],
            sample_rate=MIX_SAMPLE_RATE,
            source_id="ambient_long",
        )

        prepared = prepare_ambient_stem(ambient, duration_s=4 / MIX_SAMPLE_RATE)

        np.testing.assert_array_equal(prepared.audio[:, 0], np.arange(4, dtype=np.float32))
        self.assertEqual(prepared.metadata["duration_operation"], "trim")

    def test_rejects_invalid_ambient_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite"):
            prepare_ambient_stem(_stem("ambient"), duration_s=0.0)

    def test_prepares_weather_with_crossfade_loop(self) -> None:
        weather = LayerStem(
            role="weather",
            audio=np.array([[1.0], [1.0], [-1.0], [-1.0]], dtype=np.float32),
            sample_rate=MIX_SAMPLE_RATE,
            source_id="weather_short",
        )

        prepared = prepare_weather_stem(
            weather,
            duration_s=8 / MIX_SAMPLE_RATE,
            crossfade_s=2 / MIX_SAMPLE_RATE,
        )

        self.assertEqual(prepared.audio.shape, (8, 1))
        self.assertEqual(prepared.metadata["duration_operation"], "loop_crossfade")
        self.assertEqual(prepared.metadata["crossfade_frames"], 2)
        maximum_jump = float(np.max(np.abs(np.diff(prepared.audio[:, 0]))))
        self.assertLess(maximum_jump, 2.0)

    def test_prepares_weather_by_trimming_without_crossfade(self) -> None:
        weather = _stem("weather", value=0.1, frames=10)

        prepared = prepare_weather_stem(
            weather,
            duration_s=5 / MIX_SAMPLE_RATE,
            crossfade_s=WEATHER_CROSSFADE_S,
        )

        self.assertEqual(prepared.audio.shape, (5, 1))
        self.assertEqual(prepared.metadata["duration_operation"], "trim")
        self.assertEqual(prepared.metadata["crossfade_frames"], 0)

    def test_prepares_single_frame_weather_without_infinite_crossfade_loop(self) -> None:
        weather = _stem("weather", value=0.1, frames=1)

        prepared = prepare_weather_stem(
            weather,
            duration_s=5 / MIX_SAMPLE_RATE,
        )

        self.assertEqual(prepared.audio.shape, (5, 1))
        np.testing.assert_allclose(prepared.audio, 0.1)
        self.assertEqual(prepared.metadata["duration_operation"], "loop")
        self.assertEqual(prepared.metadata["crossfade_frames"], 0)

    def test_rejects_non_weather_stem_for_weather_preparation(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a weather stem"):
            prepare_weather_stem(_stem("ambient"), duration_s=1.0)

    def test_places_event_on_target_timeline(self) -> None:
        event = LayerStem(
            role="event",
            audio=np.full((10_000, 1), 0.2, dtype=np.float32),
            sample_rate=MIX_SAMPLE_RATE,
            source_id="event_one",
        )

        prepared = prepare_event_timeline(
            (EventPlacement(stem=event, start_s=1.0),),
            duration_s=2.0,
        )

        placement = prepared.metadata["placements"][0]
        self.assertEqual(placement["start_frame"], MIX_SAMPLE_RATE)
        self.assertEqual(placement["written_frames"], 10_000)
        self.assertFalse(placement["trimmed_at_end"])
        self.assertEqual(placement["activity_envelope"]["active_region_count"], 1)
        self.assertEqual(float(prepared.audio[0, 0]), 0.0)
        self.assertGreater(float(np.max(prepared.audio)), 0.19)

    def test_overlapping_events_are_summed_and_end_is_trimmed(self) -> None:
        first = _stem("event", value=0.2, frames=10_000)
        second = _stem("event", value=0.3, frames=10_000)

        prepared = prepare_event_timeline(
            (
                EventPlacement(stem=first, start_s=0.0),
                EventPlacement(stem=second, start_s=0.25),
            ),
            duration_s=0.5,
        )

        self.assertTrue(prepared.metadata["placements"][1]["trimmed_at_end"])
        self.assertGreater(float(np.max(prepared.audio)), 0.3)

    def test_event_activity_envelope_fades_noise_boundaries(self) -> None:
        silence = np.zeros((4_000, 1), dtype=np.float32)
        noisy_event = np.full((10_000, 1), 0.2, dtype=np.float32)
        event = LayerStem(
            role="event",
            audio=np.concatenate((silence, noisy_event, silence)),
            sample_rate=MIX_SAMPLE_RATE,
            source_id="noisy_event",
        )

        prepared = prepare_event_timeline(
            (EventPlacement(stem=event, start_s=0.0),),
            duration_s=event.audio.shape[0] / MIX_SAMPLE_RATE,
        )
        placement = prepared.metadata["placements"][0]

        self.assertEqual(placement["activity_envelope"]["active_region_count"], 1)
        self.assertEqual(placement["activity_envelope"]["fade_curve"], "smoothstep")
        self.assertEqual(float(prepared.audio[4_000, 0]), 0.0)
        self.assertGreater(float(prepared.audio[6_000, 0]), 0.0)
        self.assertEqual(float(prepared.audio[13_999, 0]), 0.0)

    def test_event_boundary_fade_duration_is_configurable(self) -> None:
        event = _stem("event", value=0.2, frames=44_100)

        prepared = prepare_event_timeline(
            (EventPlacement(stem=event, start_s=0.0),),
            duration_s=2.0,
            boundary_fade_s=1.0,
        )

        activity = prepared.metadata["placements"][0]["activity_envelope"]
        self.assertEqual(activity["boundary_fade_s"], 1.0)
        self.assertEqual(activity["fade_curve"], "smoothstep")
        self.assertLess(float(prepared.audio[5_000, 0]), float(prepared.audio[11_025, 0]))

    def test_event_bandpass_reduces_out_of_band_energy(self) -> None:
        t = np.arange(MIX_SAMPLE_RATE, dtype=np.float32) / MIX_SAMPLE_RATE
        low_noise = 0.2 * np.sin(2 * np.pi * 100 * t)
        bird_band = 0.2 * np.sin(2 * np.pi * 3_000 * t)
        high_noise = 0.2 * np.sin(2 * np.pi * 10_000 * t)
        event = LayerStem(
            role="event",
            audio=(low_noise + bird_band + high_noise)[:, None].astype(np.float32),
            sample_rate=MIX_SAMPLE_RATE,
            source_id="bandpass_event",
        )

        prepared = prepare_event_timeline(
            (EventPlacement(stem=event, start_s=0.0),),
            duration_s=1.0,
            apply_activity_envelope=False,
            bandpass_hz=(500.0, 8_000.0),
        )

        bandpass = prepared.metadata["placements"][0]["bandpass"]
        self.assertTrue(bandpass["applied"])
        self.assertEqual(bandpass["low_hz"], 500.0)
        self.assertEqual(bandpass["high_hz"], 8_000.0)
        self.assertLess(float(np.sqrt(np.mean(prepared.audio**2))), 0.18)
        self.assertGreater(float(np.sqrt(np.mean(prepared.audio**2))), 0.10)

    def test_event_bandpass_skips_source_too_short_for_zero_phase_filter(self) -> None:
        event = _stem("event", value=0.2, frames=10)

        prepared = prepare_event_timeline(
            (EventPlacement(stem=event, start_s=0.0),),
            duration_s=10 / MIX_SAMPLE_RATE,
            apply_activity_envelope=False,
            bandpass_hz=(500.0, 8_000.0),
        )

        bandpass = prepared.metadata["placements"][0]["bandpass"]
        self.assertFalse(bandpass["applied"])
        self.assertEqual(bandpass["reason"], "source_too_short")
        np.testing.assert_allclose(prepared.audio, event.audio)

    def test_rejects_negative_event_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative finite"):
            prepare_event_timeline(
                (EventPlacement(stem=_stem("event"), start_s=-1.0),),
                duration_s=1.0,
            )

    def test_can_reproduce_event_timeline_without_activity_envelope(self) -> None:
        event = LayerStem(
            role="event",
            audio=np.full((100, 1), 0.2, dtype=np.float32),
            sample_rate=MIX_SAMPLE_RATE,
            source_id="legacy_event",
        )

        prepared = prepare_event_timeline(
            (EventPlacement(stem=event, start_s=0.0),),
            duration_s=100 / MIX_SAMPLE_RATE,
            apply_activity_envelope=False,
        )

        np.testing.assert_allclose(prepared.audio, event.audio)
        self.assertFalse(prepared.metadata["activity_envelope_applied"])
        self.assertFalse(prepared.metadata["placements"][0]["activity_envelope"]["applied"])

    def test_renders_end_to_end_mix_request(self) -> None:
        duration_s = 8 / MIX_SAMPLE_RATE
        request = MixRequest(
            ambient=_stem("ambient", value=0.1, frames=4),
            weather=_stem("weather", value=0.2, frames=4),
            events=(
                EventPlacement(
                    stem=_stem("event", value=0.3, frames=2),
                    start_s=2 / MIX_SAMPLE_RATE,
                ),
            ),
            duration_s=duration_s,
        )

        result = render_mix(request)

        self.assertEqual(result.audio.shape, (8, 1))
        self.assertEqual(result.sample_rate, MIX_SAMPLE_RATE)
        self.assertEqual(result.explanation["duration_s"], duration_s)
        self.assertEqual(
            result.explanation["prepared_layers"]["ambient"]["duration_operation"],
            "loop",
        )
        self.assertEqual(
            result.explanation["prepared_layers"]["weather"]["duration_operation"],
            "loop_crossfade",
        )
        self.assertEqual(
            result.explanation["prepared_layers"]["events"]["placements"][0][
                "start_frame"
            ],
            2,
        )


def _stem(role: str, *, value: float = 0.0, frames: int = 16_000) -> LayerStem:
    return LayerStem(
        role=role,
        audio=np.full((frames, 1), value, dtype=np.float32),
        sample_rate=MIX_SAMPLE_RATE,
        source_id=f"{role}_test",
    )


if __name__ == "__main__":
    unittest.main()
