"""Tests for the Layer D multi-clip mixer handler contract."""

from __future__ import annotations

import io
import unittest

import numpy as np
import soundfile as sf

from acoustic_ai.layers.layer_d.attempts.songke__mvp_2__multi_clip_mix.code import handler


class LayerDMultiClipHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = handler.load(
            None,
            {
                "default_duration_s": 30.0,
                "event_gain_db": -8.0,
                "event_activity_envelope": True,
                "event_boundary_fade_s": 1.0,
                "event_bandpass_hz": [500.0, 8000.0],
                "weather_gain_db": -2.0,
                "peak_ceiling": 0.95,
            },
        )

    def test_legacy_single_stem_input_still_generates_final_wav(self) -> None:
        result = handler.generate(
            self.state,
            ambient_wav_bytes=_wav_bytes(0.01, sample_rate=16_000, duration_s=1.0),
            weather_wav_bytes=_wav_bytes(0.02, sample_rate=44_100, duration_s=1.0),
            event_wav_bytes=_wav_bytes(0.03, sample_rate=22_050, duration_s=1.0),
            duration_s=2.0,
            placement_seed=42,
        )

        info = sf.info(io.BytesIO(result["wav_bytes"]))
        layer_d = result["metadata"]["layer_d"]

        self.assertEqual(info.samplerate, 22_050)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.subtype, "PCM_16")
        self.assertAlmostEqual(info.duration, 2.0, places=3)
        self.assertEqual(layer_d["attempt_contract"], "multi_clip_mix_v2")
        self.assertFalse(layer_d["multi_clip_enabled"])
        self.assertEqual(layer_d["placement_seed"], 42)
        self.assertEqual(layer_d["input_contract"]["weather"]["mode"], "legacy_single_stem")
        self.assertEqual(layer_d["input_contract"]["events"]["mode"], "legacy_single_stem")
        self.assertEqual(layer_d["weather_gain_db"], -2.0)
        self.assertEqual(layer_d["event_gain_db"], -8.0)
        self.assertEqual(result["mel_db"].shape[0], 128)

    def test_accepts_v2_clip_lists_with_explicit_event_onsets(self) -> None:
        result = handler.generate(
            self.state,
            ambient_wav_bytes=_wav_bytes(0.01, sample_rate=16_000, duration_s=1.0),
            weather_clips=[
                {
                    "wav": _wav_bytes(0.02, sample_rate=44_100, duration_s=1.0),
                    "weather_type": "rain",
                    "continuous": True,
                    "onsets_s": None,
                    "gain_db": None,
                    "change": None,
                }
            ],
            event_clips=[
                {
                    "wav": _wav_bytes(0.03, sample_rate=22_050, duration_s=0.25),
                    "species": "boobook_owl",
                    "onsets_s": [0.1, 0.8],
                    "gain_db": None,
                },
                {
                    "wav": _wav_bytes(0.04, sample_rate=22_050, duration_s=0.25),
                    "species": "tawny_frogmouth",
                    "onsets_s": [1.2],
                    "gain_db": None,
                },
            ],
            duration_s=2.0,
            placement_seed=42,
        )

        info = sf.info(io.BytesIO(result["wav_bytes"]))
        layer_d = result["metadata"]["layer_d"]
        input_contract = layer_d["input_contract"]
        event_metadata = layer_d["prepared_layers"]["events"]

        self.assertAlmostEqual(info.duration, 2.0, places=3)
        self.assertTrue(layer_d["multi_clip_enabled"])
        self.assertEqual(input_contract["weather"]["mode"], "multi_clip")
        self.assertEqual(input_contract["weather"]["clip_count"], 1)
        self.assertEqual(input_contract["events"]["mode"], "multi_clip")
        self.assertEqual(input_contract["events"]["clip_count"], 2)
        self.assertEqual(input_contract["events"]["placement_count"], 3)
        self.assertEqual(len(event_metadata["placements"]), 3)

    def test_random_onset_fallback_is_reproducible_with_placement_seed(self) -> None:
        first = handler.generate(
            self.state,
            ambient_wav_bytes=_wav_bytes(0.01, sample_rate=16_000, duration_s=1.0),
            event_clips=[
                {
                    "wav": _wav_bytes(0.03, sample_rate=22_050, duration_s=0.25),
                    "species": "boobook_owl",
                    "onsets_s": None,
                }
            ],
            duration_s=2.0,
            placement_seed=123,
        )
        second = handler.generate(
            self.state,
            ambient_wav_bytes=_wav_bytes(0.01, sample_rate=16_000, duration_s=1.0),
            event_clips=[
                {
                    "wav": _wav_bytes(0.03, sample_rate=22_050, duration_s=0.25),
                    "species": "boobook_owl",
                    "onsets_s": None,
                }
            ],
            duration_s=2.0,
            placement_seed=123,
        )

        first_clip = first["metadata"]["layer_d"]["input_contract"]["events"]["clips"][0]
        second_clip = second["metadata"]["layer_d"]["input_contract"]["events"]["clips"][0]

        self.assertTrue(first_clip["placement_random"])
        self.assertEqual(first_clip["placement_seed"], 123)
        self.assertEqual(first_clip["onsets_s"], second_clip["onsets_s"])
        self.assertGreaterEqual(first_clip["onsets_s"][0], 0.0)
        self.assertLessEqual(first_clip["onsets_s"][0], 1.75)

    def test_different_placement_seed_changes_random_onset(self) -> None:
        base_kwargs = {
            "ambient_wav_bytes": _wav_bytes(0.01, sample_rate=16_000, duration_s=1.0),
            "event_clips": [
                {
                    "wav": _wav_bytes(0.03, sample_rate=22_050, duration_s=0.25),
                    "species": "boobook_owl",
                    "onsets_s": None,
                }
            ],
            "duration_s": 2.0,
        }
        first = handler.generate(self.state, **base_kwargs, placement_seed=1)
        second = handler.generate(self.state, **base_kwargs, placement_seed=2)

        first_onset = first["metadata"]["layer_d"]["input_contract"]["events"]["clips"][0]["onsets_s"][0]
        second_onset = second["metadata"]["layer_d"]["input_contract"]["events"]["clips"][0]["onsets_s"][0]

        self.assertNotEqual(first_onset, second_onset)

    def test_random_onset_falls_back_to_zero_when_clip_exceeds_duration(self) -> None:
        result = handler.generate(
            self.state,
            ambient_wav_bytes=_wav_bytes(0.01, sample_rate=16_000, duration_s=1.0),
            event_clips=[
                {
                    "wav": _wav_bytes(0.03, sample_rate=22_050, duration_s=3.0),
                    "species": "boobook_owl",
                    "onsets_s": None,
                }
            ],
            duration_s=2.0,
            placement_seed=123,
        )

        onset = result["metadata"]["layer_d"]["input_contract"]["events"]["clips"][0]["onsets_s"][0]

        self.assertEqual(onset, 0.0)

    def test_accepts_continuous_and_discrete_weather_clips(self) -> None:
        result = handler.generate(
            self.state,
            ambient_wav_bytes=_wav_bytes(0.01, sample_rate=16_000, duration_s=1.0),
            weather_clips=[
                {
                    "wav": _wav_bytes(0.02, sample_rate=44_100, duration_s=0.5),
                    "weather_type": "rain",
                    "continuous": True,
                    "onsets_s": None,
                },
                {
                    "wav": _wav_bytes(0.08, sample_rate=22_050, duration_s=0.25),
                    "weather_type": "thunder",
                    "continuous": False,
                    "onsets_s": [0.4, 1.5],
                },
            ],
            duration_s=2.0,
            placement_seed=42,
        )

        layer_d = result["metadata"]["layer_d"]
        weather_contract = layer_d["input_contract"]["weather"]
        weather_metadata = layer_d["prepared_layers"]["weather"]

        self.assertEqual(weather_contract["mode"], "multi_clip")
        self.assertEqual(weather_contract["clip_count"], 2)
        self.assertEqual(weather_contract["placement_count"], 2)
        self.assertEqual(weather_contract["clips"][1]["weather_type"], "thunder")
        self.assertFalse(weather_contract["clips"][1]["continuous"])
        self.assertEqual(weather_contract["clips"][1]["onsets_s"], [0.4, 1.5])
        self.assertEqual(len(weather_contract["clips"][1]["placements"]), 2)
        self.assertEqual(weather_metadata["clips"][1]["placement_count"], 2)

    def test_discrete_weather_random_onset_is_reproducible(self) -> None:
        kwargs = {
            "ambient_wav_bytes": _wav_bytes(0.01, sample_rate=16_000, duration_s=1.0),
            "weather_clips": [
                {
                    "wav": _wav_bytes(0.08, sample_rate=22_050, duration_s=0.25),
                    "weather_type": "thunder",
                    "continuous": False,
                    "onsets_s": None,
                }
            ],
            "duration_s": 2.0,
        }
        first = handler.generate(self.state, **kwargs, placement_seed=222)
        second = handler.generate(self.state, **kwargs, placement_seed=222)

        first_clip = first["metadata"]["layer_d"]["input_contract"]["weather"]["clips"][0]
        second_clip = second["metadata"]["layer_d"]["input_contract"]["weather"]["clips"][0]

        self.assertTrue(first_clip["placement_random"])
        self.assertEqual(first_clip["placement_seed"], 222)
        self.assertEqual(first_clip["onsets_s"], second_clip["onsets_s"])
        self.assertGreaterEqual(first_clip["onsets_s"][0], 0.0)
        self.assertLessEqual(first_clip["onsets_s"][0], 1.75)

    def test_event_clip_gain_override_changes_output_level(self) -> None:
        base_kwargs = {
            "ambient_wav_bytes": _wav_bytes(0.0, sample_rate=16_000, duration_s=1.0),
            "event_clips": [
                {
                    "wav": _sine_wav_bytes(0.5, sample_rate=22_050, duration_s=0.5),
                    "species": "boobook_owl",
                    "onsets_s": [0.1],
                }
            ],
            "duration_s": 1.0,
        }
        default_gain = handler.generate(self.state, **base_kwargs)
        quieter = handler.generate(
            self.state,
            **{
                **base_kwargs,
                "event_clips": [
                    {
                        **base_kwargs["event_clips"][0],
                        "gain_db": -28.0,
                    }
                ],
            },
        )

        default_clip = default_gain["metadata"]["layer_d"]["input_contract"]["events"]["clips"][0]
        quiet_clip = quieter["metadata"]["layer_d"]["input_contract"]["events"]["clips"][0]

        self.assertFalse(default_clip["gain_override"])
        self.assertTrue(quiet_clip["gain_override"])
        self.assertEqual(quiet_clip["applied_gain_db"], -28.0)
        self.assertLess(_wav_peak(quieter["wav_bytes"]), _wav_peak(default_gain["wav_bytes"]) * 0.2)

    def test_weather_clip_gain_override_changes_output_level(self) -> None:
        base_kwargs = {
            "ambient_wav_bytes": _wav_bytes(0.0, sample_rate=16_000, duration_s=1.0),
            "weather_clips": [
                {
                    "wav": _sine_wav_bytes(0.5, sample_rate=22_050, duration_s=0.5),
                    "weather_type": "thunder",
                    "continuous": False,
                    "onsets_s": [0.1],
                }
            ],
            "duration_s": 1.0,
        }
        default_gain = handler.generate(self.state, **base_kwargs)
        quieter = handler.generate(
            self.state,
            **{
                **base_kwargs,
                "weather_clips": [
                    {
                        **base_kwargs["weather_clips"][0],
                        "gain_db": -22.0,
                    }
                ],
            },
        )

        default_clip = default_gain["metadata"]["layer_d"]["input_contract"]["weather"]["clips"][0]
        quiet_clip = quieter["metadata"]["layer_d"]["input_contract"]["weather"]["clips"][0]

        self.assertFalse(default_clip["gain_override"])
        self.assertTrue(quiet_clip["gain_override"])
        self.assertEqual(quiet_clip["applied_gain_db"], -22.0)
        self.assertLess(_wav_peak(quieter["wav_bytes"]), _wav_peak(default_gain["wav_bytes"]) * 0.2)

    def test_rejects_non_finite_clip_gain(self) -> None:
        with self.assertRaisesRegex(ValueError, "gain_db must be a finite"):
            handler.generate(
                self.state,
                ambient_wav_bytes=_wav_bytes(0.0, sample_rate=16_000, duration_s=1.0),
                event_clips=[
                    {
                        "wav": _sine_wav_bytes(0.5, sample_rate=22_050, duration_s=0.5),
                        "species": "boobook_owl",
                        "onsets_s": [0.1],
                        "gain_db": float("nan"),
                    }
                ],
                duration_s=1.0,
            )

    def test_placed_clips_metadata_summarizes_weather_and_events(self) -> None:
        result = handler.generate(
            self.state,
            ambient_wav_bytes=_wav_bytes(0.0, sample_rate=16_000, duration_s=1.0),
            weather_clips=[
                {
                    "wav": _sine_wav_bytes(0.2, sample_rate=22_050, duration_s=0.5),
                    "weather_type": "rain",
                    "continuous": True,
                    "gain_db": -6.0,
                },
                {
                    "wav": _sine_wav_bytes(0.5, sample_rate=22_050, duration_s=0.25),
                    "weather_type": "thunder",
                    "continuous": False,
                    "onsets_s": None,
                },
            ],
            event_clips=[
                {
                    "wav": _sine_wav_bytes(0.5, sample_rate=22_050, duration_s=0.25),
                    "species": "boobook_owl",
                    "onsets_s": [0.3],
                    "gain_db": -18.0,
                }
            ],
            duration_s=1.0,
            placement_seed=77,
        )

        placed = result["metadata"]["layer_d"]["placed_clips"]

        self.assertEqual(len(placed["weather"]), 2)
        self.assertEqual(len(placed["events"]), 1)
        self.assertEqual(placed["weather"][0]["kind"], "weather")
        self.assertEqual(placed["weather"][0]["weather_type"], "rain")
        self.assertTrue(placed["weather"][0]["continuous"])
        self.assertEqual(placed["weather"][0]["applied_gain_db"], -6.0)
        self.assertTrue(placed["weather"][0]["gain_override"])
        self.assertEqual(placed["weather"][1]["weather_type"], "thunder")
        self.assertFalse(placed["weather"][1]["continuous"])
        self.assertTrue(placed["weather"][1]["placement_random"])
        self.assertEqual(placed["weather"][1]["placement_seed"], 77)
        self.assertEqual(placed["weather"][1]["placement_count"], 1)
        self.assertEqual(len(placed["weather"][1]["placements"]), 1)
        self.assertEqual(placed["events"][0]["kind"], "event")
        self.assertEqual(placed["events"][0]["species"], "boobook_owl")
        self.assertEqual(placed["events"][0]["onsets_s"], [0.3])
        self.assertEqual(placed["events"][0]["applied_gain_db"], -18.0)
        self.assertTrue(placed["events"][0]["gain_override"])

    def test_v2_main_contract_mixes_multi_clip_timeline(self) -> None:
        result = handler.generate(
            self.state,
            ambient_wav_bytes=_sine_wav_bytes(0.05, sample_rate=16_000, duration_s=1.0),
            weather_clips=[
                {
                    "wav": _sine_wav_bytes(0.1, sample_rate=44_100, duration_s=0.75),
                    "weather_type": "rain",
                    "continuous": True,
                    "onsets_s": None,
                    "gain_db": -4.0,
                    "change": None,
                },
                {
                    "wav": _sine_wav_bytes(0.5, sample_rate=22_050, duration_s=0.4),
                    "weather_type": "thunder",
                    "continuous": False,
                    "onsets_s": [0.6, 1.7],
                    "gain_db": -12.0,
                    "change": None,
                },
            ],
            event_clips=[
                {
                    "wav": _sine_wav_bytes(0.35, sample_rate=22_050, duration_s=0.3),
                    "species": "boobook_owl",
                    "onsets_s": [0.5, 1.2],
                    "gain_db": -10.0,
                },
                {
                    "wav": _sine_wav_bytes(0.25, sample_rate=16_000, duration_s=0.25),
                    "species": "tawny_frogmouth",
                    "onsets_s": None,
                    "gain_db": None,
                },
            ],
            duration_s=2.0,
            placement_seed=99,
        )

        info = sf.info(io.BytesIO(result["wav_bytes"]))
        layer_d = result["metadata"]["layer_d"]
        placed = layer_d["placed_clips"]

        self.assertEqual(info.samplerate, 22_050)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.subtype, "PCM_16")
        self.assertAlmostEqual(info.duration, 2.0, places=3)
        self.assertEqual(result["metadata"]["audio"]["duration_s"], 2.0)
        self.assertEqual(result["mel_db"].shape[0], 128)
        self.assertEqual(layer_d["attempt_contract"], "multi_clip_mix_v2")
        self.assertTrue(layer_d["multi_clip_enabled"])
        self.assertEqual(layer_d["peak_ceiling"], 0.95)
        self.assertLessEqual(_wav_peak(result["wav_bytes"]), 0.95)

        self.assertEqual(layer_d["input_contract"]["weather"]["clip_count"], 2)
        self.assertEqual(layer_d["input_contract"]["weather"]["placement_count"], 2)
        self.assertEqual(layer_d["input_contract"]["events"]["clip_count"], 2)
        self.assertEqual(layer_d["input_contract"]["events"]["placement_count"], 3)

        self.assertEqual(len(placed["weather"]), 2)
        self.assertEqual(placed["weather"][0]["weather_type"], "rain")
        self.assertTrue(placed["weather"][0]["continuous"])
        self.assertEqual(placed["weather"][0]["applied_gain_db"], -4.0)
        self.assertEqual(placed["weather"][1]["weather_type"], "thunder")
        self.assertFalse(placed["weather"][1]["continuous"])
        self.assertEqual(placed["weather"][1]["onsets_s"], [0.6, 1.7])
        self.assertEqual(placed["weather"][1]["placement_count"], 2)
        self.assertEqual(len(placed["weather"][1]["placements"]), 2)

        self.assertEqual(len(placed["events"]), 2)
        self.assertEqual(placed["events"][0]["species"], "boobook_owl")
        self.assertEqual(placed["events"][0]["onsets_s"], [0.5, 1.2])
        self.assertEqual(placed["events"][0]["placement_count"], 2)
        self.assertEqual(placed["events"][0]["applied_gain_db"], -10.0)
        self.assertEqual(placed["events"][1]["species"], "tawny_frogmouth")
        self.assertTrue(placed["events"][1]["placement_random"])
        self.assertEqual(placed["events"][1]["placement_seed"], 99)
        self.assertEqual(placed["events"][1]["placement_count"], 1)
        self.assertGreaterEqual(placed["events"][1]["onsets_s"][0], 0.0)
        self.assertLessEqual(placed["events"][1]["onsets_s"][0], 1.75)


def _wav_bytes(value: float, *, sample_rate: int, duration_s: float) -> bytes:
    frames = round(sample_rate * duration_s)
    audio = np.full((frames, 1), value, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _sine_wav_bytes(amplitude: float, *, sample_rate: int, duration_s: float) -> bytes:
    frames = round(sample_rate * duration_s)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    audio = (amplitude * np.sin(2.0 * np.pi * 1000.0 * t))[:, None].astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _wav_peak(wav_bytes: bytes) -> float:
    audio, _sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    return float(np.max(np.abs(audio)))


if __name__ == "__main__":
    unittest.main()
