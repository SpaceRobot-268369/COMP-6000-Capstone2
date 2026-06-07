"""Integration tests for the Layer D registry handler."""

from __future__ import annotations

import io
import unittest

import numpy as np
import soundfile as sf

from acoustic_ai.layers.layer_d.attempts.songke__mvp_1__layered_mix.code import handler


class LayerDHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = handler.load(
            None,
            {
                "default_duration_s": 30.0,
                "event_gain_db": -18.0,
                "event_activity_envelope": True,
                "event_boundary_fade_s": 1.0,
                "event_bandpass_hz": [500.0, 8000.0],
                "weather_gain_db": -12.0,
                "peak_ceiling": 0.95,
            },
        )

    def test_generates_final_wav_from_upstream_layer_bytes(self) -> None:
        result = handler.generate(
            self.state,
            ambient_wav_bytes=_wav_bytes(0.01, sample_rate=16_000, duration_s=1.0),
            weather_wav_bytes=_wav_bytes(0.02, sample_rate=44_100, duration_s=1.0),
            event_wav_bytes=_wav_bytes(0.03, sample_rate=22_050, duration_s=1.0),
            duration_s=2.0,
        )

        info = sf.info(io.BytesIO(result["wav_bytes"]))
        metadata = result["metadata"]
        layer_d = metadata["layer_d"]

        self.assertEqual(info.samplerate, 22_050)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.subtype, "PCM_16")
        self.assertAlmostEqual(info.duration, 2.0, places=3)
        self.assertEqual(metadata["audio"]["duration_s"], 2.0)
        self.assertEqual(layer_d["event_gain_db"], -18.0)
        self.assertEqual(layer_d["event_boundary_fade_s"], 1.0)
        self.assertEqual(layer_d["event_bandpass_hz"], (500.0, 8000.0))
        self.assertEqual(layer_d["weather_gain_db"], -12.0)
        self.assertEqual(layer_d["peak_ceiling"], 0.95)
        self.assertEqual(result["mel_db"].shape[0], 128)

    def test_supports_ambient_only_mix(self) -> None:
        result = handler.generate(
            self.state,
            ambient_wav_bytes=_wav_bytes(0.01, sample_rate=16_000, duration_s=0.5),
            duration_s=1.0,
        )

        info = sf.info(io.BytesIO(result["wav_bytes"]))
        self.assertAlmostEqual(info.duration, 1.0, places=3)
        self.assertIsNone(result["metadata"]["layer_d"]["prepared_layers"]["weather"])
        self.assertIsNone(result["metadata"]["layer_d"]["prepared_layers"]["events"])

    def test_requires_layer_a_ambient_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires ambient_wav_bytes"):
            handler.generate(self.state, duration_s=1.0)


def _wav_bytes(value: float, *, sample_rate: int, duration_s: float) -> bytes:
    frames = round(sample_rate * duration_s)
    audio = np.full((frames, 1), value, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
