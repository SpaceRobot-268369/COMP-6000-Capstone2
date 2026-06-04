"""Tests for Layer D technical audio-format normalization."""

from __future__ import annotations

import unittest

import numpy as np

from acoustic_ai.layers.layer_d.attempts.songke__smoke_1__layered_mix.code.audio_format import (
    normalize_audio_format,
)


class AudioFormatTest(unittest.TestCase):
    def test_mono_16k_to_mono_22050_preserves_duration(self) -> None:
        source_rate = 16_000
        audio = _tone(source_rate, duration_s=1.25)[:, None]

        result = normalize_audio_format(
            audio,
            source_rate,
            target_sample_rate=22_050,
            target_channels=1,
        )

        self.assertEqual(result.audio.shape, (round(1.25 * 22_050), 1))
        self.assertEqual(result.sample_rate, 22_050)
        self.assertTrue(np.isfinite(result.audio).all())
        self.assertIn("resample_16000_to_22050", result.operations)

    def test_mono_to_stereo_duplicates_channels(self) -> None:
        audio = _tone(22_050, duration_s=0.5)[:, None]

        result = normalize_audio_format(
            audio,
            22_050,
            target_sample_rate=22_050,
            target_channels=2,
        )

        np.testing.assert_array_equal(result.audio[:, 0], result.audio[:, 1])
        self.assertIn("duplicate_mono_to_stereo", result.operations)

    def test_stereo_to_mono_averages_channels(self) -> None:
        left = np.full(100, 0.5, dtype=np.float32)
        right = np.full(100, -0.25, dtype=np.float32)
        stereo = np.stack((left, right), axis=1)

        result = normalize_audio_format(
            stereo,
            44_100,
            target_sample_rate=44_100,
            target_channels=1,
        )

        np.testing.assert_allclose(result.audio[:, 0], 0.125, atol=1e-7)
        self.assertIn("downmix_2_to_mono", result.operations)

    def test_rejects_non_finite_audio(self) -> None:
        audio = np.array([[0.0], [np.nan]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            normalize_audio_format(
                audio,
                22_050,
                target_sample_rate=22_050,
                target_channels=1,
            )


def _tone(sample_rate: int, duration_s: float) -> np.ndarray:
    t = np.arange(round(sample_rate * duration_s), dtype=np.float32) / sample_rate
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
