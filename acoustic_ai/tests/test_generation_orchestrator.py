"""Tests for the A/B/C to D generation orchestrator."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from acoustic_ai.server import registry


class GenerationOrchestratorTest(unittest.TestCase):
    def test_calls_default_layers_and_hands_wavs_to_layer_d(self) -> None:
        calls = []

        def fake_generate(layer_id, attempt_id, seed, **params):
            calls.append((layer_id, attempt_id, seed, params))
            return {
                "wav_bytes": f"{layer_id}_wav".encode(),
                "mel_db": np.zeros((2, 2), dtype=np.float32),
                "metadata": {"audio": {"duration_s": 30.0}, "source": layer_id},
            }

        with patch.object(registry, "generate", side_effect=fake_generate):
            result = registry.orchestrate_generation(
                seed=42,
                duration_s=30.0,
                season="summer",
                diel="morning",
                weather_type="wind",
                intensity="light",
            )

        self.assertEqual([call[0] for call in calls], ["layer_a", "layer_b", "layer_c", "layer_d"])
        layer_d_params = calls[-1][3]
        self.assertIsNone(calls[-1][2])
        self.assertEqual(layer_d_params["ambient_wav_bytes"], b"layer_a_wav")
        self.assertEqual(layer_d_params["weather_wav_bytes"], b"layer_b_wav")
        self.assertEqual(layer_d_params["event_wav_bytes"], b"layer_c_wav")
        self.assertNotIn("season", layer_d_params)
        self.assertNotIn("diel", layer_d_params)
        self.assertNotIn("weather_type", layer_d_params)
        self.assertNotIn("intensity", layer_d_params)
        orchestration = result["metadata"]["orchestration"]
        self.assertEqual(orchestration["attempts"]["layer_d"], registry.default_attempt_id("layer_d"))
        self.assertEqual(orchestration["upstream"]["layer_a"]["source"], "layer_a")
        self.assertEqual(
            orchestration["parameter_routing"]["layer_d"],
            ["ambient_wav_bytes", "weather_wav_bytes", "event_wav_bytes", "duration_s"],
        )

    def test_routes_generation_parameters_only_to_owning_layers(self) -> None:
        calls = {}

        def fake_generate(layer_id, attempt_id, seed, **params):
            calls[layer_id] = {"seed": seed, **params}
            return {
                "wav_bytes": layer_id.encode(),
                "metadata": {"audio": {"duration_s": 10.0}},
            }

        with patch.object(registry, "generate", side_effect=fake_generate):
            registry.orchestrate_generation(
                seed=7,
                duration_s=10.0,
                season="winter",
                diel="night",
                weather_type="rain",
                intensity="heavy",
            )

        self.assertEqual(
            set(calls["layer_a"]),
            {"seed", "season", "diel"},
        )
        self.assertEqual(
            set(calls["layer_b"]),
            {"seed", "weather_type", "intensity", "duration_s"},
        )
        self.assertEqual(
            set(calls["layer_c"]),
            {"seed", "season", "diel", "duration_s"},
        )
        self.assertEqual(
            set(calls["layer_d"]),
            {
                "seed",
                "ambient_wav_bytes",
                "weather_wav_bytes",
                "event_wav_bytes",
                "duration_s",
            },
        )
        self.assertIsNone(calls["layer_d"]["seed"])

    def test_can_disable_optional_weather_and_events(self) -> None:
        calls = []

        def fake_generate(layer_id, attempt_id, seed, **params):
            calls.append((layer_id, params))
            return {"wav_bytes": layer_id.encode(), "metadata": {"audio": {"duration_s": 5.0}}}

        with patch.object(registry, "generate", side_effect=fake_generate):
            result = registry.orchestrate_generation(
                seed=1,
                duration_s=5.0,
                include_weather=False,
                include_events=False,
            )

        self.assertEqual([call[0] for call in calls], ["layer_a", "layer_d"])
        self.assertIsNone(calls[-1][1]["weather_wav_bytes"])
        self.assertIsNone(calls[-1][1]["event_wav_bytes"])
        self.assertFalse(result["metadata"]["orchestration"]["include_events"])

    def test_rejects_duration_above_current_layer_b_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 30 seconds"):
            registry.orchestrate_generation(seed=42, duration_s=31.0)


if __name__ == "__main__":
    unittest.main()
