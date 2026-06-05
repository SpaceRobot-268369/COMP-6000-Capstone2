"""Tests for the Layer E analysis orchestrator."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acoustic_ai.server import registry


class AnalysisOrchestratorTest(unittest.TestCase):
    def test_runs_default_heads_and_fuses_report(self) -> None:
        calls = []

        def fake_analyze(layer_id: str, attempt_id: str, audio_path: str) -> dict:
            calls.append((layer_id, attempt_id, audio_path))
            if attempt_id == registry.default_layer_e_head_attempt("ambient"):
                return {
                    "report": {
                        "estimated_conditions": {"season": "autumn", "diel_bin": "afternoon"},
                        "confidence": 0.35,
                        "season_confidence": 0.35,
                    },
                    "attempt": {"id": attempt_id, "head": "ambient"},
                }
            if attempt_id == registry.default_layer_e_head_attempt("weather"):
                return {
                    "report": {
                        "observations": {
                            "weather": {
                                "confidence": 0.8,
                                "derived_label": "wind",
                                "wind": {"summary": {"intensity": 0.62, "label": "moderate", "confidence": 0.83}},
                            }
                        }
                    },
                    "attempt": {"id": attempt_id, "head": "weather"},
                }
            if attempt_id == registry.default_layer_e_head_attempt("events"):
                return {
                    "report": {
                        "events": [
                            {
                                "label": "ninox_boobook",
                                "confidence_mean": 0.91,
                                "phenology": {
                                    "common_name": "Southern Boobook",
                                    "diel_signal": "night",
                                    "diel_confidence": 0.85,
                                    "season_signal": "weak",
                                    "season_confidence": 0.2,
                                },
                            }
                        ]
                    },
                    "attempt": {"id": attempt_id, "head": "events"},
                }
            raise AssertionError(f"unexpected attempt: {attempt_id}")

        with patch.object(registry, "analyze", side_effect=fake_analyze):
            result = registry.orchestrate_analysis("clip.wav")

        self.assertEqual([call[0] for call in calls], ["layer_e", "layer_e", "layer_e"])
        self.assertEqual(result["report"]["schema_version"], "analysis_aggregator.v1")
        self.assertEqual(result["report"]["inferred_context"]["diel"]["estimate"], "night")
        self.assertEqual(result["attempts"]["ambient"]["head"], "ambient")
        self.assertEqual(result["attempts"]["weather"]["head"], "weather")
        self.assertEqual(result["attempts"]["events"]["head"], "events")
        self.assertEqual(result["attempts"]["aggregator"]["head"], "aggregator")
        self.assertIn("head_reports", result)

    def test_can_omit_head_reports(self) -> None:
        def fake_analyze(_layer_id: str, attempt_id: str, _audio_path: str) -> dict:
            return {"report": {}, "attempt": {"id": attempt_id}}

        with patch.object(registry, "analyze", side_effect=fake_analyze):
            result = registry.orchestrate_analysis("clip.wav", include_head_reports=False)

        self.assertNotIn("head_reports", result)
        self.assertEqual(result["report"]["schema_version"], "analysis_aggregator.v1")


if __name__ == "__main__":
    unittest.main()
