"""Tests for the Layer E analysis orchestrator."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from acoustic_ai.server import registry
from acoustic_ai.server import server


FUSED_REPORT = {
    "schema_version": "analysis_aggregator.v1",
    "decision": {
        "schema_version": "analysis_decision.v1",
        "season": {"value": "autumn"},
        "time_of_day": {"value": "night"},
        "weather": {
            "rain": {"label": "none", "intensity": 0.0},
            "wind": {"label": "light", "intensity": 0.25},
            "thunder": {"label": "none", "intensity": 0.0, "events": []},
        },
        "detected_calls": [
            {"label": "ninox_boobook", "common_name": "Southern Boobook"},
        ],
    },
    "narration": {"summary": "A Southern Boobook is detected in a dry night recording."},
}


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
        self.assertEqual(result["report"]["model_lineage"]["ambient"]["head"], "ambient")
        self.assertEqual(result["report"]["model_lineage"]["aggregator"]["head"], "aggregator")
        self.assertIn("head_reports", result)

    def test_can_omit_head_reports(self) -> None:
        def fake_analyze(_layer_id: str, attempt_id: str, _audio_path: str) -> dict:
            return {"report": {}, "attempt": {"id": attempt_id}}

        with patch.object(registry, "analyze", side_effect=fake_analyze):
            result = registry.orchestrate_analysis("clip.wav", include_head_reports=False)

        self.assertNotIn("head_reports", result)
        self.assertEqual(result["report"]["schema_version"], "analysis_aggregator.v1")

    def test_explicit_attempts_are_used_for_full_analysis(self) -> None:
        ambient_id = registry.default_layer_e_head_attempt("ambient")
        weather_id = registry.default_layer_e_head_attempt("weather")
        events_id = registry.default_layer_e_head_attempt("events")
        aggregator_id = registry.default_layer_e_head_attempt("aggregator")
        calls = []

        def fake_analyze(layer_id: str, attempt_id: str, audio_path: str) -> dict:
            calls.append((layer_id, attempt_id, audio_path))
            return {"report": {}, "attempt": {"id": attempt_id}}

        with patch.object(registry, "analyze", side_effect=fake_analyze):
            result = registry.orchestrate_analysis(
                "clip.wav",
                ambient_attempt=ambient_id,
                weather_attempt=weather_id,
                events_attempt=events_id,
                aggregator_attempt=aggregator_id,
            )

        self.assertEqual(
            [attempt_id for _layer_id, attempt_id, _audio_path in calls],
            [ambient_id, weather_id, events_id],
        )
        self.assertEqual(result["attempts"]["aggregator"]["id"], aggregator_id)

    def test_analysis_run_attaches_inline_narrative(self) -> None:
        captured = {}

        def fake_orchestrate(audio_path: str, **kwargs) -> dict:
            captured["audio_path"] = audio_path
            captured["kwargs"] = kwargs
            return {
                "report": FUSED_REPORT,
                "head_reports": {},
                "attempts": {"aggregator": {"id": "songke__smoke_3__analysis_aggregator"}},
            }

        def fake_write_report(report: dict, register: str) -> dict:
            captured["report"] = report
            captured["register"] = register
            return {
                "register": register,
                "text": "A Southern Boobook calls in the night.",
                "source": "llm",
                "faithful": True,
                "violations": [],
            }

        os.environ["AI_PREWARM"] = "off"
        with patch.object(server.registry, "orchestrate_analysis", side_effect=fake_orchestrate), \
             patch("llm.write_report", side_effect=fake_write_report):
            response = TestClient(server.app).post(
                "/analysis/run",
                files={"file": ("clip.wav", b"RIFFfake", "audio/wav")},
                data={"register": "immersive"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["narrative"]["source"], "llm")
        self.assertEqual(captured["register"], "immersive")
        self.assertIs(captured["report"], FUSED_REPORT)
        self.assertEqual(captured["kwargs"]["ambient_attempt"], None)

    def test_analysis_run_uses_deterministic_fallback_on_llm_failure(self) -> None:
        os.environ["AI_PREWARM"] = "off"
        with patch.object(server.registry, "orchestrate_analysis", return_value={
            "report": FUSED_REPORT,
            "head_reports": {},
            "attempts": {},
        }), patch("llm.write_report", side_effect=RuntimeError("LLM unavailable")):
            response = TestClient(server.app).post(
                "/analysis/run",
                files={"file": ("clip.wav", b"RIFFfake", "audio/wav")},
                data={"register": "immersive"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["narrative"]["source"], "deterministic_fallback")
        self.assertEqual(body["narrative"]["text"], FUSED_REPORT["narration"]["summary"])
        self.assertIn("LLM unavailable", body["narrative_error"])


if __name__ == "__main__":
    unittest.main()
