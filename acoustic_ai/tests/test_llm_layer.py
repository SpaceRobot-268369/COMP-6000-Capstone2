"""Tests for the in-process LLM-OSS layer (parser, report writer, gate,
faithfulness, skills loader, JSON extraction).

Model-free: the LLMService singleton is replaced with a fake so these run
locally without torch/transformers or a downloaded model. Real
model-dependent checks are serverB-only (plan §7).
"""

from __future__ import annotations

import json
import unittest

import acoustic_ai.llm.service as svc_mod
from acoustic_ai.llm import parse_prompt, write_report
from acoustic_ai.llm.faithfulness import validate_narrative
from acoustic_ai.llm.gate import gate_findings
from acoustic_ai.llm.parser import _detect_weather
from acoustic_ai.llm.service import _extract_json
from acoustic_ai.llm.skills import available_skills, load_skill, report_skill_name

REPORT = {
    "schema_version": "analysis_aggregator.v1",
    "observations": {
        "events": [
            {"label": "ninox_boobook", "common_name": "Southern Boobook", "onset_s": 12.4}
        ]
    },
    "decision": {
        "schema_version": "analysis_decision.v1",
        "detected_calls": [
            {"label": "ninox_boobook", "common_name": "Southern Boobook", "confidence": 0.91}
        ],
    },
    "llm_input": {
        "task": "immersive, third-person perspective narration with an analytical tone",
        "decision": {
            "detected_calls": [
                {"label": "ninox_boobook", "common_name": "Southern Boobook"}
            ],
        },
    },
}


class _FakeService:
    """Stand-in for LLMService; returns canned outputs without a model."""

    def __init__(self, json_out=None, text_out="ok"):
        self.json_out = json_out or {}
        self.text_out = text_out
        self.text_calls = 0
        self.messages = None

    def complete_json(self, messages, schema=None, max_new_tokens=None):
        return self.json_out

    def complete(self, messages, temperature=None, max_new_tokens=None,
                 prefix_allowed_tokens_fn=None):
        self.text_calls += 1
        self.messages = messages
        # Allow a list to simulate retry: first bad, then good.
        if isinstance(self.text_out, list):
            return self.text_out[min(self.text_calls - 1, len(self.text_out) - 1)]
        return self.text_out


class GateTest(unittest.TestCase):
    def test_flags_out_of_domain_and_implausible_weather(self):
        kinds = {f["type"] for f in gate_findings("city traffic in the snow")}
        self.assertIn("out_of_domain", kinds)
        self.assertIn("implausible_weather", kinds)

    def test_clean_prompt_has_no_findings(self):
        self.assertEqual(gate_findings("a quiet autumn dawn"), [])

    def test_saturated_prompt_blocks(self):
        findings = gate_findings(
            "Midday city traffic downtown, car horns, sirens and a passing subway train")
        self.assertTrue(any(f.get("type") == "dominant_out_of_domain" for f in findings))
        self.assertTrue(all(f["action"] == "block"
                            for f in findings if f["type"] == "out_of_domain"))

    def test_few_elements_swap_not_block(self):
        findings = gate_findings("autumn dawn in the city with light rain")
        self.assertTrue(findings)
        self.assertTrue(all(f["action"] == "swap" for f in findings))
        self.assertFalse(any(f["type"] == "dominant_out_of_domain" for f in findings))

    def test_negated_mentions_not_flagged(self):
        self.assertEqual(gate_findings("a still autumn dawn, no traffic, no cars"), [])

    def test_concept_synonyms_counted_once(self):
        # car + cars is one concept, so this stays under the block threshold.
        findings = gate_findings("a dawn with cars and a car")
        self.assertTrue(all(f["action"] == "swap" for f in findings))

    def test_off_site_coast_is_a_recoverable_swap(self):
        # The "partial" preset: an in-domain scene + an off-biome (coastal)
        # element -> a single coastal swap finding, so the parser corrects
        # (drops the coast) rather than accepting as-is or rejecting.
        findings = gate_findings(
            "Summer night, a Spotted Nightjar calling, with ocean waves breaking on a beach")
        off = [f for f in findings if f["type"] == "off_biome"]
        self.assertEqual(len(off), 1)
        self.assertEqual(off[0]["action"], "swap")
        self.assertFalse(any(f["type"] == "dominant_out_of_domain" for f in findings))

    def test_inland_scene_has_no_off_biome_finding(self):
        self.assertEqual(
            [f for f in gate_findings("a still autumn dawn with distant birdsong")
             if f["type"] == "off_biome"],
            [],
        )


class FaithfulnessTest(unittest.TestCase):
    def test_pass_when_species_observed(self):
        ok, v = validate_narrative("A Southern Boobook calls at 0:12.", REPORT)
        self.assertTrue(ok)
        self.assertEqual(v, [])

    def test_catches_unobserved_species(self):
        ok, v = validate_narrative("A laughing kookaburra and a galah.", REPORT)
        self.assertFalse(ok)
        self.assertIn("laughing kookaburra", v)
        self.assertIn("galah", v)


class SkillsTest(unittest.TestCase):
    def test_loads_parser_and_reports(self):
        self.assertIn("Prompt Parser", load_skill("parser"))
        self.assertEqual(set(available_skills()),
                         {"parser", "report_analytical", "report_immersive"})

    def test_register_mapping(self):
        self.assertEqual(report_skill_name("immersive"), "report_immersive")
        self.assertEqual(report_skill_name("bogus"), "report_analytical")


class JsonExtractionTest(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(_extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(_extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_embedded(self):
        self.assertEqual(_extract_json('here: {"a": 1} done'), {"a": 1})


class ParserTest(unittest.TestCase):
    def tearDown(self):
        svc_mod._service = None

    def test_corrected_with_weather_and_defaults(self):
        svc_mod._service = _FakeService(json_out={
            "status": "corrected", "note": "swapped", "filled_defaults": [],
            "layer_a": {"season": "autumn", "diel": "dawn"},
            "layer_b": {"weather_type": "rain", "intensity": "light", "duration_s": 10.0},
            "layer_c": {"species": [], "density": "sparse"},
        })
        r = parse_prompt("autumn dawn city with light rain")
        self.assertEqual(r["status"], "corrected")
        self.assertEqual(r["layer_a"], {"season": "autumn", "diel": "dawn"})
        self.assertEqual(r["layer_b"]["weather_type"], "rain")
        self.assertIn("events:empty", r["filled_defaults"])

    def test_weather_off_fills_default_and_nulls_layer_b(self):
        svc_mod._service = _FakeService(json_out={
            "status": "ok", "note": "", "filled_defaults": [],
            "layer_a": {"season": None, "diel": None},
            "layer_b": None, "layer_c": {"species": [], "density": "sparse"},
        })
        r = parse_prompt("a misty dawn")
        self.assertIsNone(r["layer_b"])
        self.assertIn("weather:none", r["filled_defaults"])
        self.assertEqual(r["layer_a"], {"season": None, "diel": None})

    def test_invalid_enums_coerced_to_null(self):
        svc_mod._service = _FakeService(json_out={
            "status": "ok", "note": "", "filled_defaults": [],
            "layer_a": {"season": "monsoon", "diel": "midnight"},
            "layer_b": None, "layer_c": {"species": [], "density": "bogus"},
        })
        r = parse_prompt("x")
        self.assertIsNone(r["layer_a"]["season"])
        self.assertIsNone(r["layer_a"]["diel"])
        self.assertEqual(r["layer_c"]["density"], "sparse")

    def test_weather_backstop_restores_dropped_rain(self):
        # Model slips and nulls layer_b, but the prompt clearly asks for rain.
        svc_mod._service = _FakeService(json_out={
            "status": "ok", "note": "", "filled_defaults": [],
            "layer_a": {"season": None, "diel": None},
            "layer_b": None, "layer_c": {"species": [], "density": "sparse"},
        })
        r = parse_prompt("a quiet dawn with light rain")
        self.assertIsNotNone(r["layer_b"])
        self.assertEqual(r["layer_b"]["weather_type"], "rain")
        self.assertEqual(r["layer_b"]["intensity"], "light")
        self.assertNotIn("weather:none", r["filled_defaults"])

    def test_detect_weather_priority_and_intensity(self):
        self.assertIsNone(_detect_weather("a still summer night"))
        self.assertEqual(_detect_weather("heavy rain")["intensity"], "heavy")
        self.assertEqual(_detect_weather("a thunderstorm")["weather_type"], "rain+wind")
        self.assertEqual(_detect_weather("a gentle breeze")["weather_type"], "wind")

    def test_rejected_nulls_all_layers(self):
        svc_mod._service = _FakeService(json_out={
            "status": "rejected", "note": "try a woodland scene",
            "layer_a": None, "layer_b": None, "layer_c": None,
        })
        r = parse_prompt("a heavy metal concert downtown")
        self.assertEqual(r["status"], "rejected")
        self.assertIsNone(r["layer_a"])

    def test_saturated_prompt_rejected_even_if_model_corrects(self):
        # Fake LLM tries to "correct" into a default bed; the deterministic
        # block must override and reject, nulling all layers.
        svc_mod._service = _FakeService(json_out={
            "status": "corrected", "note": "dropped the city",
            "layer_a": {"season": None, "diel": None},
            "layer_b": None, "layer_c": {"species": [], "density": "sparse"},
        })
        r = parse_prompt(
            "Midday city traffic downtown, car horns, sirens and a passing subway train")
        self.assertEqual(r["status"], "rejected")
        self.assertIsNone(r["layer_a"])
        self.assertIsNone(r["layer_b"])
        self.assertIsNone(r["layer_c"])


class ReportTest(unittest.TestCase):
    def tearDown(self):
        svc_mod._service = None

    def test_faithful_immersive(self):
        fake = _FakeService(text_out="A Southern Boobook calls in the dark.")
        svc_mod._service = fake
        r = write_report(REPORT, "immersive")
        self.assertEqual(r["register"], "immersive")
        self.assertTrue(r["faithful"])
        self.assertEqual(r["source"], "llm")

    def test_full_aggregator_report_serializes_decision_payload(self):
        fake = _FakeService(text_out="A Southern Boobook calls in the dark.")
        svc_mod._service = fake
        write_report(REPORT, "immersive")

        user_payload = json.loads(fake.messages[1]["content"])
        self.assertEqual(user_payload["schema_version"], "analysis_decision.v1")
        self.assertIn("detected_calls", user_payload)
        self.assertNotIn("observations", user_payload)
        self.assertNotIn("task", user_payload)

    def test_retries_then_reports_violation(self):
        # both attempts hallucinate -> faithful False, violation surfaced
        fake = _FakeService(text_out=["A galah sings.", "A galah sings."])
        svc_mod._service = fake
        r = write_report(REPORT, "analytical", max_retries=1)
        self.assertEqual(fake.text_calls, 2)
        self.assertFalse(r["faithful"])
        self.assertIn("galah", r["violations"])

    def test_bad_register_falls_back_to_analytical(self):
        svc_mod._service = _FakeService(text_out="ok")
        r = write_report(REPORT, "bogus")
        self.assertEqual(r["register"], "analytical")


if __name__ == "__main__":
    unittest.main()
