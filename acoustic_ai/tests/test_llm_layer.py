"""Tests for the in-process LLM-OSS layer (parser, report writer, gate,
faithfulness, skills loader, JSON extraction).

Model-free: the LLMService singleton is replaced with a fake so these run
locally without torch/transformers or a downloaded model. Real
model-dependent checks are serverB-only (plan §7).
"""

from __future__ import annotations

import unittest

import acoustic_ai.llm.service as svc_mod
from acoustic_ai.llm import parse_prompt, write_report
from acoustic_ai.llm.faithfulness import validate_narrative
from acoustic_ai.llm.gate import gate_findings
from acoustic_ai.llm.service import _extract_json
from acoustic_ai.llm.skills import available_skills, load_skill, report_skill_name

REPORT = {"observations": {"events": [{"label": "Southern Boobook", "onset_s": 12.4}]}}


class _FakeService:
    """Stand-in for LLMService; returns canned outputs without a model."""

    def __init__(self, json_out=None, text_out="ok"):
        self.json_out = json_out or {}
        self.text_out = text_out
        self.text_calls = 0

    def complete_json(self, messages, schema=None, max_new_tokens=None):
        return self.json_out

    def complete(self, messages, temperature=None, max_new_tokens=None,
                 prefix_allowed_tokens_fn=None):
        self.text_calls += 1
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

    def test_rejected_nulls_all_layers(self):
        svc_mod._service = _FakeService(json_out={
            "status": "rejected", "note": "try a woodland scene",
            "layer_a": None, "layer_b": None, "layer_c": None,
        })
        r = parse_prompt("a heavy metal concert downtown")
        self.assertEqual(r["status"], "rejected")
        self.assertIsNone(r["layer_a"])


class ReportTest(unittest.TestCase):
    def tearDown(self):
        svc_mod._service = None

    def test_faithful_immersive(self):
        svc_mod._service = _FakeService(text_out="A Southern Boobook calls in the dark.")
        r = write_report(REPORT, "immersive")
        self.assertEqual(r["register"], "immersive")
        self.assertTrue(r["faithful"])

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
