import unittest

from acoustic_ai.layers.layer_e.attempts.songke__smoke_3__analysis_aggregator.code.aggregator import (
    aggregate_reports,
)


class AnalysisAggregatorFusionTest(unittest.TestCase):
    def test_events_override_conflicting_ambient_diel(self) -> None:
        result = aggregate_reports(
            ambient_report={
                "estimated_conditions": {"season": "autumn", "diel_bin": "afternoon"},
                "confidence": 0.35,
                "season_confidence": 0.35,
            },
            weather_report={"observations": {"weather": {"confidence": 0.8, "derived_label": "none"}}},
            events_report={
                "events": [
                    {
                        "label": "ninox_boobook",
                        "confidence_mean": 0.91,
                        "onset_s": 12.0,
                        "offset_s": 17.0,
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
        )

        self.assertEqual(result["inferred_context"]["diel"]["estimate"], "night")
        self.assertGreater(result["inferred_context"]["diel"]["posterior"], 0.5)
        self.assertEqual(result["decision"]["time_of_day"]["value"], "night")
        self.assertEqual(result["decision"]["detected_calls"][0]["common_name"], "Southern Boobook")
        self.assertEqual(result["llm_input"]["decision"]["time_of_day"]["value"], "night")
        self.assertIn("immersive", result["llm_input"]["task"])
        self.assertIn("third-person", result["llm_input"]["task"])
        self.assertEqual(result["narration"]["schema_version"], "analysis_narration.v1")
        self.assertIn("Southern Boobook", result["narration"]["summary"])
        self.assertIn("night", result["narration"]["summary"])
        self.assertIn("overall_confidence", result)
        self.assertNotIn("confidence", result)
        self.assertEqual(result["disagreements"][0]["field"], "diel")
        self.assertEqual(result["disagreements"][0]["resolution"], "events_preferred")
        self.assertEqual(result["decision"]["disagreements"][0]["resolution"], "events_preferred")
        self.assertEqual(result["inferred_context"]["season"]["estimate"], "undetermined")

    def test_weak_ambient_alone_does_not_force_precise_context(self) -> None:
        result = aggregate_reports(
            ambient_report={
                "estimated_conditions": {"season": "summer", "diel_bin": "morning"},
                "confidence": 0.22,
                "season_confidence": 0.3,
            }
        )

        self.assertEqual(result["inferred_context"]["season"]["estimate"], "undetermined")
        self.assertEqual(result["inferred_context"]["diel"]["estimate"], "undetermined")
        self.assertEqual(result["disagreements"][0]["resolution"], "low_confidence_range_reported")

    def test_weather_observation_passes_through(self) -> None:
        weather = {
            "wind": {"summary": {"intensity": 0.62, "label": "moderate", "confidence": 0.83}},
            "rain": {"summary": {"intensity": 0.0, "label": "none", "confidence": 0.9}},
            "thunder": {
                "summary": {"intensity": 0.2, "label": "light", "confidence": 0.7},
                "events": [{"start_s": 4.0, "end_s": 5.2, "confidence": 0.66}],
                "mean_interval_s": None,
            },
            "confidence": 0.88,
            "derived_label": "wind",
            "warnings": ["possible_wind_overload"],
        }
        result = aggregate_reports(weather_report={"observations": {"weather": weather}})

        self.assertEqual(result["observations"]["weather"]["derived_label"], "wind")
        self.assertEqual(result["decision"]["weather"]["label"], "wind")
        self.assertEqual(result["decision"]["weather"]["wind"]["label"], "moderate")
        self.assertEqual(result["disagreements"][0]["resolution"], "direct_observation_kept")
        self.assertEqual(result["observations"]["weather"]["warnings"], ["possible_wind_overload"])
        self.assertEqual(result["observations"]["weather"]["wind"]["summary"]["intensity"], 0.62)
        self.assertEqual(result["decision"]["weather"]["thunder"]["events"][0]["onset_s"], 4.0)
        self.assertEqual(result["decision"]["weather"]["thunder"]["events"][0]["offset_s"], 5.2)
        self.assertIsNone(result["decision"]["weather"]["thunder"]["mean_interval_s"])

    def test_strong_ambient_can_be_used_as_context_fallback(self) -> None:
        result = aggregate_reports(
            ambient_report={
                "estimated_conditions": {"season": "summer", "diel_bin": "morning"},
                "confidence": 0.9,
                "season_confidence": 0.9,
            }
        )

        resolutions = {item["resolution"] for item in result["disagreements"]}
        self.assertIn("ambient_used_as_fallback", resolutions)
        fallback = next(item for item in result["disagreements"] if item["resolution"] == "ambient_used_as_fallback")
        self.assertEqual(fallback["events"], "inconclusive")
        self.assertEqual(result["decision"]["time_of_day"]["value"], "morning")
        self.assertEqual(result["decision"]["season"]["value"], "summer")

    def test_broad_event_signal_stays_undetermined_but_preserves_distribution(self) -> None:
        result = aggregate_reports(
            events_report={
                "events": [
                    {
                        "label": "rainbow_bee_eater",
                        "confidence_mean": 0.8,
                        "phenology": {
                            "common_name": "Rainbow Bee-eater",
                            "diel_signal": "day",
                            "diel_confidence": 0.65,
                            "season_signal": "warm_season",
                            "season_confidence": 0.5,
                        },
                    }
                ]
            }
        )

        self.assertEqual(result["inferred_context"]["season"]["estimate"], "undetermined")
        self.assertGreater(result["inferred_context"]["season"]["distribution"]["summer"], 0.25)
        self.assertEqual(result["inferred_context"]["diel"]["estimate"], "undetermined")
        self.assertGreater(result["inferred_context"]["diel"]["distribution"]["morning"], 0.25)


if __name__ == "__main__":
    unittest.main()
