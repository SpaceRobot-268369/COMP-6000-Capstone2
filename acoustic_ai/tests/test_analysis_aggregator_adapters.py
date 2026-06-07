import unittest

from acoustic_ai.layers.layer_e.attempts.songke__smoke_3__analysis_aggregator.code.adapters import (
    adapt_head_reports,
)


class AnalysisAggregatorAdaptersTest(unittest.TestCase):
    def test_adapts_ambient_weather_and_event_observations(self) -> None:
        adapted = adapt_head_reports(
            ambient_report={
                "estimated_conditions": {
                    "season": "autumn",
                    "diel_bin": "night",
                    "hour": 22.4,
                    "month": 4.1,
                },
                "similar_clips": [{"segment_id": "seg_001", "similarity": 0.71}],
                "confidence": 0.35,
                "season_confidence": 0.42,
                "ood_flag": False,
            },
            weather_report={
                "observations": {
                    "weather": {
                        "wind": {"summary": {"intensity": 0.62, "label": "moderate", "confidence": 0.83}},
                        "rain": {"summary": {"intensity": 0.1, "label": "light", "confidence": 0.55}},
                        "thunder": {"summary": {"intensity": 0.0, "label": "none", "confidence": 0.9}},
                        "confidence": 0.8,
                        "derived_label": "rain+wind",
                        "warnings": ["weather_mixed_with_ambient"],
                    }
                }
            },
            events_report={
                "events": [
                    {
                        "label": "ninox_boobook",
                        "onset_s": 12.4,
                        "offset_s": 17.1,
                        "confidence_mean": 0.91,
                        "phenology": {
                            "common_name": "Southern Boobook",
                            "scientific_name": "Ninox boobook",
                            "diel_signal": "night",
                            "diel_confidence": 0.85,
                            "season_signal": "weak",
                            "season_confidence": 0.2,
                        },
                    },
                ]
            },
        )

        self.assertEqual(adapted["observations"]["ambient"]["estimated_conditions"]["season"], "autumn")
        self.assertEqual(adapted["observations"]["weather"]["derived_label"], "rain+wind")
        self.assertEqual(adapted["observations"]["events"][0]["common_name"], "Southern Boobook")
        self.assertEqual(adapted["evidence"]["diel"][0]["source_head"], "ambient")
        self.assertEqual(adapted["evidence"]["diel"][1]["source_head"], "events")
        self.assertEqual(adapted["evidence"]["diel"][1]["candidates"], ["night"])
        self.assertEqual(
            adapted["evidence"]["season"],
            [
                {
                    "source_head": "ambient",
                    "field": "season",
                    "value": "autumn",
                    "candidates": ["autumn"],
                    "confidence": 0.35,
                    "reason": "Ambient head estimated autumn.",
                }
            ],
        )


    def test_accepts_registry_wrapped_reports_and_fills_weather_defaults(self) -> None:
        adapted = adapt_head_reports(
            ambient_report={"report": {"estimated_conditions": {"season": "summer", "diel_bin": "afternoon"}}},
            weather_report={"report": {}},
            events_report={"report": {"events": []}},
        )

        self.assertEqual(adapted["observations"]["weather"]["derived_label"], "none")
        self.assertEqual(adapted["observations"]["weather"]["wind"]["summary"]["label"], "none")
        self.assertEqual(adapted["evidence"]["season"][0]["value"], "summer")
        self.assertEqual(adapted["evidence"]["diel"][0]["value"], "afternoon")


    def test_maps_broad_phenology_signals_to_candidate_bins(self) -> None:
        adapted = adapt_head_reports(
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

        self.assertEqual(adapted["evidence"]["diel"][0]["candidates"], ["morning", "afternoon"])
        self.assertEqual(adapted["evidence"]["season"][0]["candidates"], ["spring", "summer", "autumn"])
        self.assertEqual(adapted["evidence"]["season"][0]["confidence"], 0.4)


if __name__ == "__main__":
    unittest.main()
