import unittest

from acoustic_ai.layers.layer_e.attempts.songke__smoke_3__analysis_aggregator.code import handler


class AnalysisAggregatorHandlerTest(unittest.TestCase):
    def test_load_and_aggregate_reports(self) -> None:
        state = handler.load(params={"fusion": {"undetermined_threshold": 0.45}})
        result = handler.aggregate(
            state,
            ambient_report={
                "estimated_conditions": {"season": "autumn", "diel_bin": "afternoon"},
                "confidence": 0.35,
                "season_confidence": 0.35,
            },
            events_report={
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
        )

        self.assertEqual(result["schema_version"], "analysis_aggregator.v1")
        self.assertEqual(result["inferred_context"]["diel"]["estimate"], "night")

    def test_generate_is_not_supported(self) -> None:
        state = handler.load()
        with self.assertRaises(NotImplementedError):
            handler.generate(state, seed=42)


if __name__ == "__main__":
    unittest.main()
