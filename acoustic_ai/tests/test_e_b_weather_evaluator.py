"""Tests for the Layer E-B weather output evaluator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acoustic_ai.layers.layer_e.attempts.murphy__mvp_1__weather_direct_detection.code.evaluate_weather_outputs import (
    load_manifest,
    summarize,
)


def _write_result(path: Path, label: str, warnings: list[str] | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "weather": {
                    "overall_label": label,
                    "warnings": warnings or [],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )


class WeatherEvaluatorTest(unittest.TestCase):
    def test_reports_exact_and_element_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"
            results_dir.mkdir()
            _write_result(results_dir / "rain_a.json", "rain")
            _write_result(results_dir / "mix_a.json", "wind", ["possible_rain_under_wind"])

            manifest = tmp_path / "manifest.csv"
            manifest.write_text(
                "audio_id,expected_label\n"
                "rain_a,rain\n"
                "mix_a,rain+wind\n",
                encoding="utf-8",
            )

            summary = summarize(load_manifest(manifest), manifest, results_dir)

            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["evaluated"], 2)
            self.assertEqual(summary["exact"], 1)
            self.assertEqual(summary["by_expected"]["rain"], {"rain": 1})
            self.assertEqual(summary["by_expected"]["rain+wind"], {"wind": 1})
            self.assertEqual(summary["element_counts"]["rain"]["true_positive"], 1)
            self.assertEqual(summary["element_counts"]["rain"]["false_negative"], 1)
            self.assertEqual(summary["element_counts"]["wind"]["true_positive"], 1)
            self.assertEqual(
                summary["mismatches"],
                [
                    {
                        "audio_id": "mix_a",
                        "expected": "rain+wind",
                        "predicted": "wind",
                        "result_json": str(results_dir / "mix_a.json"),
                        "warnings": "possible_rain_under_wind",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
