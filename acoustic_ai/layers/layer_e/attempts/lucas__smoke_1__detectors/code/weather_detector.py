"""Module E-B — weather intensity detector.

Estimates audible rain, wind, and thunder components from an uploaded audio
clip. The MVP detector will start with spectral heuristics and pretrained
taggers, then use Layer B's curated weather asset index for calibration or a
small supervised head once enough labels accumulate.

Inputs: mel spectrogram (128, T) plus optional waveform stats.
Outputs:
  {
    "rain_intensity": "none|light|medium|heavy|unclear",
    "wind_intensity": "none|light|medium|heavy|unclear",
    "thunder_intensity": "none|light|medium|heavy|unclear",
    "confidence": float,
  }

Shared label source:
  acoustic_ai/layers/layer_b/attempts/lucas__smoke_1__curated_assets/weather_asset_index_schema.md
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_WEATHER_ASSET_INDEX = (
    PROJECT_ROOT
    / "acoustic_ai"
    / "layers"
    / "layer_b"
    / "attempts"
    / "lucas__smoke_1__curated_assets"
    / "data"
    / "weather"
    / "asset_index.csv"
)

WEATHER_COMPONENTS = ("rain", "wind", "thunder")
REQUIRED_ASSET_COLUMNS = {
    "asset_id",
    "clip_path",
    "primary_weather",
    "has_rain",
    "has_wind",
    "has_thunder",
    "rain_intensity",
    "wind_intensity",
    "thunder_intensity",
    "analysis_use",
    "analysis_label_quality",
    "recording_group_id",
    "near_duplicate_group_id",
}


def load_weather_asset_index(
    index_path: Path = DEFAULT_WEATHER_ASSET_INDEX,
) -> list[dict[str, str]]:
    """Load the Layer B weather asset index for E-B calibration/training."""
    if not index_path.exists():
        raise FileNotFoundError(f"Weather asset index not found: {index_path}")

    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fieldnames = set(rows[0].keys()) if rows else set()
    missing = sorted(REQUIRED_ASSET_COLUMNS - fieldnames)
    if missing:
        raise ValueError(
            f"Weather asset index is missing required columns: {', '.join(missing)}"
        )

    return rows


def summarize_weather_asset_index(
    index_path: Path = DEFAULT_WEATHER_ASSET_INDEX,
) -> dict:
    """Return a compact summary of the shared Layer B / E-B label pool."""
    rows = load_weather_asset_index(index_path)
    component_counts = {
        component: sum(
            1 for row in rows if _truthy_or_unclear(row.get(f"has_{component}", ""))
        )
        for component in WEATHER_COMPONENTS
    }

    return {
        "asset_count": len(rows),
        "primary_weather": dict(Counter(row["primary_weather"] for row in rows)),
        "analysis_use": dict(Counter(row["analysis_use"] for row in rows)),
        "analysis_label_quality": dict(
            Counter(row["analysis_label_quality"] for row in rows)
        ),
        "component_counts": component_counts,
        "recording_group_count": len({row["recording_group_id"] for row in rows}),
        "near_duplicate_group_count": len(
            {row["near_duplicate_group_id"] for row in rows}
        ),
    }


def _truthy_or_unclear(value: str) -> bool:
    return (value or "").strip().lower() in {"true", "yes", "1", "unclear"}


# TODO: implement detection (start with spectral heuristics + PANNs/CLAP).
