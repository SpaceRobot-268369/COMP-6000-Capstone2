"""Runtime configuration for the demo mock AI service."""

from __future__ import annotations

import os
from pathlib import Path

FIXTURES = Path(os.environ.get("MOCK_FIXTURES_ROOT", "/fixtures"))
REGISTRY_PATH = Path(os.environ.get("MOCK_REGISTRY_PATH", "/registry.yaml"))

# Fake think-time, so the frontend's staged progress text has something to
# narrate instead of flashing past in one frame.
LATENCY_MS = {
    "parse": int(os.environ.get("MOCK_LATENCY_MS_PARSE", "800")),
    "generate": int(os.environ.get("MOCK_LATENCY_MS_GENERATE", "3000")),
    "analysis": int(os.environ.get("MOCK_LATENCY_MS_ANALYSIS", "1500")),
    "narrative": int(os.environ.get("MOCK_LATENCY_MS_NARRATIVE", "400")),
}

SEASONS = ("spring", "summer", "autumn", "winter")
DIELS = ("dawn", "morning", "afternoon", "night")
CELLS = tuple(f"{s}_{d}" for s in SEASONS for d in DIELS)

DEFAULT_CELL = "autumn_dawn"
