from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class EnvironmentRecord:
    location_id: str
    timestamp: datetime
    temperature: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    rainfall: float | None = None
    pressure: float | None = None


@dataclass(slots=True)
class AlignedEnvironmentRecord:
    audio_id: str
    matched: bool
    matched_timestamp: datetime | None = None
    time_diff_minutes: float | None = None
    temperature: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    rainfall: float | None = None
    pressure: float | None = None
