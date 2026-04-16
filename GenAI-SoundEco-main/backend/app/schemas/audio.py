from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AudioMetadata:
    audio_id: str
    file_path: Path
    location_id: str | None = None
    recorded_at: datetime | None = None
    duration: float | None = None
    sample_rate: int | None = None


@dataclass(slots=True)
class AudioSegment:
    segment_index: int
    start_sec: float
    end_sec: float
    waveform: Any


@dataclass(slots=True)
class AudioProcessingResult:
    audio_id: str
    waveform: Any | None
    sample_rate: int | None
    duration: float | None
    num_channels: int | None
    is_valid: bool
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
