"""Runtime loader for Layer C retrieval event indexes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EventSnippet:
    snippet_id: str
    event_type: str
    species_common_name: str
    species_scientific_name: str
    audio_event_id: str
    audio_path: str
    score: float
    quality_score: float | None
    diel_bin: str
    season: str
    duration_s: float
    recording_id: str
    event_start_seconds: float | None
    event_end_seconds: float | None
    source_manifest: str
    verdict: str
    notes: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_index(path: Path) -> list[EventSnippet]:
    snippets: list[EventSnippet] = []
    for row in read_csv(path):
        snippets.append(
            EventSnippet(
                snippet_id=row["snippet_id"],
                event_type=row["event_type"],
                species_common_name=row["species_common_name"],
                species_scientific_name=row["species_scientific_name"],
                audio_event_id=row["audio_event_id"],
                audio_path=row["audio_path"],
                score=parse_float(row.get("score")) or 0.0,
                quality_score=parse_float(row.get("quality_score")),
                diel_bin=row.get("diel_bin", ""),
                season=row.get("season", "unknown"),
                duration_s=parse_float(row.get("duration_s")) or 0.0,
                recording_id=row.get("recording_id", ""),
                event_start_seconds=parse_float(row.get("event_start_seconds")),
                event_end_seconds=parse_float(row.get("event_end_seconds")),
                source_manifest=row.get("source_manifest", ""),
                verdict=row.get("verdict", ""),
                notes=row.get("notes", ""),
            )
        )
    return snippets
