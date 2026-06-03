"""Build and load the Layer C event retrieval index.

The retrieval baseline uses real audited event snippets, not generated audio.
This module normalises prior Layer C audit/reference CSVs into one compact index
that the selector and scheduler can consume.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_CUCKOO_REFERENCE_BANK = (
    REPO_ROOT
    / "resources"
    / "site_257_bowra-dry-a"
    / "layer_c_smoke_fairywren_robin_bellbird"
    / "bronze_cuckoo_natural_core_v1"
    / "manual_audit_horsfields_bronze_cuckoo_pass24_trainset.csv"
)
DEFAULT_CUCKOO_MANIFEST = (
    REPO_ROOT
    / "resources"
    / "site_257_bowra-dry-a"
    / "layer_c_smoke_fairywren_robin_bellbird"
    / "bronze_cuckoo_natural_core_v1"
    / "manifest.csv"
)
DEFAULT_FAIRYWREN_AUDIT = (
    REPO_ROOT
    / "resources"
    / "site_257_bowra-dry-a"
    / "layer_c_smoke_fairywren_robin_bellbird"
    / "natural_core_v1"
    / "manual_audit_splendid_fairywren_natural_core_top55.csv"
)
DEFAULT_FAIRYWREN_RETRIEVAL_AUDIT = (
    REPO_ROOT
    / "resources"
    / "site_257_bowra-dry-a"
    / "layer_c_retrieval_cuckoo_fairywren"
    / "fairywren"
    / "retrieval_pool_v3_target30"
    / "manual_audit_splendid_fairywren_retrieval_pass23.csv"
)
DEFAULT_SHARED_PREPARED_MANIFEST = (
    REPO_ROOT
    / "resources"
    / "site_257_bowra-dry-a"
    / "layer_c_smoke_fairywren_robin_bellbird"
    / "prepared_manifest.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "acoustic_ai"
    / "data"
    / "events"
    / "retrieval"
    / "layer_c_event_index.csv"
)

INDEX_COLUMNS = [
    "snippet_id",
    "event_type",
    "species_common_name",
    "species_scientific_name",
    "audio_event_id",
    "audio_path",
    "score",
    "quality_score",
    "diel_bin",
    "season",
    "duration_s",
    "recording_id",
    "event_start_seconds",
    "event_end_seconds",
    "source_manifest",
    "verdict",
    "notes",
]


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

    def to_row(self) -> dict[str, str]:
        return {
            "snippet_id": self.snippet_id,
            "event_type": self.event_type,
            "species_common_name": self.species_common_name,
            "species_scientific_name": self.species_scientific_name,
            "audio_event_id": self.audio_event_id,
            "audio_path": self.audio_path,
            "score": f"{self.score:.4f}",
            "quality_score": "" if self.quality_score is None else f"{self.quality_score:.6f}",
            "diel_bin": self.diel_bin,
            "season": self.season,
            "duration_s": f"{self.duration_s:.3f}",
            "recording_id": self.recording_id,
            "event_start_seconds": (
                "" if self.event_start_seconds is None else f"{self.event_start_seconds:.3f}"
            ),
            "event_end_seconds": (
                "" if self.event_end_seconds is None else f"{self.event_end_seconds:.3f}"
            ),
            "source_manifest": self.source_manifest,
            "verdict": self.verdict,
            "notes": self.notes,
        }


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


def infer_season(aest_datetime: str) -> str:
    if not aest_datetime:
        return "unknown"
    try:
        month = datetime.fromisoformat(aest_datetime).month
    except ValueError:
        return "unknown"
    if month in (12, 1, 2):
        return "summer"
    if month in (3, 4, 5):
        return "autumn"
    if month in (6, 7, 8):
        return "winter"
    return "spring"


def manifest_by_event_id(path: Path) -> dict[str, dict[str, str]]:
    return {row["audio_event_id"]: row for row in read_csv(path)}


def normalise_path(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)
    return path


def build_cuckoo_snippets(reference_bank: Path, manifest: Path) -> list[EventSnippet]:
    manifest_rows = manifest_by_event_id(manifest)
    snippets: list[EventSnippet] = []

    for row in read_csv(reference_bank):
        event_id = row["audio_event_id"]
        event = manifest_rows.get(event_id, {})
        rank = (
            row.get("reference_rank")
            or row.get("audit_index")
            or row.get("smoke_event_index")
            or str(len(snippets) + 1)
        )
        duration = parse_float(row.get("crop_duration_seconds"))
        if duration is None:
            duration = parse_float(event.get("event_duration_seconds")) or 0.0

        snippets.append(
            EventSnippet(
                snippet_id=f"horsfields_bronze_cuckoo_ref{rank}_{event_id}",
                event_type="horsfields_bronze_cuckoo",
                species_common_name="Horsfield's Bronze-cuckoo",
                species_scientific_name="Chrysococcyx basalis",
                audio_event_id=event_id,
                audio_path=normalise_path(row.get("crop_audio_path") or row["audio_path"]),
                score=parse_float(row.get("score")) or parse_float(event.get("score")) or 0.0,
                quality_score=parse_float(row.get("quality_score")),
                diel_bin=event.get("diel_bin", ""),
                season=infer_season(event.get("event_start_datetime_aest", "")),
                duration_s=duration,
                recording_id=event.get("audio_recording_id", ""),
                event_start_seconds=parse_float(event.get("event_start_seconds")),
                event_end_seconds=parse_float(event.get("event_end_seconds")),
                source_manifest=normalise_path(str(reference_bank)),
                verdict=row.get("verdict") or row.get("manual_verdict") or "Pass",
                notes=row.get("notes") or row.get("manual_notes", ""),
            )
        )

    return snippets


def build_fairywren_snippets(audit_csv: Path, prepared_manifest: Path) -> list[EventSnippet]:
    """Normalise audited Splendid Fairywren snippets into retrieval rows.

    Only explicit Pass rows enter the default retrieval pool. The broader
    natural-core shortlist can still be supplied manually for experiments.
    """

    prepared_rows = manifest_by_event_id(prepared_manifest)
    snippets: list[EventSnippet] = []

    for row in read_csv(audit_csv):
        event_id = row["audio_event_id"]
        event = prepared_rows.get(event_id, {})
        verdict = row.get("verdict", "").strip()
        if verdict.casefold() != "pass":
            continue
        rank = row.get("natural_rank") or row.get("audit_index") or str(len(snippets) + 1)
        duration = (
            parse_float(row.get("crop_duration_seconds"))
            or parse_float(event.get("event_duration_seconds"))
            or 0.0
        )

        snippets.append(
            EventSnippet(
                snippet_id=f"splendid_fairywren_natural{rank}_{event_id}",
                event_type="splendid_fairywren",
                species_common_name="Splendid Fairywren",
                species_scientific_name="Malurus splendens",
                audio_event_id=event_id,
                audio_path=normalise_path(row["audio_path"]),
                score=parse_float(row.get("score")) or parse_float(event.get("score")) or 0.0,
                quality_score=(
                    parse_float(row.get("natural_core_score"))
                    or parse_float(row.get("quality_score"))
                ),
                diel_bin=row.get("diel_bin") or event.get("diel_bin", ""),
                season=infer_season(event.get("event_start_datetime_aest", "")),
                duration_s=duration,
                recording_id=event.get("audio_recording_id", ""),
                event_start_seconds=parse_float(event.get("event_start_seconds")),
                event_end_seconds=parse_float(event.get("event_end_seconds")),
                source_manifest=normalise_path(str(row.get("source_manifest") or audit_csv)),
                verdict=verdict,
                notes=row.get("notes", ""),
            )
        )

    return snippets


def write_index(snippets: Iterable[EventSnippet], output: Path) -> None:
    rows = [snippet.to_row() for snippet in snippets]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_index(path: Path = DEFAULT_OUTPUT) -> list[EventSnippet]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Layer C retrieval event index.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cuckoo-reference-bank", type=Path, default=DEFAULT_CUCKOO_REFERENCE_BANK)
    parser.add_argument("--cuckoo-manifest", type=Path, default=DEFAULT_CUCKOO_MANIFEST)
    parser.add_argument("--fairywren-audit", type=Path, default=DEFAULT_FAIRYWREN_RETRIEVAL_AUDIT)
    parser.add_argument("--prepared-manifest", type=Path, default=DEFAULT_SHARED_PREPARED_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snippets = build_cuckoo_snippets(args.cuckoo_reference_bank, args.cuckoo_manifest)
    snippets.extend(build_fairywren_snippets(args.fairywren_audit, args.prepared_manifest))
    write_index(snippets, args.output)
    print(f"Written {len(snippets)} snippets to {args.output}")
    for snippet in snippets[:5]:
        print(
            f"- {snippet.snippet_id}: score={snippet.score:.4f} "
            f"diel={snippet.diel_bin} season={snippet.season}"
        )


if __name__ == "__main__":
    main()
