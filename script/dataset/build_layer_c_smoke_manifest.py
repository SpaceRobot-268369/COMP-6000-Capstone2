#!/usr/bin/env python3
"""Build a small Layer C event manifest for smoke tests.

The full site 257 annotation archive is too large for smoke-test audio
downloads. This script scans the downloaded annotation CSVs, applies the Layer C
quality filter, and writes a deterministic manifest with a fixed number of
high-confidence annotated event segments per species-level event type.

Default smoke policy:
  - event types:
      1. Southern Boobook / Ninox boobook, nocturnal owl call
      2. Splendid Fairywren / Malurus splendens, common dawn/diurnal passerine
  - 50 selected annotation events per event type
  - source: BirdNET.results.csv
  - score >= 0.9
  - raw annotation duration in [1, 10] seconds
  - event-type-specific diel preference
  - spread selected events across distinct recordings before reusing recordings
  - extract with the project standard +/-3 second buffer

Usage:
  python3 script/dataset/build_layer_c_smoke_manifest.py
  python3 script/dataset/build_layer_c_smoke_manifest.py --event-type boobook --segments-per-type 50
  python3 script/dataset/build_layer_c_smoke_manifest.py --event-type boobook --event-type splendid_fairywren --dry-run
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "resources" / "site_257_bowra-dry-a"
ITEMS_CSV = BASE_DIR / "site_257_all_items.csv"
ANNOTATIONS_DIR = BASE_DIR / "all_items_annotation"
OUT_DIR = BASE_DIR / "smoking_test_1_layer_C_dataset_1"
MANIFEST_PATH = OUT_DIR / "manifest.csv"

AEST = timezone(timedelta(hours=10))
DEFAULT_BUFFER_SECONDS = 3.0


@dataclass(frozen=True)
class EventType:
    event_type: str
    common_name: str
    scientific_name: str
    prompt_label: str
    preferred_diel: tuple[str, ...]


DEFAULT_EVENT_TYPES: dict[str, EventType] = {
    "boobook": EventType(
        event_type="boobook",
        common_name="Southern Boobook",
        scientific_name="Ninox boobook",
        prompt_label="Southern Boobook owl call, nocturnal bird vocal event",
        preferred_diel=("night", "evening", "dusk"),
    ),
    "splendid_fairywren": EventType(
        event_type="splendid_fairywren",
        common_name="Splendid Fairywren",
        scientific_name="Malurus splendens",
        prompt_label="Splendid Fairywren bird call, dawn woodland passerine vocal event",
        preferred_diel=("dawn", "morning", "afternoon", "dusk"),
    ),
}
DEFAULT_EVENT_TYPE_ORDER = ("boobook", "splendid_fairywren")


@dataclass(frozen=True)
class Candidate:
    event_type: EventType
    item_count: int
    recording_id: str
    recording_duration_seconds: float
    audio_event_id: str
    score: float
    event_start_seconds: float
    event_end_seconds: float
    event_duration_seconds: float
    event_start_datetime_utc: str
    event_start_datetime_aest: str
    diel_bin: str
    common_name_tags: str
    species_name_tags: str
    other_tags: str
    import_file_name: str
    annotation_csv: Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "event"


def parse_event_type(value: str) -> EventType:
    """Parse a preset key or common/scientific event type spec."""
    if value in DEFAULT_EVENT_TYPES:
        return DEFAULT_EVENT_TYPES[value]

    parts = [part.strip() for part in value.split("|")]
    if len(parts) not in (2, 3):
        raise ValueError(
            "--event-type must be a preset key or 'common name|scientific name'"
        )

    common_name, scientific_name = parts[0], parts[1]
    event_type = parts[2] if len(parts) == 3 else slugify(common_name)
    if not common_name or not scientific_name:
        raise ValueError("--event-type common and scientific names must be non-empty")

    return EventType(
        event_type=event_type,
        common_name=common_name,
        scientific_name=scientific_name,
        prompt_label=f"{common_name} bird call, Bowra woodland vocal event",
        preferred_diel=("dawn", "morning", "afternoon", "dusk", "evening", "night"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Layer C smoke-test event manifest.")
    parser.add_argument("--items-csv", type=Path, default=ITEMS_CSV)
    parser.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--event-type",
        action="append",
        default=None,
        help=(
            "Event type preset or spec. Presets: boobook, splendid_fairywren. "
            "Custom form: 'Common Name|Scientific name'. Repeat for two types."
        ),
    )
    parser.add_argument("--segments-per-type", type=int, default=50)
    parser.add_argument(
        "--target-events",
        type=int,
        default=None,
        help="Deprecated alias for one event type's segment count.",
    )
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--buffer-seconds", type=float, default=DEFAULT_BUFFER_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_event_types(args: argparse.Namespace) -> list[EventType]:
    if args.event_type:
        return [parse_event_type(value) for value in args.event_type]
    return [DEFAULT_EVENT_TYPES[key] for key in DEFAULT_EVENT_TYPE_ORDER]


def load_items(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_utc_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def diel_bin_from_aest(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    hour = dt.astimezone(AEST).hour
    if 5 <= hour < 8:
        return "dawn"
    if 8 <= hour < 11:
        return "morning"
    if 11 <= hour < 13:
        return "midday"
    if 13 <= hour < 16:
        return "afternoon"
    if 16 <= hour < 19:
        return "dusk"
    if 19 <= hour < 22:
        return "evening"
    return "night"


def row_matches_species(row: dict[str, str], event_type: EventType) -> bool:
    label_text = "|".join(
        [
            row.get("common_name_tags", ""),
            row.get("species_name_tags", ""),
            row.get("other_tags", ""),
        ]
    )
    return event_type.common_name in label_text or event_type.scientific_name in label_text


def iter_candidates(args: argparse.Namespace, event_types: list[EventType]) -> list[Candidate]:
    items = load_items(args.items_csv)
    candidates: list[Candidate] = []

    for annotation_csv in sorted(args.annotations_dir.glob("site_257_item_*/site_257_item_*.csv")):
        if annotation_csv.stat().st_size == 0:
            continue
        with annotation_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                matched_event_type = next(
                    (
                        event_type
                        for event_type in event_types
                        if row_matches_species(row, event_type)
                    ),
                    None,
                )
                if matched_event_type is None:
                    continue
                if (row.get("audio_event_import_file_name") or "") != "BirdNET.results.csv":
                    continue

                recording_id = (row.get("audio_recording_id") or "").strip()
                item = items.get(recording_id)
                if item is None:
                    continue

                score = parse_float(row.get("score"))
                event_start = parse_float(row.get("event_start_seconds"))
                event_end = parse_float(row.get("event_end_seconds"))
                recording_duration = parse_float(item.get("duration_seconds"))
                if (
                    score is None
                    or event_start is None
                    or event_end is None
                    or recording_duration is None
                ):
                    continue
                if score < args.min_score or event_end <= event_start:
                    continue

                event_duration = event_end - event_start
                if event_duration < args.min_duration or event_duration > args.max_duration:
                    continue

                utc_text = row.get("event_start_datetime_utc_00_00", "")
                utc_dt = parse_utc_datetime(utc_text)
                aest_dt = utc_dt.astimezone(AEST) if utc_dt else None

                candidates.append(
                    Candidate(
                        event_type=matched_event_type,
                        item_count=int(item["count"]),
                        recording_id=recording_id,
                        recording_duration_seconds=recording_duration,
                        audio_event_id=(row.get("audio_event_id") or "").strip(),
                        score=score,
                        event_start_seconds=event_start,
                        event_end_seconds=event_end,
                        event_duration_seconds=event_duration,
                        event_start_datetime_utc=utc_text,
                        event_start_datetime_aest=(
                            aest_dt.isoformat(timespec="seconds") if aest_dt else ""
                        ),
                        diel_bin=diel_bin_from_aest(utc_dt),
                        common_name_tags=row.get("common_name_tags", ""),
                        species_name_tags=row.get("species_name_tags", ""),
                        other_tags=row.get("other_tags", ""),
                        import_file_name=row.get("audio_event_import_file_name", ""),
                        annotation_csv=annotation_csv,
                    )
                )

    return candidates


def select_for_event_type(
    candidates: list[Candidate],
    event_type: EventType,
    segments_per_type: int,
) -> list[Candidate]:
    preferred_rank = {name: index for index, name in enumerate(event_type.preferred_diel)}
    event_candidates = [c for c in candidates if c.event_type == event_type]
    ranked = sorted(
        event_candidates,
        key=lambda c: (
            preferred_rank.get(c.diel_bin, len(preferred_rank)),
            -c.score,
            c.item_count,
            c.audio_event_id,
        ),
    )

    selected: list[Candidate] = []
    used_recordings: set[str] = set()
    for candidate in ranked:
        if candidate.recording_id in used_recordings:
            continue
        selected.append(candidate)
        used_recordings.add(candidate.recording_id)
        if len(selected) >= segments_per_type:
            return selected

    for candidate in ranked:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= segments_per_type:
            break

    return selected


def select_candidates(
    candidates: list[Candidate],
    event_types: list[EventType],
    segments_per_type: int,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for event_type in event_types:
        selected.extend(select_for_event_type(candidates, event_type, segments_per_type))
    return selected


def manifest_row(
    candidate: Candidate,
    row_index: int,
    event_type_index: int,
    buffer_seconds: float,
    output_path: Path,
) -> dict[str, str]:
    extracted_start = max(0.0, candidate.event_start_seconds - buffer_seconds)
    extracted_end = min(
        candidate.recording_duration_seconds,
        candidate.event_end_seconds + buffer_seconds,
    )
    output_dir = output_path.parent
    if output_dir.is_absolute():
        segment_base = output_dir / "segments"
    else:
        segment_base = output_dir / "segments"
    segment_path = (
        segment_base
        / f"site_257_item_{candidate.recording_id}"
        / f"site_257_item_{candidate.recording_id}_audioevent_{candidate.audio_event_id}"
        / f"site_257_item_{candidate.recording_id}_audioevent_{candidate.audio_event_id}.webm"
    )

    caption = (
        f"{candidate.event_type.prompt_label}, Bowra dry woodland, "
        f"{candidate.diel_bin}, BirdNET score {candidate.score:.4f}"
    )

    return {
        "smoke_event_index": str(row_index),
        "event_type_index": str(event_type_index),
        "event_type": candidate.event_type.event_type,
        "species_common_name": candidate.event_type.common_name,
        "species_scientific_name": candidate.event_type.scientific_name,
        "audio_recording_id": candidate.recording_id,
        "item_count": str(candidate.item_count),
        "audio_event_id": candidate.audio_event_id,
        "score": f"{candidate.score:.4f}",
        "event_start_seconds": f"{candidate.event_start_seconds:.3f}",
        "event_end_seconds": f"{candidate.event_end_seconds:.3f}",
        "event_duration_seconds": f"{candidate.event_duration_seconds:.3f}",
        "buffer_seconds": f"{buffer_seconds:.3f}",
        "extracted_start_seconds": f"{extracted_start:.3f}",
        "extracted_end_seconds": f"{extracted_end:.3f}",
        "extracted_duration_seconds": f"{extracted_end - extracted_start:.3f}",
        "event_start_datetime_utc": candidate.event_start_datetime_utc,
        "event_start_datetime_aest": candidate.event_start_datetime_aest,
        "diel_bin": candidate.diel_bin,
        "annotation_source": candidate.import_file_name,
        "annotation_csv": str(candidate.annotation_csv.relative_to(REPO_ROOT)),
        "segment_path": str(segment_path),
        "caption": caption,
    }


def main() -> None:
    args = parse_args()
    if args.target_events is not None:
        args.segments_per_type = args.target_events
    if args.segments_per_type < 1:
        raise ValueError("--segments-per-type must be >= 1")
    if args.buffer_seconds < 0:
        raise ValueError("--buffer-seconds must be >= 0")
    if not args.items_csv.exists():
        raise FileNotFoundError(f"items CSV not found: {args.items_csv}")
    if not args.annotations_dir.exists():
        raise FileNotFoundError(f"annotations dir not found: {args.annotations_dir}")

    event_types = resolve_event_types(args)
    candidates = iter_candidates(args, event_types)
    selected = select_candidates(candidates, event_types, args.segments_per_type)
    per_type_index: dict[str, int] = {}
    rows = []
    for index, candidate in enumerate(selected, start=1):
        per_type_index[candidate.event_type.event_type] = (
            per_type_index.get(candidate.event_type.event_type, 0) + 1
        )
        rows.append(
            manifest_row(
                candidate=candidate,
                row_index=index,
                event_type_index=per_type_index[candidate.event_type.event_type],
                buffer_seconds=args.buffer_seconds,
                output_path=args.output,
            )
        )

    print(
        f"Candidates: {len(candidates)} after event-type/source/score/duration filters; "
        f"selected: {len(rows)}"
    )
    for event_type in event_types:
        candidate_count = sum(1 for c in candidates if c.event_type == event_type)
        selected_count = sum(1 for r in rows if r["event_type"] == event_type.event_type)
        print(
            f"- {event_type.event_type}: candidates={candidate_count} "
            f"selected={selected_count} species={event_type.common_name}"
        )
    for row in rows[:10]:
        print(
            f"{row['smoke_event_index']}. type={row['event_type']} "
            f"recording={row['audio_recording_id']} event={row['audio_event_id']} "
            f"score={row['score']} diel={row['diel_bin']} "
            f"extract={row['extracted_duration_seconds']}s"
        )
    if len(rows) > 10:
        print(f"... {len(rows) - 10} more selected rows")

    if args.dry_run:
        print("Dry run - no file written.")
        return

    if not rows:
        raise RuntimeError("No candidates selected; no manifest written.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
