#!/usr/bin/env python3
"""Build a stricter Layer C smoke-test manifest for 2-3 bird species.

This is the first step of the short Layer C smoke workflow:

1. Filter annotation rows into high-confidence, downloadable event snippets.
2. Spread selected snippets across recordings.
3. Write a manifest for exact segment download plus a rejection/report trail.

The script only uses annotation metadata. Audio-quality rejection happens after
segment download, when WAVs/spectrograms can be inspected.
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
OUT_DIR = BASE_DIR / "layer_c_smoke_2_3_species"
MANIFEST_PATH = OUT_DIR / "manifest.csv"
REJECTED_PATH = OUT_DIR / "rejected_manifest.csv"
REPORT_PATH = OUT_DIR / "filter_report.md"

AEST = timezone(timedelta(hours=10))
DEFAULT_BUFFER_SECONDS = 3.0


@dataclass(frozen=True)
class SpeciesSpec:
    key: str
    common_name: str
    scientific_name: str
    prompt_label: str
    preferred_diel: tuple[str, ...]


SPECIES_PRESETS: dict[str, SpeciesSpec] = {
    "splendid_fairywren": SpeciesSpec(
        key="splendid_fairywren",
        common_name="Splendid Fairywren",
        scientific_name="Malurus splendens",
        prompt_label="Splendid Fairywren bird call, Bowra dry woodland",
        preferred_diel=("dawn", "morning", "afternoon", "dusk"),
    ),
    "chestnut_rumped_thornbill": SpeciesSpec(
        key="chestnut_rumped_thornbill",
        common_name="Chestnut-rumped Thornbill",
        scientific_name="Acanthiza uropygialis",
        prompt_label="Chestnut-rumped Thornbill bird call, Bowra dry woodland",
        preferred_diel=("dawn", "morning", "afternoon", "dusk"),
    ),
    "boobook": SpeciesSpec(
        key="boobook",
        common_name="Southern Boobook",
        scientific_name="Ninox boobook",
        prompt_label="Southern Boobook owl call, nocturnal Bowra woodland",
        preferred_diel=("night", "evening", "dusk"),
    ),
}

DEFAULT_SPECIES_ORDER = (
    "splendid_fairywren",
    "chestnut_rumped_thornbill",
    "boobook",
)


@dataclass(frozen=True)
class Candidate:
    species: SpeciesSpec
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


def parse_species(value: str) -> SpeciesSpec:
    if value in SPECIES_PRESETS:
        return SPECIES_PRESETS[value]

    parts = [part.strip() for part in value.split("|")]
    if len(parts) not in (2, 3):
        raise ValueError(
            "--species must be a preset key or 'Common Name|Scientific name'"
        )

    common_name, scientific_name = parts[0], parts[1]
    key = parts[2] if len(parts) == 3 else slugify(common_name)
    if not common_name or not scientific_name:
        raise ValueError("--species common and scientific names must be non-empty")

    return SpeciesSpec(
        key=key,
        common_name=common_name,
        scientific_name=scientific_name,
        prompt_label=f"{common_name} bird call, Bowra dry woodland",
        preferred_diel=("dawn", "morning", "afternoon", "dusk", "evening", "night"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict 2-3 species Layer C smoke-test manifest."
    )
    parser.add_argument("--items-csv", type=Path, default=ITEMS_CSV)
    parser.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--rejected-output", type=Path, default=REJECTED_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--species",
        action="append",
        default=None,
        help=(
            "Species preset or custom spec. Presets: "
            f"{', '.join(SPECIES_PRESETS)}. Custom: 'Common Name|Scientific name'. "
            "Repeat for 2-3 species."
        ),
    )
    parser.add_argument("--segments-per-species", type=int, default=50)
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--buffer-seconds", type=float, default=DEFAULT_BUFFER_SECONDS)
    parser.add_argument(
        "--max-per-recording",
        type=int,
        default=1,
        help="Preferred cap during the first selection pass.",
    )
    parser.add_argument(
        "--require-full-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject events whose buffered window would hit recording boundaries.",
    )
    parser.add_argument(
        "--avoid-clip-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject events whose buffered window crosses a 300 s S3 source clip boundary.",
    )
    parser.add_argument(
        "--strict-diel",
        action="store_true",
        help="Reject events outside the species preferred diel bins instead of just ranking them lower.",
    )
    parser.add_argument(
        "--exclude-recording-id",
        action="append",
        default=None,
        help="Recording ID to exclude, useful when a source clip is missing from S3. Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_species(args: argparse.Namespace) -> list[SpeciesSpec]:
    values = args.species or list(DEFAULT_SPECIES_ORDER)
    species = [parse_species(value) for value in values]
    seen: set[str] = set()
    deduped: list[SpeciesSpec] = []
    for spec in species:
        if spec.key in seen:
            continue
        seen.add(spec.key)
        deduped.append(spec)
    return deduped


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


def diel_bin_from_utc(dt: datetime | None) -> str:
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


def row_matches_species(row: dict[str, str], species: SpeciesSpec) -> bool:
    label_text = "|".join(
        [
            row.get("common_name_tags", ""),
            row.get("species_name_tags", ""),
            row.get("other_tags", ""),
        ]
    ).lower()
    return (
        species.common_name.lower() in label_text
        or species.scientific_name.lower() in label_text
    )


def annotation_csvs(annotations_dir: Path) -> list[Path]:
    paths = {
        *annotations_dir.glob("site_257_item_*/site_257_item_*.csv"),
        *annotations_dir.glob("annotations_*.csv"),
    }
    return sorted(paths)


def reject_row(
    row: dict[str, str],
    species: SpeciesSpec,
    annotation_csv: Path,
    reason: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    payload = {
        "species": species.key,
        "species_common_name": species.common_name,
        "audio_recording_id": (row.get("audio_recording_id") or "").strip(),
        "audio_event_id": (row.get("audio_event_id") or "").strip(),
        "score": row.get("score", ""),
        "event_start_seconds": row.get("event_start_seconds", ""),
        "event_end_seconds": row.get("event_end_seconds", ""),
        "annotation_source": row.get("audio_event_import_file_name", ""),
        "annotation_csv": str(annotation_csv.relative_to(REPO_ROOT)),
        "reject_reason": reason,
    }
    if extra:
        payload.update(extra)
    return payload


def iter_candidates(
    args: argparse.Namespace,
    species_specs: list[SpeciesSpec],
) -> tuple[list[Candidate], list[dict[str, str]], dict[str, int]]:
    items = load_items(args.items_csv)
    excluded_recording_ids = set(args.exclude_recording_id or [])
    candidates: list[Candidate] = []
    rejected: list[dict[str, str]] = []
    counters = {
        "annotation_csvs": 0,
        "matched_species_rows": 0,
        "empty_annotation_csvs": 0,
    }

    for annotation_csv in annotation_csvs(args.annotations_dir):
        counters["annotation_csvs"] += 1
        if annotation_csv.stat().st_size == 0:
            counters["empty_annotation_csvs"] += 1
            continue

        with annotation_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                matched_species = next(
                    (
                        species
                        for species in species_specs
                        if row_matches_species(row, species)
                    ),
                    None,
                )
                if matched_species is None:
                    continue

                counters["matched_species_rows"] += 1
                if (row.get("audio_event_import_file_name") or "") != "BirdNET.results.csv":
                    rejected.append(
                        reject_row(row, matched_species, annotation_csv, "non_birdnet_source")
                    )
                    continue

                recording_id = (row.get("audio_recording_id") or "").strip()
                if recording_id in excluded_recording_ids:
                    rejected.append(
                        reject_row(row, matched_species, annotation_csv, "excluded_recording_id")
                    )
                    continue
                item = items.get(recording_id)
                if item is None:
                    rejected.append(
                        reject_row(row, matched_species, annotation_csv, "missing_recording_item")
                    )
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
                    rejected.append(
                        reject_row(row, matched_species, annotation_csv, "invalid_numeric_field")
                    )
                    continue
                if event_end <= event_start:
                    rejected.append(
                        reject_row(row, matched_species, annotation_csv, "invalid_event_window")
                    )
                    continue
                if score < args.min_score:
                    rejected.append(
                        reject_row(row, matched_species, annotation_csv, "low_score")
                    )
                    continue

                event_duration = event_end - event_start
                if event_duration < args.min_duration:
                    rejected.append(
                        reject_row(row, matched_species, annotation_csv, "too_short")
                    )
                    continue
                if event_duration > args.max_duration:
                    rejected.append(
                        reject_row(row, matched_species, annotation_csv, "too_long")
                    )
                    continue
                if args.require_full_buffer and (
                    event_start - args.buffer_seconds < 0
                    or event_end + args.buffer_seconds > recording_duration
                ):
                    rejected.append(
                        reject_row(row, matched_species, annotation_csv, "buffer_hits_recording_edge")
                    )
                    continue
                if args.avoid_clip_boundary:
                    extracted_start = event_start - args.buffer_seconds
                    extracted_end = event_end + args.buffer_seconds
                    start_clip = int(extracted_start // 300.0)
                    end_clip = int((extracted_end - 1e-6) // 300.0)
                    if start_clip != end_clip:
                        rejected.append(
                            reject_row(
                                row,
                                matched_species,
                                annotation_csv,
                                "crosses_s3_clip_boundary",
                            )
                        )
                        continue

                utc_text = row.get("event_start_datetime_utc_00_00", "")
                utc_dt = parse_utc_datetime(utc_text)
                aest_dt = utc_dt.astimezone(AEST) if utc_dt else None
                diel_bin = diel_bin_from_utc(utc_dt)
                if args.strict_diel and diel_bin not in matched_species.preferred_diel:
                    rejected.append(
                        reject_row(
                            row,
                            matched_species,
                            annotation_csv,
                            "outside_preferred_diel",
                            {"diel_bin": diel_bin},
                        )
                    )
                    continue

                candidates.append(
                    Candidate(
                        species=matched_species,
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
                        diel_bin=diel_bin,
                        common_name_tags=row.get("common_name_tags", ""),
                        species_name_tags=row.get("species_name_tags", ""),
                        other_tags=row.get("other_tags", ""),
                        import_file_name=row.get("audio_event_import_file_name", ""),
                        annotation_csv=annotation_csv,
                    )
                )

    return candidates, rejected, counters


def select_for_species(
    candidates: list[Candidate],
    species: SpeciesSpec,
    segments_per_species: int,
    max_per_recording: int,
) -> list[Candidate]:
    preferred_rank = {name: index for index, name in enumerate(species.preferred_diel)}
    species_candidates = [candidate for candidate in candidates if candidate.species == species]
    ranked = sorted(
        species_candidates,
        key=lambda candidate: (
            preferred_rank.get(candidate.diel_bin, len(preferred_rank)),
            -candidate.score,
            candidate.item_count,
            candidate.audio_event_id,
        ),
    )

    selected: list[Candidate] = []
    per_recording_count: dict[str, int] = {}
    for candidate in ranked:
        count = per_recording_count.get(candidate.recording_id, 0)
        if count >= max_per_recording:
            continue
        selected.append(candidate)
        per_recording_count[candidate.recording_id] = count + 1
        if len(selected) >= segments_per_species:
            return selected

    for candidate in ranked:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= segments_per_species:
            break

    return selected


def manifest_row(
    candidate: Candidate,
    row_index: int,
    species_index: int,
    buffer_seconds: float,
    output_path: Path,
) -> dict[str, str]:
    extracted_start = max(0.0, candidate.event_start_seconds - buffer_seconds)
    extracted_end = min(
        candidate.recording_duration_seconds,
        candidate.event_end_seconds + buffer_seconds,
    )
    segment_base = output_path.parent / "segments"
    segment_path = (
        segment_base
        / f"site_257_item_{candidate.recording_id}"
        / f"site_257_item_{candidate.recording_id}_audioevent_{candidate.audio_event_id}"
        / f"site_257_item_{candidate.recording_id}_audioevent_{candidate.audio_event_id}.webm"
    )
    caption = (
        f"{candidate.species.prompt_label}, {candidate.diel_bin}, "
        f"BirdNET score {candidate.score:.4f}"
    )

    return {
        "smoke_event_index": str(row_index),
        "species_index": str(species_index),
        "event_type": candidate.species.key,
        "species_common_name": candidate.species.common_name,
        "species_scientific_name": candidate.species.scientific_name,
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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    args: argparse.Namespace,
    species_specs: list[SpeciesSpec],
    candidates: list[Candidate],
    selected_rows: list[dict[str, str]],
    rejected_rows: list[dict[str, str]],
    counters: dict[str, int],
) -> None:
    lines = [
        "# Layer C 2-3 Species Smoke Filter Report",
        "",
        "## Policy",
        "",
        f"- Species: {', '.join(spec.common_name for spec in species_specs)}",
        f"- Segments per species: `{args.segments_per_species}`",
        f"- BirdNET score: `>= {args.min_score}`",
        f"- Event duration: `{args.min_duration}-{args.max_duration}` seconds",
        f"- Buffer: `+/-{args.buffer_seconds}` seconds",
        f"- Require full buffer: `{args.require_full_buffer}`",
        f"- Avoid S3 300 s clip boundary: `{args.avoid_clip_boundary}`",
        f"- Strict diel: `{args.strict_diel}`",
        f"- Max per recording first pass: `{args.max_per_recording}`",
        f"- Excluded recordings: `{', '.join(args.exclude_recording_id or []) or 'none'}`",
        "",
        "## Summary",
        "",
        f"- Annotation CSVs scanned: `{counters['annotation_csvs']}`",
        f"- Empty annotation CSVs skipped: `{counters['empty_annotation_csvs']}`",
        f"- Rows matching target species: `{counters['matched_species_rows']}`",
        f"- Candidates after hard filters: `{len(candidates)}`",
        f"- Selected rows: `{len(selected_rows)}`",
        f"- Rejected target-species rows: `{len(rejected_rows)}`",
        "",
        "## Per Species",
        "",
        "| Species | Candidates | Selected | Preferred diel |",
        "|---|---:|---:|---|",
    ]
    for spec in species_specs:
        candidate_count = sum(1 for c in candidates if c.species == spec)
        selected_count = sum(1 for r in selected_rows if r["event_type"] == spec.key)
        lines.append(
            f"| {spec.common_name} | {candidate_count} | {selected_count} | "
            f"{', '.join(spec.preferred_diel)} |"
        )

    reject_counts: dict[str, int] = {}
    for row in rejected_rows:
        reason = row["reject_reason"]
        reject_counts[reason] = reject_counts.get(reason, 0) + 1

    lines.extend(["", "## Rejection Counts", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in sorted(reject_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "Download exact selected event segments:",
            "",
            "```bash",
            "python3 script/download/download_site_257_event_segments.py \\",
            f"  --event-manifest {args.output} \\",
            f"  --output-dir {args.output.parent / 'segments'} \\",
            f"  --min-score {args.min_score} \\",
            f"  --min-duration {args.min_duration} \\",
            f"  --max-duration {args.max_duration} \\",
            "  --workers 2",
            "```",
            "",
            "Prepare WAV/caption/spectrogram artifacts after download:",
            "",
            "```bash",
            "./acoustic_ai/.venv/bin/python script/dataset/prepare_layer_c_smoke_segments.py \\",
            f"  --dataset-dir {args.output.parent}",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.segments_per_species < 1:
        raise ValueError("--segments-per-species must be >= 1")
    if args.max_per_recording < 1:
        raise ValueError("--max-per-recording must be >= 1")
    if args.min_duration <= 0 or args.max_duration < args.min_duration:
        raise ValueError("duration bounds are invalid")
    if not args.items_csv.exists():
        raise FileNotFoundError(f"items CSV not found: {args.items_csv}")
    if not args.annotations_dir.exists():
        raise FileNotFoundError(f"annotations dir not found: {args.annotations_dir}")

    species_specs = resolve_species(args)
    if len(species_specs) < 2 or len(species_specs) > 3:
        print(
            "[warn] Current short smoke plan expects 2-3 species; "
            f"received {len(species_specs)}."
        )

    candidates, rejected_rows, counters = iter_candidates(args, species_specs)
    selected: list[Candidate] = []
    for spec in species_specs:
        selected.extend(
            select_for_species(
                candidates,
                spec,
                args.segments_per_species,
                args.max_per_recording,
            )
        )

    per_species_index: dict[str, int] = {}
    selected_rows: list[dict[str, str]] = []
    for row_index, candidate in enumerate(selected, start=1):
        per_species_index[candidate.species.key] = (
            per_species_index.get(candidate.species.key, 0) + 1
        )
        selected_rows.append(
            manifest_row(
                candidate,
                row_index,
                per_species_index[candidate.species.key],
                args.buffer_seconds,
                args.output,
            )
        )

    print(
        f"Scanned {counters['annotation_csvs']} annotation CSVs; "
        f"target rows={counters['matched_species_rows']}; "
        f"candidates={len(candidates)}; selected={len(selected_rows)}; "
        f"rejected={len(rejected_rows)}"
    )
    for spec in species_specs:
        candidate_count = sum(1 for c in candidates if c.species == spec)
        selected_count = sum(1 for r in selected_rows if r["event_type"] == spec.key)
        print(f"- {spec.key}: candidates={candidate_count} selected={selected_count}")

    if args.dry_run:
        print("Dry run - no files written.")
        return 0

    write_csv(args.output, selected_rows)
    write_csv(args.rejected_output, rejected_rows)
    write_report(
        args.report_output,
        args,
        species_specs,
        candidates,
        selected_rows,
        rejected_rows,
        counters,
    )
    print(f"Written: {args.output}")
    print(f"Written: {args.rejected_output}")
    print(f"Written: {args.report_output}")

    if not selected_rows:
        print(
            "[warn] No selected rows. If the full annotation CSVs are not present, "
            "run `dvc pull` or ask a teammate for the annotation index first."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
