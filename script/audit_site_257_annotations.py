#!/usr/bin/env python3
"""Audit downloaded Site 257 A2O annotation CSV files for Layer C planning.

This script only reports annotation coverage, schema, labels, sources, and
basic quality signals. It does not build event indexes or snippet manifests.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE_DIR = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a"
DEFAULT_ANNOTATION_DIR = RESOURCE_DIR / "downloaded_annotations"
DEFAULT_OUTPUT = RESOURCE_DIR / "annotation_audit_report.md"

EXPECTED_FIELDS = [
    "audio_event_id",
    "audio_recording_id",
    "audio_recording_uuid",
    "audio_recording_start_datetime_utc_00_00",
    "site_id",
    "site_name",
    "event_start_seconds",
    "event_end_seconds",
    "event_duration_seconds",
    "low_frequency_hertz",
    "high_frequency_hertz",
    "score",
    "is_reference",
    "common_name_tags",
    "species_name_tags",
    "other_tags",
    "verifications",
    "verification_counts",
    "verification_correct",
    "verification_incorrect",
    "verification_skip",
    "verification_unsure",
    "verification_decisions",
    "verification_consensus",
    "audio_event_import_file_name",
    "audio_event_import_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit downloaded Site 257 annotation CSVs.")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def recording_id_from_path(path: Path) -> str:
    match = re.search(r"annotations_(\d+)\.csv$", path.name)
    return match.group(1) if match else path.stem


def split_tags(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    parts = re.split(r"\s*[|;,]\s*", value)
    return [part.strip() for part in parts if part.strip()]


def has_value(row: dict, key: str) -> bool:
    return bool(str(row.get(key, "")).strip())


def read_annotation_file(path: Path) -> tuple[list[str], list[dict], str | None]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        return fieldnames, rows, None
    except Exception as exc:
        return [], [], f"{exc.__class__.__name__}: {exc}"


def pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "0.0%"
    return f"{part / whole * 100:.1f}%"


def md_counter_table(counter: Counter, headers: tuple[str, str], limit: int = 20) -> list[str]:
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---:|"]
    if not counter:
        lines.append("| none | 0 |")
        return lines
    for key, count in counter.most_common(limit):
        safe_key = str(key).replace("|", "\\|") or "blank"
        lines.append(f"| {safe_key} | {count} |")
    return lines


def main() -> None:
    args = parse_args()
    if not args.annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {args.annotation_dir}")

    files = sorted(args.annotation_dir.glob("annotations_*.csv"))
    schema_counts: Counter[tuple[str, ...]] = Counter()
    missing_expected: Counter[str] = Counter()
    row_presence: Counter[str] = Counter()
    common_names: Counter[str] = Counter()
    species_names: Counter[str] = Counter()
    other_tags: Counter[str] = Counter()
    import_files: Counter[str] = Counter()
    import_names: Counter[str] = Counter()
    verification_consensus: Counter[str] = Counter()
    score_bands: Counter[str] = Counter()
    duration_bands: Counter[str] = Counter()
    frequency_status: Counter[str] = Counter()
    parse_errors: dict[str, str] = {}
    rows_by_recording: dict[str, int] = {}
    labelled_recordings: set[str] = set()
    species_recordings: set[str] = set()
    common_recordings: set[str] = set()
    activity_recordings: set[str] = set()
    source_recordings: defaultdict[str, set[str]] = defaultdict(set)

    total_rows = 0
    files_with_rows = 0
    score_only_rows = 0
    rows_with_score = 0
    rows_with_common = 0
    rows_with_species = 0
    rows_with_other = 0
    rows_with_any_label = 0
    rows_with_verification = 0
    rows_with_import = 0
    rows_with_timing = 0
    rows_with_frequency = 0
    usable_timing_rows = 0

    for path in files:
        recording_id = recording_id_from_path(path)
        fieldnames, rows, error = read_annotation_file(path)
        if error:
            parse_errors[str(path.relative_to(PROJECT_ROOT))] = error
            continue

        schema_counts[tuple(fieldnames)] += 1
        missing = [field for field in EXPECTED_FIELDS if field not in fieldnames]
        for field in missing:
            missing_expected[field] += 1

        row_count = len(rows)
        rows_by_recording[recording_id] = row_count
        if row_count:
            files_with_rows += 1
        total_rows += row_count

        for row in rows:
            common = split_tags(row.get("common_name_tags", ""))
            species = split_tags(row.get("species_name_tags", ""))
            other = split_tags(row.get("other_tags", ""))
            has_common = bool(common)
            has_species = bool(species)
            has_other = bool(other)
            has_score = has_value(row, "score")
            has_import = has_value(row, "audio_event_import_file_name") or has_value(row, "audio_event_import_name")
            has_verification = any(
                has_value(row, key)
                for key in (
                    "verifications",
                    "verification_counts",
                    "verification_correct",
                    "verification_incorrect",
                    "verification_skip",
                    "verification_unsure",
                    "verification_decisions",
                    "verification_consensus",
                )
            )
            has_timing = has_value(row, "event_start_seconds") and has_value(row, "event_end_seconds")
            has_frequency = has_value(row, "low_frequency_hertz") and has_value(row, "high_frequency_hertz")

            rows_with_score += int(has_score)
            rows_with_common += int(has_common)
            rows_with_species += int(has_species)
            rows_with_other += int(has_other)
            rows_with_any_label += int(has_common or has_species or has_other)
            rows_with_verification += int(has_verification)
            rows_with_import += int(has_import)
            rows_with_timing += int(has_timing)
            rows_with_frequency += int(has_frequency)
            score_only_rows += int(has_score and not (has_common or has_species or has_other))

            if has_timing:
                try:
                    start = float(row.get("event_start_seconds", ""))
                    end = float(row.get("event_end_seconds", ""))
                    duration = float(row.get("event_duration_seconds") or (end - start))
                    usable_timing_rows += int(start >= 0 and end > start)
                    if duration < 0.25:
                        duration_bands["<0.25s"] += 1
                    elif duration < 1.0:
                        duration_bands["0.25-1s"] += 1
                    elif duration < 5.0:
                        duration_bands["1-5s"] += 1
                    elif duration < 30.0:
                        duration_bands["5-30s"] += 1
                    else:
                        duration_bands[">=30s"] += 1
                except ValueError:
                    duration_bands["invalid"] += 1

            if has_frequency:
                frequency_status["present"] += 1
            else:
                frequency_status["missing"] += 1

            if has_score:
                try:
                    score = float(row.get("score", ""))
                    if score < 0.25:
                        score_bands["<0.25"] += 1
                    elif score < 0.5:
                        score_bands["0.25-0.5"] += 1
                    elif score < 0.75:
                        score_bands["0.5-0.75"] += 1
                    elif score < 0.9:
                        score_bands["0.75-0.9"] += 1
                    else:
                        score_bands[">=0.9"] += 1
                except ValueError:
                    score_bands["invalid"] += 1

            for tag in common:
                common_names[tag] += 1
            for tag in species:
                species_names[tag] += 1
            for tag in other:
                other_tags[tag] += 1

            if has_common or has_species or has_other:
                labelled_recordings.add(recording_id)
            if has_species:
                species_recordings.add(recording_id)
            if has_common:
                common_recordings.add(recording_id)
            if has_score:
                activity_recordings.add(recording_id)

            import_file = (row.get("audio_event_import_file_name") or "").strip()
            import_name = (row.get("audio_event_import_name") or "").strip()
            if import_file:
                import_files[import_file] += 1
                source_recordings[import_file].add(recording_id)
            if import_name:
                import_names[import_name] += 1

            consensus = (row.get("verification_consensus") or "").strip() or "blank"
            verification_consensus[consensus] += 1

            for key in row:
                if has_value(row, key):
                    row_presence[key] += 1

    empty_files = len(files) - files_with_rows - len(parse_errors)
    lines: list[str] = []
    lines.append("# Site 257 Annotation Audit")
    lines.append("")
    lines.append("Scope: downloaded A2O annotation CSVs only. No event index, snippet manifest, classifier, or Layer C runtime was built in this step.")
    lines.append("")
    lines.append("## Coverage Summary")
    lines.append("")
    lines.append(f"- Annotation CSV files found: {len(files)}")
    lines.append(f"- Files with at least one event row: {files_with_rows}")
    lines.append(f"- Empty annotation files: {empty_files}")
    lines.append(f"- Parse errors: {len(parse_errors)}")
    lines.append(f"- Total event rows: {total_rows}")
    lines.append(f"- Recordings with any label tags: {len(labelled_recordings)}")
    lines.append(f"- Recordings with common-name tags: {len(common_recordings)}")
    lines.append(f"- Recordings with species-name tags: {len(species_recordings)}")
    lines.append(f"- Recordings with score/activity rows: {len(activity_recordings)}")
    lines.append("")
    lines.append("## Row-Level Signals")
    lines.append("")
    lines.append(f"- Rows with usable start/end timing: {usable_timing_rows} / {total_rows} ({pct(usable_timing_rows, total_rows)})")
    lines.append(f"- Rows with score: {rows_with_score} / {total_rows} ({pct(rows_with_score, total_rows)})")
    lines.append(f"- Rows with common-name tags: {rows_with_common} / {total_rows} ({pct(rows_with_common, total_rows)})")
    lines.append(f"- Rows with species-name tags: {rows_with_species} / {total_rows} ({pct(rows_with_species, total_rows)})")
    lines.append(f"- Rows with other tags: {rows_with_other} / {total_rows} ({pct(rows_with_other, total_rows)})")
    lines.append(f"- Rows with any label tags: {rows_with_any_label} / {total_rows} ({pct(rows_with_any_label, total_rows)})")
    lines.append(f"- Score-only rows without common/species/other labels: {score_only_rows} / {total_rows} ({pct(score_only_rows, total_rows)})")
    lines.append(f"- Rows with verification fields populated: {rows_with_verification} / {total_rows} ({pct(rows_with_verification, total_rows)})")
    lines.append(f"- Rows with import/source fields populated: {rows_with_import} / {total_rows} ({pct(rows_with_import, total_rows)})")
    lines.append(f"- Rows with frequency bounds: {rows_with_frequency} / {total_rows} ({pct(rows_with_frequency, total_rows)})")
    lines.append("")
    lines.append("## Schema")
    lines.append("")
    lines.append(f"- Unique schemas found: {len(schema_counts)}")
    lines.append(f"- Dominant schema file count: {schema_counts.most_common(1)[0][1] if schema_counts else 0}")
    lines.append(f"- Dominant schema columns: {len(schema_counts.most_common(1)[0][0]) if schema_counts else 0}")
    lines.append("")
    if missing_expected:
        lines.append("### Missing Expected Fields")
        lines.append("")
        lines.extend(md_counter_table(missing_expected, ("Field", "File count"), limit=50))
        lines.append("")
    else:
        lines.append("All expected audit fields are present in every parsed CSV schema.")
        lines.append("")

    lines.append("## Top Common Name Tags")
    lines.append("")
    lines.extend(md_counter_table(common_names, ("Common name", "Rows"), limit=30))
    lines.append("")
    lines.append("## Top Species Name Tags")
    lines.append("")
    lines.extend(md_counter_table(species_names, ("Species name", "Rows"), limit=30))
    lines.append("")
    lines.append("## Top Other Tags")
    lines.append("")
    lines.extend(md_counter_table(other_tags, ("Other tag", "Rows"), limit=30))
    lines.append("")
    lines.append("## Import Sources")
    lines.append("")
    lines.extend(md_counter_table(import_files, ("Import file", "Rows"), limit=30))
    lines.append("")
    if import_names:
        lines.append("### Import Names")
        lines.append("")
        lines.extend(md_counter_table(import_names, ("Import name", "Rows"), limit=30))
        lines.append("")

    lines.append("## Score Bands")
    lines.append("")
    lines.extend(md_counter_table(score_bands, ("Score band", "Rows"), limit=20))
    lines.append("")
    lines.append("## Duration Bands")
    lines.append("")
    lines.extend(md_counter_table(duration_bands, ("Duration band", "Rows"), limit=20))
    lines.append("")
    lines.append("## Verification Consensus")
    lines.append("")
    lines.extend(md_counter_table(verification_consensus, ("Consensus", "Rows"), limit=20))
    lines.append("")
    lines.append("## Event Rows By Recording")
    lines.append("")
    recording_counter = Counter(rows_by_recording)
    lines.extend(md_counter_table(recording_counter, ("Recording ID", "Event rows"), limit=30))
    lines.append("")
    lines.append("## Audit Interpretation")
    lines.append("")
    if rows_with_species and len(species_recordings) >= 10:
        lines.append("- Species-name labels exist across multiple recordings, but class balance should be checked before training a supervised species classifier.")
    elif rows_with_species:
        lines.append("- Species-name labels exist, but recording coverage is sparse; retrieval or pseudo-label assistance is safer than supervised multiclass training for MVP.")
    else:
        lines.append("- No species-name labels were found; Layer C should not make species-specific claims from these annotations alone.")
    if score_only_rows > rows_with_any_label:
        lines.append("- Score-only activity rows dominate labelled rows, so a weak biological/acoustic activity layer is likely useful.")
    if import_files:
        lines.append("- Import/source metadata is available and should be retained in later indexes for reliability filtering.")
    if usable_timing_rows == total_rows and total_rows:
        lines.append("- Event timing fields are consistently usable, so the next preprocessing step can map events to 300-second clips.")
    lines.append("- Next step should be event-index construction only after accepting this audit result.")
    lines.append("")

    if parse_errors:
        lines.append("## Parse Errors")
        lines.append("")
        for path, error in parse_errors.items():
            lines.append(f"- `{path}`: {error}")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written audit report -> {args.output}")
    print(f"files={len(files)} files_with_rows={files_with_rows} empty={empty_files} rows={total_rows}")


if __name__ == "__main__":
    main()
