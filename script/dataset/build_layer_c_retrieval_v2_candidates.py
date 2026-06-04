#!/usr/bin/env python3
"""Build Layer C retrieval v2 candidate and preselected sample manifests.

This is the metadata-only pass for the v2 retrieval route. It scans the
site_257 per-recording annotation CSVs, keeps events matching the first 63
species in the inventory, marks samples that were previously human-approved,
and selects up to the v2 per-species target quota.

No audio is downloaded or cut here. The output feeds the later review-package
builder that will create full crops, target-band crops, mel images, metadata,
and review files.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_ROOT = REPO_ROOT / "resources" / "site_257_bowra-dry-a"
DEFAULT_QUOTA = SITE_ROOT / "layer_c_retrieval_event_library_v2" / "species_quota_v2.csv"
DEFAULT_ANNOTATIONS = SITE_ROOT / "all_items_annotation"
DEFAULT_ITEMS = SITE_ROOT / "site_257_filtered_items.csv"
DEFAULT_ALL_ITEMS = SITE_ROOT / "site_257_all_items.csv"
DEFAULT_OUT = SITE_ROOT / "layer_c_retrieval_event_library_v2"
PRIOR_LIBRARIES = (
    SITE_ROOT
    / "layer_c_retrieval_top8_species_candidates_v1"
    / "final_retrieval_library_top35_human_v1"
    / "layer_c_retrieval_top35_library.csv",
    SITE_ROOT
    / "layer_c_retrieval_event_library_split_v1"
    / "final_pass_library_v1"
    / "layer_c_retrieval_final_pass_library.csv",
)


def slugify(value: str) -> str:
    value = value.lower().replace("'", "")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def season_from_date(value: str) -> str:
    if not value:
        return "unknown"
    try:
        month = date.fromisoformat(value[:10]).month
    except ValueError:
        return "unknown"
    if month in (9, 10, 11):
        return "spring"
    if month in (12, 1, 2):
        return "summer"
    if month in (3, 4, 5):
        return "autumn"
    return "winter"


def diel_from_hour(hour: int) -> str:
    if 5 <= hour < 8:
        return "dawn"
    if 8 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "night"


def local_context_from_event(row: dict[str, str]) -> dict[str, str]:
    value = row.get("event_start_datetime_utc_00_00", "")
    if not value:
        return {"diel_bin": "unknown", "sample_local_date": "", "season": "unknown"}
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return {"diel_bin": "unknown", "sample_local_date": "", "season": "unknown"}
    # Bowra recordings in this dataset use AEST (+10) naming. This is a
    # metadata fallback only; audio cutting still uses event seconds.
    local_dt = dt + timedelta(hours=10)
    local_date = local_dt.date().isoformat()
    return {
        "diel_bin": diel_from_hour(local_dt.hour),
        "sample_local_date": local_date,
        "season": season_from_date(local_date),
    }


def tag_names(value: str) -> set[str]:
    """Extract names from A2O tag cells like `144:Spotted Nightjar:general`."""
    names: set[str] = set()
    for part in str(value or "").split("|"):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) >= 2:
            names.add(pieces[1].strip().lower())
        else:
            names.add(part.lower())
    return names


def row_tag_names(row: dict[str, str]) -> set[str]:
    names: set[str] = set()
    for col in ("common_name_tags", "species_name_tags", "other_tags"):
        names.update(tag_names(row.get(col, "")))
    return names


def load_quota(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = read_csv(path)
    by_name = {r["species_common_name"].lower(): r for r in rows}
    for row in rows:
        row["species_slug"] = slugify(row["species_common_name"])
    return rows, by_name


def load_excluded_event_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    rows = read_csv(path)
    if not rows:
        return set()
    key = "audio_event_id" if "audio_event_id" in rows[0] else next(iter(rows[0]))
    return {str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()}


def load_item_context(filtered_path: Path, all_items_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if all_items_path.exists():
        for row in read_csv(all_items_path):
            recording_id = str(row.get("id", "")).strip()
            if not recording_id:
                continue
            recorded_date = row.get("recorded_date", "")
            out[recording_id] = {
                "diel_bin": "unknown",
                "sample_local_date": "",
                "season": season_from_date(recorded_date),
                "recording_duration_s": row.get("duration_seconds", ""),
                "canonical_file_name": row.get("canonical_file_name", ""),
                "notes_relative_path": row.get("notes_relative_path", ""),
            }
    if not filtered_path.exists():
        return out
    for row in read_csv(filtered_path):
        recording_id = str(row.get("id", "")).strip()
        if not recording_id:
            continue
        sample_local_date = row.get("sample_local_date", "")
        out[recording_id] = {
            "diel_bin": row.get("sample_bin", "") or "unknown",
            "sample_local_date": sample_local_date,
            "season": season_from_date(sample_local_date),
            "recording_duration_s": row.get("duration_seconds", ""),
            "canonical_file_name": row.get("canonical_file_name", ""),
            "notes_relative_path": row.get("notes_relative_path", ""),
        }
    return out


def load_prior_pass(paths: tuple[Path, ...]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            verdict = str(row.get("verdict", "")).lower()
            if verdict and verdict != "pass":
                continue
            event_id = str(row.get("audio_event_id", "")).strip()
            if not event_id:
                continue
            existing = out.get(event_id)
            priority = 2 if "final_pass_library_v1" in str(path) else 1
            if existing and parse_int(existing.get("prior_priority"), 0) >= priority:
                continue
            out[event_id] = {
                "reused_from_prior_library": "true",
                "prior_library_path": str(path.relative_to(REPO_ROOT)),
                "prior_retrieval_audio_path": row.get("retrieval_audio_path", ""),
                "prior_retrieval_spectrogram_path": row.get("retrieval_spectrogram_path", ""),
                "prior_source_audio_path": row.get("source_audio_path", ""),
                "prior_notes": row.get("notes", ""),
                "prior_quality_score": row.get("quality_score", ""),
                "prior_priority": str(priority),
            }
    return out


def match_species(row: dict[str, str], quota_rows: list[dict[str, str]]) -> dict[str, str] | None:
    names = row_tag_names(row)
    for species in quota_rows:
        common = species["species_common_name"].lower()
        scientific = species["species_scientific_name"].lower()
        if common in names or scientific in names:
            return species
    return None


def source_annotation_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def candidate_row(
    row: dict[str, str],
    species: dict[str, str],
    item_context: dict[str, dict[str, str]],
    prior_pass: dict[str, dict[str, str]],
    annotation_csv: Path,
) -> dict[str, Any]:
    event_id = str(row.get("audio_event_id", "")).strip()
    recording_id = str(row.get("audio_recording_id", "")).strip()
    ctx = item_context.get(recording_id, {})
    event_start = parse_float(row.get("event_start_seconds"))
    event_end = parse_float(row.get("event_end_seconds"))
    duration = parse_float(row.get("event_duration_seconds"), event_end - event_start)
    low_hz = row.get("low_frequency_hertz", "")
    high_hz = row.get("high_frequency_hertz", "")
    score = parse_float(row.get("score"))
    prior = prior_pass.get(event_id, {})
    quality_score = prior.get("prior_quality_score") or f"{score:.6f}"
    event_local_ctx = local_context_from_event(row)

    reject_reasons: list[str] = []
    if not event_id:
        reject_reasons.append("missing_audio_event_id")
    if not recording_id:
        reject_reasons.append("missing_recording_id")
    if event_end <= event_start:
        reject_reasons.append("invalid_time_range")
    if duration <= 0:
        reject_reasons.append("invalid_duration")
    if duration < 0.3:
        reject_reasons.append("duration_below_0.3s")

    warning_reasons: list[str] = []
    if duration < 0.5:
        warning_reasons.append("shorter_than_recommended_0.5s")
    if duration > 8.0:
        warning_reasons.append("longer_than_recommended_8.0s")
    if not low_hz or not high_hz:
        warning_reasons.append("missing_annotation_frequency_box")
    if not ctx:
        warning_reasons.append("missing_item_context")

    status = "basic_reject" if reject_reasons else "candidate"
    return {
        "species_rank": species["rank"],
        "species_common_name": species["species_common_name"],
        "species_scientific_name": species["species_scientific_name"],
        "species_slug": species["species_slug"],
        "target_sample_count": species["target_sample_count"],
        "selection_bucket": species["selection_bucket"],
        "reuse_old_pass_allowed": species["reuse_old_pass_allowed"],
        "audio_event_id": event_id,
        "recording_id": recording_id,
        "audio_recording_uuid": row.get("audio_recording_uuid", ""),
        "score": f"{score:.6f}",
        "quality_score": quality_score,
        "event_start_s": f"{event_start:.4f}",
        "event_end_s": f"{event_end:.4f}",
        "duration_s": f"{duration:.4f}",
        "low_frequency_hertz": low_hz,
        "high_frequency_hertz": high_hz,
        "diel_bin": ctx.get("diel_bin") if ctx.get("diel_bin") != "unknown" else event_local_ctx["diel_bin"],
        "season": ctx.get("season") if ctx.get("season") != "unknown" else event_local_ctx["season"],
        "sample_local_date": ctx.get("sample_local_date") or event_local_ctx["sample_local_date"],
        "recording_duration_s": ctx.get("recording_duration_s", ""),
        "canonical_file_name": ctx.get("canonical_file_name", ""),
        "notes_relative_path": ctx.get("notes_relative_path", ""),
        "source_annotation_csv": source_annotation_path(annotation_csv),
        "listen_url": row.get("listen_url", ""),
        "library_url": row.get("library_url", ""),
        "candidate_status": status,
        "reject_reason": "; ".join(reject_reasons),
        "warning_reason": "; ".join(warning_reasons),
        "reused_from_prior_library": prior.get("reused_from_prior_library", "false"),
        "prior_library_path": prior.get("prior_library_path", ""),
        "prior_retrieval_audio_path": prior.get("prior_retrieval_audio_path", ""),
        "prior_retrieval_spectrogram_path": prior.get("prior_retrieval_spectrogram_path", ""),
        "prior_source_audio_path": prior.get("prior_source_audio_path", ""),
        "prior_notes": prior.get("prior_notes", ""),
    }


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    reused = 1 if row.get("reused_from_prior_library") == "true" else 0
    score = parse_float(row.get("score"))
    quality = parse_float(row.get("quality_score"), score)
    duration = parse_float(row.get("duration_s"))
    duration_ok = 1 if 0.5 <= duration <= 8.0 else 0
    has_freq = 1 if row.get("low_frequency_hertz") and row.get("high_frequency_hertz") else 0
    return (-reused, -duration_ok, -quality, -score, -has_freq, duration)


def select_rows(
    candidates: list[dict[str, Any]],
    max_per_recording: int = 10,
    excluded_event_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded_event_ids = excluded_event_ids or set()
    selected: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_species[str(row["species_common_name"])].append(row)

    for species_name in sorted(by_species):
        rows = [
            r
            for r in by_species[species_name]
            if r["candidate_status"] == "candidate" and str(r.get("audio_event_id", "")) not in excluded_event_ids
        ]
        rows.sort(key=candidate_sort_key)
        target = parse_int(rows[0]["target_sample_count"], 0) if rows else 0
        kept: list[dict[str, Any]] = []
        recording_counts: dict[str, int] = defaultdict(int)
        deferred: list[dict[str, Any]] = []
        for row in rows:
            recording_id = str(row.get("recording_id", ""))
            if recording_id and recording_counts[recording_id] >= max_per_recording:
                deferred.append(row)
                continue
            kept.append(row)
            if recording_id:
                recording_counts[recording_id] += 1
            if len(kept) >= target:
                break
        if len(kept) < target:
            for row in deferred:
                kept.append(row)
                if len(kept) >= target:
                    break
        for i, row in enumerate(kept, start=1):
            row["selection_rank"] = i
            row["selected_for_v2_review"] = "true"
            row["recording_cap_applied"] = str(max_per_recording)
            selected.append(row)
        prior_count = sum(1 for r in rows if r.get("reused_from_prior_library") == "true")
        selected_prior_count = sum(1 for r in kept if r.get("reused_from_prior_library") == "true")
        summary.append(
            {
                "species_common_name": species_name,
                "species_slug": rows[0]["species_slug"] if rows else "",
                "target_sample_count": target,
                "candidate_count": len(rows),
                "selected_count": len(kept),
                "prior_pass_candidate_count": prior_count,
                "prior_pass_selected_count": selected_prior_count,
                "unique_recordings_selected": len({str(r.get("recording_id", "")) for r in kept}),
                "recording_cap_applied": max_per_recording,
                "shortfall_after_local_scan": max(0, target - len(kept)),
                "needs_s3_backfill": "true" if len(kept) < target else "false",
            }
        )
    return selected, summary


FIELDNAMES = [
    "species_rank",
    "species_common_name",
    "species_scientific_name",
    "species_slug",
    "target_sample_count",
    "selection_bucket",
    "reuse_old_pass_allowed",
    "selection_rank",
    "selected_for_v2_review",
    "recording_cap_applied",
    "audio_event_id",
    "recording_id",
    "audio_recording_uuid",
    "score",
    "quality_score",
    "event_start_s",
    "event_end_s",
    "duration_s",
    "low_frequency_hertz",
    "high_frequency_hertz",
    "diel_bin",
    "season",
    "sample_local_date",
    "recording_duration_s",
    "canonical_file_name",
    "notes_relative_path",
    "source_annotation_csv",
    "listen_url",
    "library_url",
    "candidate_status",
    "reject_reason",
    "warning_reason",
    "reused_from_prior_library",
    "prior_library_path",
    "prior_retrieval_audio_path",
    "prior_retrieval_spectrogram_path",
    "prior_source_audio_path",
    "prior_notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quota-csv", type=Path, default=DEFAULT_QUOTA)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--filtered-items-csv", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--all-items-csv", type=Path, default=DEFAULT_ALL_ITEMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exclude-event-ids", type=Path, default=None)
    parser.add_argument("--selected-output-name", default="selected_samples_v2.csv")
    parser.add_argument("--summary-output-name", default="selection_summary_v2.csv")
    args = parser.parse_args()

    quota_rows, _ = load_quota(args.quota_csv)
    item_context = load_item_context(args.filtered_items_csv, args.all_items_csv)
    prior_pass = load_prior_pass(PRIOR_LIBRARIES)
    excluded_event_ids = load_excluded_event_ids(args.exclude_event_ids)

    candidates: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    annotation_paths = sorted(args.annotation_dir.glob("site_257_item_*/site_257_item_*.csv"))
    for annotation_csv in annotation_paths:
        for row in read_csv(annotation_csv):
            species = match_species(row, quota_rows)
            if species is None:
                continue
            out = candidate_row(row, species, item_context, prior_pass, annotation_csv)
            out["selection_rank"] = ""
            out["selected_for_v2_review"] = "false"
            if out["candidate_status"] == "basic_reject":
                rejects.append(out)
            else:
                candidates.append(out)

    selected, summary = select_rows(candidates, excluded_event_ids=excluded_event_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_samples_v2.csv", candidates, FIELDNAMES)
    write_csv(args.output_dir / args.selected_output_name, selected, FIELDNAMES)
    write_csv(args.output_dir / "rejected_samples_v2.csv", rejects, FIELDNAMES)
    write_csv(
        args.output_dir / args.summary_output_name,
        summary,
        [
            "species_common_name",
            "species_slug",
            "target_sample_count",
            "candidate_count",
            "selected_count",
            "prior_pass_candidate_count",
            "prior_pass_selected_count",
            "unique_recordings_selected",
            "recording_cap_applied",
            "shortfall_after_local_scan",
            "needs_s3_backfill",
        ],
    )

    print(f"annotation_csvs={len(annotation_paths)}")
    print(f"candidate_rows={len(candidates)}")
    print(f"selected_rows={len(selected)}")
    print(f"basic_reject_rows={len(rejects)}")
    shortfalls = [r for r in summary if r["needs_s3_backfill"] == "true"]
    print(f"species_needing_s3_backfill={len(shortfalls)}")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
