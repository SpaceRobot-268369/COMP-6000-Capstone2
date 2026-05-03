#!/usr/bin/env python3
"""Build Layer C generic biological activity manifests from A2O annotations.

Layer C uses BirdNET-derived pseudo-labels and event timing for retrieval-based
event insertion. This script maps annotation events to downloaded 300-second
clips and writes small CSV indexes; it does not extract audio snippets.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE_DIR = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a"
DEFAULT_ANNOTATION_DIR = RESOURCE_DIR / "downloaded_annotations"
DEFAULT_TRAINING_MANIFEST = RESOURCE_DIR / "site_257_training_manifest.csv"
DEFAULT_EVENT_INDEX = RESOURCE_DIR / "annotation_event_index.csv"
DEFAULT_SNIPPET_MANIFEST = RESOURCE_DIR / "event_snippet_manifest.csv"
DEFAULT_ACTIVITY_MANIFEST = RESOURCE_DIR / "activity_label_manifest.csv"


EVENT_FIELDS = [
    "event_id",
    "recording_id",
    "clip_path",
    "clip_index",
    "clip_start_seconds",
    "clip_end_seconds",
    "event_start_seconds",
    "event_end_seconds",
    "event_duration_seconds",
    "event_start_in_clip_seconds",
    "event_end_in_clip_seconds",
    "event_midpoint_seconds",
    "event_midpoint_in_clip_seconds",
    "score",
    "confidence_band",
    "label",
    "label_kind",
    "common_name_tags",
    "species_name_tags",
    "other_tags",
    "source",
    "verification_consensus",
    "snippet_status",
    "sample_bin",
    "sample_local_date",
    "month",
    "month_range",
    "day_of_year",
    "hour_local",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "precipitation_mm",
    "precipitation_daily_mm",
    "days_since_rain",
]

ACTIVITY_FIELDS = [
    "clip_path",
    "recording_id",
    "clip_index",
    "sample_bin",
    "sample_local_date",
    "month",
    "month_range",
    "day_of_year",
    "hour_local",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "precipitation_mm",
    "precipitation_daily_mm",
    "days_since_rain",
    "has_activity",
    "event_count",
    "high_confidence_event_count",
    "activity_score_max",
    "activity_score_mean",
    "activity_density_per_minute",
    "top_labels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Layer C event and activity manifests.")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--training-manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST)
    parser.add_argument("--event-index", type=Path, default=DEFAULT_EVENT_INDEX)
    parser.add_argument("--snippet-manifest", type=Path, default=DEFAULT_SNIPPET_MANIFEST)
    parser.add_argument("--activity-manifest", type=Path, default=DEFAULT_ACTIVITY_MANIFEST)
    parser.add_argument("--min-score", type=float, default=0.5)
    return parser.parse_args()


def to_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def split_tags(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    parts = re.split(r"\s*[|;,]\s*", value)
    return [part.strip() for part in parts if part.strip()]


def clean_tag(tag: str) -> str:
    text = tag.strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    if text.endswith(":general"):
        text = text[: -len(":general")]
    return text.strip()


def best_label(row: dict) -> tuple[str, str]:
    common = [clean_tag(tag) for tag in split_tags(row.get("common_name_tags", ""))]
    species = [clean_tag(tag) for tag in split_tags(row.get("species_name_tags", ""))]
    other = [clean_tag(tag) for tag in split_tags(row.get("other_tags", ""))]

    if common:
        return common[0], "birdnet_common_name"
    if species:
        return species[0], "birdnet_species_name"
    if other:
        return other[0], "birdnet_other_tag"
    return "biological_activity", "generic_activity"


def confidence_band(score: float) -> str:
    if score >= 0.9:
        return "very_high"
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    if score > 0:
        return "low"
    return "unknown"


def load_clip_manifest(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_recording: dict[str, list[dict]] = {}
    for row in rows:
        by_recording.setdefault(str(row.get("recording_id", "")), []).append(row)
    for clips in by_recording.values():
        clips.sort(key=lambda item: to_float(item.get("clip_start_seconds")))
    return rows, by_recording


def find_clip_for_event(clips: list[dict], midpoint: float) -> dict | None:
    for clip in clips:
        start = to_float(clip.get("clip_start_seconds"))
        end = to_float(clip.get("clip_end_seconds"))
        if start <= midpoint < end:
            return clip
    return None


def read_annotation_rows(annotation_dir: Path) -> list[dict]:
    out = []
    for path in sorted(annotation_dir.glob("annotations_*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            out.extend(csv.DictReader(f))
    return out


def base_env_from_clip(clip: dict) -> dict:
    return {
        "sample_bin": clip.get("sample_bin", ""),
        "sample_local_date": clip.get("sample_local_date", ""),
        "month": clip.get("month", ""),
        "month_range": clip.get("month_range", ""),
        "day_of_year": clip.get("day_of_year", ""),
        "hour_local": clip.get("hour_local", ""),
        "temperature_c": clip.get("temperature_c", ""),
        "humidity_pct": clip.get("humidity_pct", ""),
        "wind_speed_ms": clip.get("wind_speed_ms", ""),
        "wind_direction_deg": clip.get("wind_direction_deg", ""),
        "precipitation_mm": clip.get("precipitation_mm", ""),
        "precipitation_daily_mm": clip.get("precipitation_daily_mm", ""),
        "days_since_rain": clip.get("days_since_rain", ""),
    }


def main() -> None:
    args = parse_args()
    if not args.annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {args.annotation_dir}")
    if not args.training_manifest.exists():
        raise FileNotFoundError(f"Training manifest not found: {args.training_manifest}")

    clip_rows, clips_by_recording = load_clip_manifest(args.training_manifest)
    annotation_rows = read_annotation_rows(args.annotation_dir)
    events = []
    skipped_low_score = 0
    skipped_no_clip = 0

    for row in annotation_rows:
        score = to_float(row.get("score"))
        if score < args.min_score:
            skipped_low_score += 1
            continue

        recording_id = str(row.get("audio_recording_id", "")).strip()
        clips = clips_by_recording.get(recording_id, [])
        start = to_float(row.get("event_start_seconds"))
        end = to_float(row.get("event_end_seconds"))
        duration = to_float(row.get("event_duration_seconds"), end - start)
        midpoint = start + max(0.0, duration) / 2.0
        clip = find_clip_for_event(clips, midpoint)
        if not clip:
            skipped_no_clip += 1
            continue

        clip_start = to_float(clip.get("clip_start_seconds"))
        clip_end = to_float(clip.get("clip_end_seconds"))
        start_in_clip = max(0.0, start - clip_start)
        end_in_clip = min(to_float(clip.get("clip_duration_seconds")), end - clip_start)
        snippet_status = "extractable" if end_in_clip > start_in_clip else "invalid_timing"
        label, label_kind = best_label(row)
        source = (row.get("audio_event_import_file_name") or row.get("audio_event_import_name") or "").strip()

        out = {
            "event_id": row.get("audio_event_id", ""),
            "recording_id": recording_id,
            "clip_path": clip.get("clip_path", ""),
            "clip_index": clip.get("clip_index", ""),
            "clip_start_seconds": clip.get("clip_start_seconds", ""),
            "clip_end_seconds": clip.get("clip_end_seconds", ""),
            "event_start_seconds": round(start, 3),
            "event_end_seconds": round(end, 3),
            "event_duration_seconds": round(duration, 3),
            "event_start_in_clip_seconds": round(start_in_clip, 3),
            "event_end_in_clip_seconds": round(end_in_clip, 3),
            "event_midpoint_seconds": round(midpoint, 3),
            "event_midpoint_in_clip_seconds": round(midpoint - clip_start, 3),
            "score": round(score, 4),
            "confidence_band": confidence_band(score),
            "label": label,
            "label_kind": label_kind,
            "common_name_tags": row.get("common_name_tags", ""),
            "species_name_tags": row.get("species_name_tags", ""),
            "other_tags": row.get("other_tags", ""),
            "source": source,
            "verification_consensus": row.get("verification_consensus", ""),
            "snippet_status": snippet_status,
        }
        out.update(base_env_from_clip(clip))
        events.append(out)

    activity_by_clip: dict[str, list[dict]] = {}
    for event in events:
        if event["snippet_status"] == "extractable":
            activity_by_clip.setdefault(event["clip_path"], []).append(event)

    activity_rows = []
    for clip in clip_rows:
        clip_path = clip.get("clip_path", "")
        clip_events = activity_by_clip.get(clip_path, [])
        scores = [to_float(event.get("score")) for event in clip_events]
        labels: dict[str, int] = {}
        for event in clip_events:
            labels[event["label"]] = labels.get(event["label"], 0) + 1
        top_labels = "; ".join(
            f"{label}:{count}" for label, count in sorted(labels.items(), key=lambda item: item[1], reverse=True)[:5]
        )
        duration_min = max(to_float(clip.get("clip_duration_seconds"), 300.0) / 60.0, 1e-6)
        row = {
            "clip_path": clip_path,
            "recording_id": clip.get("recording_id", ""),
            "clip_index": clip.get("clip_index", ""),
            **base_env_from_clip(clip),
            "has_activity": bool(clip_events),
            "event_count": len(clip_events),
            "high_confidence_event_count": sum(1 for score in scores if score >= 0.75),
            "activity_score_max": round(max(scores), 4) if scores else 0.0,
            "activity_score_mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "activity_density_per_minute": round(len(clip_events) / duration_min, 4),
            "top_labels": top_labels,
        }
        activity_rows.append(row)

    for path, rows, fields in (
        (args.event_index, events, EVENT_FIELDS),
        (args.snippet_manifest, [row for row in events if row["snippet_status"] == "extractable"], EVENT_FIELDS),
        (args.activity_manifest, activity_rows, ACTIVITY_FIELDS),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Annotation rows: {len(annotation_rows)}")
    print(f"Events written: {len(events)}")
    print(f"Activity clip rows: {len(activity_rows)}")
    print(f"Skipped low score: {skipped_low_score}")
    print(f"Skipped no clip: {skipped_no_clip}")
    print(f"Written -> {args.event_index}")
    print(f"Written -> {args.snippet_manifest}")
    print(f"Written -> {args.activity_manifest}")


if __name__ == "__main__":
    main()
