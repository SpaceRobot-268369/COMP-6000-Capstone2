"""Layer C generic biological activity planner for generation mode.

Layer C is retrieval-first: it uses BirdNET-derived pseudo-labels and event
timing from local annotation manifests to prepare biological activity snippets.
It does not train a species classifier or generate species calls from scratch.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE_DIR = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a"
DEFAULT_EVENT_MANIFEST = RESOURCE_DIR / "event_snippet_manifest.csv"
DEFAULT_ACTIVITY_MANIFEST = RESOURCE_DIR / "activity_label_manifest.csv"


@dataclass(frozen=True)
class EventCandidate:
    row: dict
    score: float
    confidence_score: float
    env_score: float
    context_score: float
    diversity_score: float


def _to_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_event_assets(manifest_path: Path = DEFAULT_EVENT_MANIFEST) -> list[dict]:
    rows = _load_csv(manifest_path)
    return [
        row for row in rows
        if row.get("snippet_status") == "extractable"
        and row.get("clip_path")
        and (PROJECT_ROOT / row["clip_path"]).exists()
    ]


def load_activity_assets(manifest_path: Path = DEFAULT_ACTIVITY_MANIFEST) -> list[dict]:
    return _load_csv(manifest_path)


def _month_context_score(env: dict, row: dict) -> float:
    score = 0.0
    if str(env.get("sample_bin", "")).lower() == str(row.get("sample_bin", "")).lower():
        score += 0.4

    q_month = _to_float(env.get("month"), 0)
    r_month = _to_float(row.get("month"), 0)
    if q_month and r_month:
        diff = abs(q_month - r_month) % 12
        score += 0.35 * (1.0 - min(diff, 12 - diff) / 6.0)

    if str(env.get("month_range", "")).lower() == str(row.get("month_range", "")).lower():
        score += 0.15

    q_hour = _to_float(env.get("hour_local"), -1)
    r_hour = _to_float(row.get("hour_local"), -1)
    if q_hour >= 0 and r_hour >= 0:
        diff = abs(q_hour - r_hour) % 24
        score += 0.10 * (1.0 - min(diff, 24 - diff) / 12.0)

    return _clamp(score)


def _env_score(env: dict, row: dict) -> float:
    checks = [
        ("temperature_c", 18.0),
        ("humidity_pct", 60.0),
        ("wind_speed_ms", 8.0),
        ("precipitation_mm", 6.0),
        ("days_since_rain", 45.0),
    ]
    parts = []
    for key, scale in checks:
        if key not in env:
            continue
        query = _to_float(env.get(key))
        actual = _to_float(row.get(key))
        parts.append(1.0 - _clamp(abs(query - actual) / scale))
    if not parts:
        return 0.0
    return _clamp(sum(parts) / len(parts))


def _activity_probability(env: dict, row: dict) -> tuple[float, float, float]:
    context_score = _month_context_score(env, row)
    env_score = _env_score(env, row)
    activity_strength = _clamp(_to_float(row.get("activity_density_per_minute")) / 0.8)
    probability = _clamp(0.45 * context_score + 0.35 * env_score + 0.20 * activity_strength)
    return probability, context_score, env_score


def _candidate_activity_rows(env: dict, activity_rows: list[dict]) -> list[dict]:
    query_bin = str(env.get("sample_bin", "")).lower()
    query_month = int(_to_float(env.get("month"), 0))

    month_bin = [
        row for row in activity_rows
        if str(row.get("sample_bin", "")).lower() == query_bin
        and int(_to_float(row.get("month"), 0)) == query_month
    ]
    if len(month_bin) >= 12:
        return month_bin

    same_bin = [
        row for row in activity_rows
        if str(row.get("sample_bin", "")).lower() == query_bin
    ]
    if len(same_bin) >= 12:
        return same_bin

    return activity_rows


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in values) / total_weight


def _activity_level(env: dict, activity_rows: list[dict]) -> tuple[str, int, dict]:
    if not activity_rows:
        return "unknown", 0, {
            "estimated_event_density_per_minute": 0.0,
            "estimated_event_count": 0,
            "activity_probability": 0.0,
            "matching_clip_count": 0,
        }

    matching = _candidate_activity_rows(env, activity_rows)
    scored = []
    for row in matching:
        probability, context_score, env_score = _activity_probability(env, row)
        density = _to_float(row.get("activity_density_per_minute"))
        event_count = _to_float(row.get("event_count"))
        has_activity = 1.0 if str(row.get("has_activity", "")).lower() == "true" else 0.0
        weight = 0.15 + 0.55 * context_score + 0.30 * env_score
        scored.append({
            "row": row,
            "weight": weight,
            "probability": probability,
            "context_score": context_score,
            "env_score": env_score,
            "density": density,
            "event_count": event_count,
            "has_activity": has_activity,
        })

    if not scored:
        return "none", 0, {
            "estimated_event_density_per_minute": 0.0,
            "estimated_event_count": 0,
            "activity_probability": 0.0,
            "matching_clip_count": 0,
        }

    density = _weighted_mean([(item["density"], item["weight"]) for item in scored])
    observed_count = _weighted_mean([(item["event_count"], item["weight"]) for item in scored])
    activity_probability = _weighted_mean([(item["has_activity"], item["weight"]) for item in scored])
    similarity = _weighted_mean([
        ((item["context_score"] + item["env_score"]) / 2.0, item["weight"])
        for item in scored
    ])

    estimated_count = round(0.55 * observed_count + 0.45 * density * 5.0)
    if activity_probability < 0.04 and density < 0.03:
        target_count = 0
    elif estimated_count <= 0:
        target_count = 1
    else:
        target_count = int(max(1, min(4, estimated_count)))

    if target_count <= 0:
        level = "none"
    elif target_count == 1:
        level = "sparse"
    elif target_count == 2:
        level = "moderate"
    else:
        level = "active"

    model = {
        "estimated_event_density_per_minute": round(density, 4),
        "estimated_event_count": target_count,
        "raw_estimated_event_count": round(estimated_count, 3),
        "activity_probability": round(activity_probability, 4),
        "mean_similarity": round(similarity, 4),
        "matching_clip_count": len(matching),
        "method": "weighted_manifest_statistics",
        "features": [
            "month",
            "month_range",
            "sample_bin",
            "hour_local",
            "temperature_c",
            "humidity_pct",
            "wind_speed_ms",
            "precipitation_mm",
            "days_since_rain",
        ],
    }
    return level, target_count, model


def _select_events(events: list[dict], env: dict, target_count: int, seed: Optional[int]) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    label_counts: dict[str, int] = {}

    for row in events:
        confidence_score = _clamp(_to_float(row.get("score")))
        if confidence_score < 0.5:
            continue
        context_score = _month_context_score(env, row)
        env_score = _env_score(env, row)
        label = row.get("label") or "biological_activity"
        label_penalty = min(label_counts.get(label, 0) * 0.12, 0.36)
        diversity_score = 1.0 - label_penalty
        total = (
            0.35 * confidence_score
            + 0.30 * context_score
            + 0.20 * env_score
            + 0.15 * diversity_score
        )
        candidates.append(EventCandidate(row, total, confidence_score, env_score, context_score, diversity_score))
        label_counts[label] = label_counts.get(label, 0) + 1

    if not candidates or target_count <= 0:
        return []

    candidates.sort(key=lambda item: item.score, reverse=True)
    shortlist = candidates[: min(max(target_count * 6, 12), len(candidates))]
    rng = random.Random(seed)
    selected: list[EventCandidate] = []
    used_labels: set[str] = set()
    for candidate in shortlist:
        if len(selected) >= target_count:
            break
        label = candidate.row.get("label") or "biological_activity"
        if label in used_labels and len(used_labels) < min(target_count, 3):
            continue
        selected.append(candidate)
        used_labels.add(label)

    while len(selected) < target_count and shortlist:
        candidate = rng.choice(shortlist)
        if candidate not in selected:
            selected.append(candidate)

    return selected


def _scheduled_start(index: int, count: int, duration_seconds: float, seed: Optional[int]) -> float:
    rng = random.Random(None if seed is None else seed + index * 101)
    if count <= 1:
        base = duration_seconds * 0.5
    else:
        base = duration_seconds * (index + 1) / (count + 1)
    jitter = rng.uniform(-duration_seconds * 0.08, duration_seconds * 0.08)
    return round(_clamp(base + jitter, 2.0, max(2.0, duration_seconds - 3.0)), 2)


def _event_metadata(candidate: EventCandidate, index: int, count: int,
                    duration_seconds: float, seed: Optional[int]) -> dict:
    row = candidate.row
    return {
        "enabled": True,
        "label": row.get("label") or "biological_activity",
        "label_kind": row.get("label_kind") or "generic_activity",
        "source": row.get("source") or "BirdNET-derived annotation",
        "confidence": round(_clamp(candidate.score), 3),
        "birdnet_score": round(candidate.confidence_score, 3),
        "env_score": round(candidate.env_score, 3),
        "context_score": round(candidate.context_score, 3),
        "gain_db": -9.0 if candidate.confidence_score >= 0.75 else -12.0,
        "selected": {
            "event_id": row.get("event_id"),
            "clip_path": row.get("clip_path"),
            "recording_id": row.get("recording_id"),
            "clip_index": int(_to_float(row.get("clip_index"), 0)),
            "event_start_seconds": _to_float(row.get("event_start_seconds")),
            "event_end_seconds": _to_float(row.get("event_end_seconds")),
            "event_start_in_clip_seconds": _to_float(row.get("event_start_in_clip_seconds")),
            "event_end_in_clip_seconds": _to_float(row.get("event_end_in_clip_seconds")),
            "month": int(_to_float(row.get("month"), 0)),
            "month_range": row.get("month_range"),
            "sample_bin": row.get("sample_bin"),
        },
        "schedule": {
            "start_seconds": _scheduled_start(index, count, duration_seconds, seed),
            "duration_seconds": min(_to_float(row.get("event_duration_seconds"), 3.0), 12.0),
        },
    }


def prepare_event_layers(env: dict, seed: Optional[int] = None,
                         event_manifest_path: Path = DEFAULT_EVENT_MANIFEST,
                         activity_manifest_path: Path = DEFAULT_ACTIVITY_MANIFEST,
                         output_duration_seconds: float = 30.0) -> dict:
    """Return Layer C event layer plan and retrieval metadata."""
    events = load_event_assets(event_manifest_path)
    activity_rows = load_activity_assets(activity_manifest_path)
    activity_level, target_count, probability_model = _activity_level(env, activity_rows)

    if not events:
        return {
            "status": "unavailable",
            "strategy": "generic_biological_activity",
            "event_manifest": str(event_manifest_path.relative_to(PROJECT_ROOT)),
            "activity_manifest": str(activity_manifest_path.relative_to(PROJECT_ROOT)),
            "activity_level": activity_level,
            "probability_model": probability_model,
            "events": [],
            "mix_hints": {"prepared_only": True},
            "explanation": "Layer C found no local event snippet manifest. Run the annotation preprocessing step first.",
        }

    if target_count <= 0:
        return {
            "status": "no_activity_needed",
            "strategy": "generic_biological_activity",
            "event_manifest": str(event_manifest_path.relative_to(PROJECT_ROOT)),
            "activity_manifest": str(activity_manifest_path.relative_to(PROJECT_ROOT)),
            "activity_level": activity_level,
            "probability_model": probability_model,
            "events": [],
            "mix_hints": {"prepared_only": True},
            "explanation": "Layer C did not schedule events because matching local activity density is low.",
        }

    selected = _select_events(events, env, target_count, seed)
    event_meta = [
        _event_metadata(candidate, index, len(selected), output_duration_seconds, seed)
        for index, candidate in enumerate(selected)
    ]

    status = "prepared" if event_meta else "no_matching_events"
    explanation = (
        "Layer C scheduled BirdNET-derived annotation snippets as generic "
        "biological activity using confidence, month, diel bin, and local "
        "environmental similarity. Final mixing is deferred to Layer D."
        if event_meta else
        "Layer C found activity in the annotation index, but no suitable event snippets matched the request."
    )

    return {
        "status": status,
        "strategy": "birdnet_pseudo_label_activity_retrieval",
        "event_manifest": str(event_manifest_path.relative_to(PROJECT_ROOT)),
        "activity_manifest": str(activity_manifest_path.relative_to(PROJECT_ROOT)),
        "activity_level": activity_level,
        "probability_model": probability_model,
        "events": event_meta,
        "mix_hints": {
            "prepared_only": True,
            "event_count": len(event_meta),
            "avoid_species_claims": True,
        },
        "explanation": explanation,
    }
