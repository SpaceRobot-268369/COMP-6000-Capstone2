"""Layer C generic biological activity planner for generation mode.

Layer C is retrieval-first: it uses BirdNET-derived pseudo-labels and event
timing from local annotation manifests to prepare biological activity snippets.
It does not train a species classifier or generate species calls from scratch.
"""

from __future__ import annotations

import csv
import hashlib
import json
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
    type_score: float


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


def _event_type(row: dict) -> tuple[str, dict]:
    label_text = " ".join([
        str(row.get("label", "")),
        str(row.get("label_kind", "")),
        str(row.get("common_name_tags", "")),
        str(row.get("species_name_tags", "")),
        str(row.get("other_tags", "")),
    ]).lower()
    duration = _to_float(row.get("event_duration_seconds"))

    insect_terms = ("insect", "cicada", "cricket", "katydid", "grasshopper")
    if any(term in label_text for term in insect_terms):
        return "insect_like", {
            "method": "label_duration_heuristic",
            "signals": ["insect_label_term"],
        }

    if str(row.get("label_kind", "")).startswith("birdnet_") and row.get("label"):
        signals = ["birdnet_label"]
        if duration and duration <= 8.0:
            signals.append("short_annotated_event")
        return "bird_like", {
            "method": "birdnet_label_heuristic",
            "signals": signals,
        }

    return "unknown_activity", {
        "method": "fallback_activity_type",
        "signals": ["generic_or_score_only_annotation"],
    }


def _type_preference_score(env: dict, row: dict) -> float:
    event_type, _ = _event_type(row)
    sample_bin = str(env.get("sample_bin") or row.get("sample_bin") or "").lower()
    temperature = _to_float(env.get("temperature_c"), _to_float(row.get("temperature_c"), 20.0))

    if sample_bin in {"dawn", "morning"}:
        preference = {
            "bird_like": 1.0,
            "insect_like": 0.65,
            "unknown_activity": 0.75,
        }
    elif sample_bin == "night":
        preference = {
            "bird_like": 0.72,
            "insect_like": 0.95,
            "unknown_activity": 0.85,
        }
    else:
        preference = {
            "bird_like": 0.85,
            "insect_like": 0.8,
            "unknown_activity": 0.8,
        }

    score = preference.get(event_type, 0.75)
    if event_type == "insect_like" and temperature >= 24.0:
        score += 0.1
    return _clamp(score)


def _select_events(events: list[dict], env: dict, target_count: int, seed: Optional[int]) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    label_counts: dict[str, int] = {}

    for row in events:
        confidence_score = _clamp(_to_float(row.get("score")))
        if confidence_score < 0.5:
            continue
        context_score = _month_context_score(env, row)
        env_score = _env_score(env, row)
        type_score = _type_preference_score(env, row)
        label = row.get("label") or "biological_activity"
        label_penalty = min(label_counts.get(label, 0) * 0.12, 0.36)
        diversity_score = 1.0 - label_penalty
        total = (
            0.32 * confidence_score
            + 0.27 * context_score
            + 0.18 * env_score
            + 0.13 * diversity_score
            + 0.10 * type_score
        )
        candidates.append(EventCandidate(
            row,
            total,
            confidence_score,
            env_score,
            context_score,
            diversity_score,
            type_score,
        ))
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


def _duration_for_event(candidate: EventCandidate) -> float:
    return min(_to_float(candidate.row.get("event_duration_seconds"), 3.0), 12.0)


def _minimum_spacing(count: int, duration_seconds: float) -> float:
    if count <= 1:
        return 0.0
    if duration_seconds < 20:
        return 2.5
    return 4.0 if count <= 3 else 3.0


def _poisson_like_starts(count: int, duration_seconds: float, seed: Optional[int]) -> list[float]:
    if count <= 0:
        return []

    rng = random.Random(seed)
    earliest = 2.0
    latest = max(earliest, duration_seconds - 3.0)
    min_spacing = _minimum_spacing(count, duration_seconds)

    if count == 1:
        start = rng.uniform(duration_seconds * 0.25, duration_seconds * 0.75)
        return [round(_clamp(start, earliest, latest), 2)]

    usable = max(latest - earliest, min_spacing * (count - 1))
    best: list[float] = []
    best_score = -1.0
    for _ in range(80):
        intervals = [rng.expovariate(1.0) for _ in range(count + 1)]
        total = sum(intervals) or 1.0
        cursor = earliest
        starts = []
        for gap in intervals[1:]:
            cursor += (gap / total) * usable
            starts.append(cursor)
            if len(starts) == count:
                break

        starts.sort()
        adjusted = []
        for start in starts:
            if adjusted:
                start = max(start, adjusted[-1] + min_spacing)
            adjusted.append(start)

        overflow = adjusted[-1] - latest
        if overflow > 0:
            adjusted = [start - overflow for start in adjusted]
        adjusted = [_clamp(start, earliest, latest) for start in adjusted]
        gaps = [b - a for a, b in zip(adjusted, adjusted[1:])]
        if gaps and min(gaps) < min_spacing - 0.05:
            continue

        irregularity = (max(gaps) - min(gaps)) if gaps else 0.0
        edge_margin = min(adjusted[0] - earliest, latest - adjusted[-1])
        score = irregularity + 0.15 * edge_margin
        if score > best_score:
            best = adjusted
            best_score = score

    if not best:
        span = latest - earliest
        best = [earliest + span * (index + 1) / (count + 1) for index in range(count)]

    return [round(start, 2) for start in best]


def _avoid_adjacent_label_repeats(candidates: list[EventCandidate]) -> list[EventCandidate]:
    remaining = list(candidates)
    ordered: list[EventCandidate] = []
    previous_label = ""

    while remaining:
        pick_index = 0
        for index, candidate in enumerate(remaining):
            label = candidate.row.get("label") or "biological_activity"
            if label != previous_label:
                pick_index = index
                break
        candidate = remaining.pop(pick_index)
        ordered.append(candidate)
        previous_label = candidate.row.get("label") or "biological_activity"

    return ordered


def _sample_bin_gain_adjustment(sample_bin: str) -> float:
    sample_bin = sample_bin.lower()
    if sample_bin == "night":
        return 1.5
    if sample_bin == "dawn":
        return 0.75
    if sample_bin == "morning":
        return 0.25
    return -0.5


def _context_gain(candidate: EventCandidate, env: dict, event_count: int) -> tuple[float, dict]:
    wind_speed = _to_float(env.get("wind_speed_ms"))
    wind_max = _to_float(env.get("wind_max_ms"), wind_speed)
    precipitation = max(
        _to_float(env.get("precipitation_mm")),
        _to_float(env.get("precipitation_daily_mm")) / 12.0,
    )
    humidity = _to_float(env.get("humidity_pct"), 50.0)
    sample_bin = str(env.get("sample_bin") or candidate.row.get("sample_bin") or "")

    base_gain = -10.5 if candidate.confidence_score >= 0.75 else -12.5
    confidence_adjustment = (candidate.confidence_score - 0.65) * 2.0
    wind_adjustment = -_clamp(max(wind_speed, wind_max * 0.75) / 10.0) * 4.0
    rain_adjustment = -_clamp(precipitation / 4.0) * 5.0
    humidity_adjustment = 0.5 if humidity >= 70 and precipitation <= 0.5 else 0.0
    bin_adjustment = _sample_bin_gain_adjustment(sample_bin)
    density_adjustment = -0.5 * max(event_count - 2, 0)

    gain_db = _clamp(
        base_gain
        + confidence_adjustment
        + wind_adjustment
        + rain_adjustment
        + humidity_adjustment
        + bin_adjustment
        + density_adjustment,
        -22.0,
        -7.0,
    )

    model = {
        "method": "context_aware_rule_mixing_hint",
        "base_gain_db": round(base_gain, 2),
        "confidence_adjustment_db": round(confidence_adjustment, 2),
        "wind_adjustment_db": round(wind_adjustment, 2),
        "rain_adjustment_db": round(rain_adjustment, 2),
        "humidity_adjustment_db": round(humidity_adjustment, 2),
        "sample_bin_adjustment_db": round(bin_adjustment, 2),
        "density_adjustment_db": round(density_adjustment, 2),
    }
    return round(gain_db, 2), model


def _stable_event_variation_seed(candidate: EventCandidate, env: dict, seed: Optional[int],
                                 scheduled_start: float, event_type: str) -> int:
    row = candidate.row
    payload = {
        "seed": seed,
        "event_id": row.get("event_id"),
        "clip_path": row.get("clip_path"),
        "recording_id": row.get("recording_id"),
        "clip_index": row.get("clip_index"),
        "scheduled_start": round(scheduled_start, 2),
        "event_type": event_type,
        "env": {
            key: env.get(key)
            for key in (
                "month",
                "month_range",
                "sample_bin",
                "hour_local",
                "temperature_c",
                "humidity_pct",
                "wind_speed_ms",
                "precipitation_mm",
                "days_since_rain",
            )
        },
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16)


def _event_transform_plan(candidate: EventCandidate, env: dict, seed: Optional[int],
                          scheduled_start: float, event_type: str,
                          duration: float) -> dict:
    row = candidate.row
    variation_seed = _stable_event_variation_seed(candidate, env, seed, scheduled_start, event_type)
    rng = random.Random(variation_seed)

    source_start = _to_float(row.get("event_start_in_clip_seconds"))
    source_end = _to_float(row.get("event_end_in_clip_seconds"), source_start + duration)
    source_duration = max(source_end - source_start, duration, 0.5)
    trim_room = max(0.0, min(0.4, source_duration * 0.08))
    source_offset = source_start + rng.uniform(0.0, trim_room)

    wind_speed = _to_float(env.get("wind_speed_ms"))
    precipitation = max(
        _to_float(env.get("precipitation_mm")),
        _to_float(env.get("precipitation_daily_mm")) / 12.0,
    )
    weather_pressure = _clamp(wind_speed / 10.0 + precipitation / 8.0)

    if event_type == "bird_like":
        pitch_span = 0.25
        stretch_lo, stretch_hi = 0.985, 1.015
        highpass_base = (900, 1400)
        lowpass_base = (7200, 9800)
    elif event_type == "insect_like":
        pitch_span = 0.15
        stretch_lo, stretch_hi = 0.98, 1.02
        highpass_base = (1800, 3200)
        lowpass_base = (8500, 11000)
    else:
        pitch_span = 0.2
        stretch_lo, stretch_hi = 0.98, 1.02
        highpass_base = (600, 1400)
        lowpass_base = (6500, 10000)

    target_duration = _clamp(duration * rng.uniform(0.96, 1.04), 0.5, 12.0)
    gain_variation = rng.uniform(-1.0, 1.0) - 0.6 * weather_pressure
    pan = rng.uniform(-0.18, 0.18)

    return {
        "source_offset_sec": round(source_offset, 2),
        "target_duration_sec": round(target_duration, 2),
        "gain_variation_db": round(_clamp(gain_variation, -2.0, 1.5), 2),
        "time_stretch": round(rng.uniform(stretch_lo, stretch_hi), 4),
        "pitch_shift_semitones": round(rng.uniform(-pitch_span, pitch_span), 3),
        "highpass_hz": int(round(rng.uniform(*highpass_base))),
        "lowpass_hz": int(round(rng.uniform(*lowpass_base))),
        "fade_in_sec": round(rng.uniform(0.04, 0.18), 3),
        "fade_out_sec": round(rng.uniform(0.08, 0.28), 3),
        "pan": round(pan, 3),
        "variation_seed": variation_seed,
        "planning_note": (
            "Subtle seed-controlled transform hints only; source event audio is not generated or processed here."
        ),
    }


def _event_metadata(candidate: EventCandidate, index: int, count: int,
                    scheduled_start: float, env: dict, seed: Optional[int]) -> dict:
    row = candidate.row
    duration = _duration_for_event(candidate)
    gain_db, gain_model = _context_gain(candidate, env, count)
    event_type, type_model = _event_type(row)
    transform = _event_transform_plan(candidate, env, seed, scheduled_start, event_type, duration)
    return {
        "enabled": True,
        "label": row.get("label") or "biological_activity",
        "label_kind": row.get("label_kind") or "generic_activity",
        "event_type": event_type,
        "source": row.get("source") or "BirdNET-derived annotation",
        "confidence": round(_clamp(candidate.score), 3),
        "birdnet_score": round(candidate.confidence_score, 3),
        "env_score": round(candidate.env_score, 3),
        "context_score": round(candidate.context_score, 3),
        "type_score": round(candidate.type_score, 3),
        "type_model": type_model,
        "gain_db": gain_db,
        "gain_model": gain_model,
        "transform": transform,
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
            "start_seconds": scheduled_start,
            "duration_seconds": duration,
            "spacing_policy": "poisson_like_min_gap",
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
    selected = _avoid_adjacent_label_repeats(selected)
    starts = _poisson_like_starts(len(selected), output_duration_seconds, seed)
    event_meta = [
        _event_metadata(candidate, index, len(selected), starts[index], env, seed)
        for index, candidate in enumerate(selected)
    ]
    event_type_counts: dict[str, int] = {}
    for event in event_meta:
        event_type = event.get("event_type") or "unknown_activity"
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

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
            "scheduling_policy": "poisson_like_spacing_with_minimum_gap",
            "minimum_spacing_seconds": _minimum_spacing(len(event_meta), output_duration_seconds),
            "gain_policy": "context_aware_environmental_attenuation",
            "event_type_policy": "lightweight_label_and_context_control",
            "event_type_counts": event_type_counts,
        },
        "explanation": explanation,
    }
