"""Adapters from Layer E head reports to the aggregator's internal v1 shape.

The three current heads were built independently, so their JSON contracts do
not line up exactly. This module only normalizes field names and extracts
evidence; it does not decide the final season/diel answer.
"""

from __future__ import annotations

from typing import Any


SEASONS = ("spring", "summer", "autumn", "winter")
DIELS = ("dawn", "morning", "afternoon", "night")


def adapt_head_reports(
    *,
    ambient_report: dict[str, Any] | None = None,
    weather_report: dict[str, Any] | None = None,
    events_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return normalized observations plus season/diel evidence.

    Accepts either raw handler reports or registry wrapper responses shaped as
    ``{"report": {...}, "attempt": {...}}``.
    """
    ambient = _unwrap_report(ambient_report)
    weather = _unwrap_report(weather_report)
    events = _unwrap_report(events_report)

    ambient_observation = adapt_ambient_observation(ambient)
    weather_observation = adapt_weather_observation(weather)
    event_observations = adapt_event_observations(events)

    return {
        "observations": {
            "ambient": ambient_observation,
            "weather": weather_observation,
            "events": event_observations,
        },
        "evidence": {
            "season": (
                ambient_context_evidence(ambient_observation, "season")
                + event_context_evidence(event_observations, "season")
            ),
            "diel": (
                ambient_context_evidence(ambient_observation, "diel")
                + event_context_evidence(event_observations, "diel")
            ),
        },
    }


def adapt_ambient_observation(report: dict[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    return {
        "similar_clips": list(report.get("similar_clips") or []),
        "estimated_conditions": report.get("estimated_conditions"),
        "confidence": _safe_float(report.get("confidence"), default=0.0),
        "season_confidence": _safe_float(report.get("season_confidence"), default=0.0),
        "ood_flag": bool(report.get("ood_flag", False)),
    }


def adapt_weather_observation(report: dict[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    weather = _get_nested(report, "observations", "weather")
    if isinstance(weather, dict):
        return _weather_with_defaults(weather)
    return _default_weather_observation()


def adapt_event_observations(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    report = report or {}
    events = report.get("events")
    if not isinstance(events, list):
        return []

    normalized: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        phenology = event.get("phenology") if isinstance(event.get("phenology"), dict) else {}
        normalized.append(
            {
                "label": str(event.get("label", "unknown")),
                "common_name": phenology.get("common_name") or _species_label(event.get("label")),
                "scientific_name": phenology.get("scientific_name"),
                "confidence": _safe_float(
                    event.get("confidence_mean", event.get("confidence")),
                    default=0.0,
                ),
                "confidence_max": _safe_float(event.get("confidence_max"), default=None),
                "onset_s": _safe_float(event.get("onset_s"), default=None),
                "offset_s": _safe_float(event.get("offset_s"), default=None),
                "window_count": event.get("window_count"),
                "phenology": {
                    "diel_signal": phenology.get("diel_signal"),
                    "diel_confidence": _safe_float(phenology.get("diel_confidence"), default=0.0),
                    "season_signal": phenology.get("season_signal"),
                    "season_confidence": _safe_float(phenology.get("season_confidence"), default=0.0),
                    "habitat_signal": phenology.get("habitat_signal"),
                    "inference_notes": phenology.get("inference_notes"),
                },
            }
        )
    return normalized


def ambient_context_evidence(observation: dict[str, Any], field: str) -> list[dict[str, Any]]:
    conditions = observation.get("estimated_conditions")
    if not isinstance(conditions, dict):
        return []

    if field == "season":
        value = _clean_token(conditions.get("season"))
        if value not in SEASONS:
            return []
        confidence = min(
            _safe_float(observation.get("confidence"), default=0.0),
            _safe_float(observation.get("season_confidence"), default=0.0),
        )
        return [
            {
                "source_head": "ambient",
                "field": "season",
                "value": value,
                "candidates": [value],
                "confidence": round(confidence, 6),
                "reason": f"Ambient head estimated {value}.",
            }
        ]

    if field == "diel":
        value = _clean_token(conditions.get("diel_bin"))
        if value not in DIELS:
            return []
        confidence = _safe_float(observation.get("confidence"), default=0.0)
        return [
            {
                "source_head": "ambient",
                "field": "diel",
                "value": value,
                "candidates": [value],
                "confidence": round(confidence, 6),
                "reason": f"Ambient head estimated {value}.",
            }
        ]

    return []


def event_context_evidence(events: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for event in events:
        phenology = event.get("phenology") if isinstance(event.get("phenology"), dict) else {}
        event_confidence = _safe_float(event.get("confidence"), default=0.0)
        common_name = event.get("common_name") or event.get("label") or "Unknown species"

        if field == "diel":
            signal = _clean_token(phenology.get("diel_signal"))
            candidates = _diel_candidates(signal)
            signal_confidence = _safe_float(phenology.get("diel_confidence"), default=0.0)
        elif field == "season":
            signal = _clean_token(phenology.get("season_signal"))
            candidates = _season_candidates(signal)
            signal_confidence = _safe_float(phenology.get("season_confidence"), default=0.0)
        else:
            continue

        if not candidates:
            continue

        evidence.append(
            {
                "source_head": "events",
                "field": field,
                "value": signal,
                "candidates": candidates,
                "confidence": round(event_confidence * signal_confidence, 6),
                "event_label": event.get("label"),
                "common_name": common_name,
                "onset_s": event.get("onset_s"),
                "offset_s": event.get("offset_s"),
                "reason": f"{common_name} has a {signal} {field} signal.",
            }
        )
    return evidence


def _weather_with_defaults(weather: dict[str, Any]) -> dict[str, Any]:
    normalized = _default_weather_observation()
    for element in ("wind", "rain", "thunder"):
        current = weather.get(element)
        if isinstance(current, dict):
            normalized[element].update(current)
            summary = current.get("summary")
            if isinstance(summary, dict):
                normalized[element]["summary"] = {
                    **normalized[element]["summary"],
                    **summary,
                }
    normalized["confidence"] = _safe_float(weather.get("confidence"), default=0.0)
    normalized["derived_label"] = str(weather.get("derived_label", "none"))
    normalized["warnings"] = list(weather.get("warnings") or [])
    return normalized


def _default_weather_observation() -> dict[str, Any]:
    def summary() -> dict[str, Any]:
        return {
            "intensity": 0.0,
            "variability": 0.0,
            "coverage": 0.0,
            "label": "none",
            "confidence": 0.0,
        }

    return {
        "wind": {"summary": summary()},
        "rain": {"summary": summary()},
        "thunder": {"summary": summary(), "events": [], "mean_interval_s": None},
        "confidence": 0.0,
        "derived_label": "none",
        "warnings": [],
    }


def _season_candidates(signal: str | None) -> list[str]:
    mapping = {
        "spring": ["spring"],
        "summer": ["summer"],
        "autumn": ["autumn"],
        "winter": ["winter"],
        "spring_summer": ["spring", "summer"],
        "warm_season": ["spring", "summer", "autumn"],
    }
    if not signal or signal in {"weak", "unknown", "year_round", "year-round"}:
        return []
    return mapping.get(signal, [signal] if signal in SEASONS else [])


def _diel_candidates(signal: str | None) -> list[str]:
    mapping = {
        "dawn": ["dawn"],
        "morning": ["morning"],
        "afternoon": ["afternoon"],
        "night": ["night"],
        "day": ["morning", "afternoon"],
        "diurnal": ["morning", "afternoon"],
        "day_dawn_dusk": ["dawn", "morning", "afternoon"],
        "crepuscular": ["dawn"],
        "day_night": list(DIELS),
    }
    if not signal or signal in {"weak", "unknown"}:
        return []
    return mapping.get(signal, [signal] if signal in DIELS else [])


def _unwrap_report(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    inner = report.get("report")
    return inner if isinstance(inner, dict) else report


def _get_nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _safe_float(value: Any, *, default: float | None = 0.0) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return default


def _clean_token(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower().replace("-", "_")


def _species_label(value: Any) -> str:
    if value is None:
        return "Unknown species"
    return " ".join(part.capitalize() for part in str(value).split("_") if part)
