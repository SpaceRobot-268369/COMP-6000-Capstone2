"""Deterministic Layer E analysis aggregator.

This module owns the first real fusion step for Analysis Mode. It trusts
weather and event observations as observations, then fuses only the latent
context fields: season and diel.
"""

from __future__ import annotations

from typing import Any

from .adapters import DIELS, SEASONS, adapt_head_reports


SCHEMA_VERSION = "analysis_aggregator.v1"

DEFAULT_PARAMS: dict[str, Any] = {
    "fusion": {
        "ambient_weight_cap": 0.5,
        "event_weight_cap": 1.0,
        "ambient_fallback_min_confidence": 0.6,
        "undetermined_threshold": 0.50,
        "conflict_margin": 0.20,
    },
    "defaults": {
        "season_distribution": {
            "spring": 0.25,
            "summer": 0.25,
            "autumn": 0.25,
            "winter": 0.25,
        },
        "diel_distribution": {
            "dawn": 0.25,
            "morning": 0.25,
            "afternoon": 0.25,
            "night": 0.25,
        },
    },
    "limitations": [
        "Season is difficult to infer from audio alone at this site.",
        "Ambient context is a weak prior, not ground truth.",
        "The species detector only covers the known species in its checkpoint.",
    ],
}


def aggregate_reports(
    *,
    ambient_report: dict[str, Any] | None = None,
    weather_report: dict[str, Any] | None = None,
    events_report: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate the three Layer E head reports into the v1 fused report."""
    cfg = _merge_params(params)
    adapted = adapt_head_reports(
        ambient_report=ambient_report,
        weather_report=weather_report,
        events_report=events_report,
    )

    season = fuse_context_field(
        "season",
        adapted["evidence"]["season"],
        bins=SEASONS,
        default_distribution=cfg["defaults"]["season_distribution"],
        cfg=cfg,
    )
    diel = fuse_context_field(
        "diel",
        adapted["evidence"]["diel"],
        bins=DIELS,
        default_distribution=cfg["defaults"]["diel_distribution"],
        cfg=cfg,
    )
    disagreements = detect_disagreements(
        season=season,
        diel=diel,
        observations=adapted["observations"],
        cfg=cfg,
    )

    limitations = list(cfg.get("limitations") or [])
    if season["estimate"] == "undetermined" or diel["estimate"] == "undetermined":
        limitations.append("One or more context fields were under-determined by the available evidence.")
    if adapted["observations"]["ambient"].get("ood_flag"):
        limitations.append("The ambient head flagged the clip as possibly outside the Bowra training distribution.")

    confidence_values = [
        float(season.get("posterior", 0.0)),
        float(diel.get("posterior", 0.0)),
        float(adapted["observations"]["weather"].get("confidence", 0.0)),
    ]
    confidence = round(sum(confidence_values) / len(confidence_values), 6)
    limitations = _dedupe(limitations)
    decision = build_decision_json(
        observations=adapted["observations"],
        diel=diel,
        season=season,
        disagreements=disagreements,
        confidence=confidence,
        limitations=limitations,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "analysis",
        "observations": adapted["observations"],
        "inferred_context": {
            "diel": diel,
            "season": season,
        },
        "decision": decision,
        "narration": build_narration(decision),
        "llm_input": build_llm_input(decision),
        "disagreements": disagreements,
        "overall_confidence": confidence,
        "limitations": limitations,
    }


def build_decision_json(
    *,
    observations: dict[str, Any],
    diel: dict[str, Any],
    season: dict[str, Any],
    disagreements: list[dict[str, Any]],
    confidence: float,
    limitations: list[str],
) -> dict[str, Any]:
    """Build the compact machine-readable decision passed to a future LLM."""
    weather = observations.get("weather") if isinstance(observations.get("weather"), dict) else {}
    events = observations.get("events") if isinstance(observations.get("events"), list) else []

    return {
        "schema_version": "analysis_decision.v1",
        "time_of_day": {
            "value": diel.get("estimate", "undetermined"),
            "confidence": float(diel.get("posterior", 0.0)),
            "distribution": dict(diel.get("distribution") or {}),
            "evidence": diel.get("primary_evidence"),
        },
        "season": {
            "value": season.get("estimate", "undetermined"),
            "confidence": float(season.get("posterior", 0.0)),
            "distribution": dict(season.get("distribution") or {}),
            "evidence": season.get("primary_evidence"),
        },
        "weather": _decision_weather(weather),
        "detected_calls": [_decision_event(event) for event in events],
        "disagreements": list(disagreements),
        "overall_confidence": confidence,
        "limitations": list(limitations),
    }


def build_llm_input(decision: dict[str, Any]) -> dict[str, Any]:
    """Package the decision JSON for a future narration step."""
    return {
        "schema_version": "analysis_llm_input.v1",
        "task": (
            "Render this ecoacoustic analysis decision JSON as immersive, "
            "third-person perspective narration with an analytical tone. "
            "Narrate only the provided observations, inferred context, "
            "disagreements, limitations, timestamps, and confidence values; "
            "do not invent species, season, time of day, weather, certainty, "
            "or causes beyond the JSON."
        ),
        "decision": decision,
    }


def build_narration(decision: dict[str, Any]) -> dict[str, Any]:
    """Create a deterministic human-readable summary from the decision JSON.

    This is a local fallback for review and demo surfaces. A future LLM-OSS
    narration layer can replace this text while keeping the same decision JSON.
    """
    time_value = _human_value(decision.get("time_of_day", {}).get("value"))
    season_value = _human_value(decision.get("season", {}).get("value"))
    weather_label = _human_value(decision.get("weather", {}).get("label"))
    calls = decision.get("detected_calls") if isinstance(decision.get("detected_calls"), list) else []
    call_names = [
        call.get("common_name") or _human_value(call.get("label"))
        for call in calls
        if isinstance(call, dict) and (call.get("common_name") or call.get("label"))
    ]

    if call_names:
        call_phrase = ", ".join(call_names[:3])
        if len(call_names) > 3:
            call_phrase += f", and {len(call_names) - 3} more"
    else:
        call_phrase = "no known species calls"

    summary = (
        f"The recording is best described as {time_value} with {weather_label} weather. "
        f"The season is {season_value}. The detected call evidence includes {call_phrase}."
    )

    bullets = [
        f"Time of day: {time_value} ({_percent(decision.get('time_of_day', {}).get('confidence'))})",
        f"Season: {season_value} ({_percent(decision.get('season', {}).get('confidence'))})",
        f"Weather: {weather_label} ({_percent(decision.get('weather', {}).get('confidence'))})",
        f"Detected calls: {call_phrase}",
    ]
    caveats = list(decision.get("limitations") or [])
    disagreements = decision.get("disagreements") if isinstance(decision.get("disagreements"), list) else []
    if disagreements:
        caveats.append(f"{len(disagreements)} analysis disagreement(s) were recorded in the fused report.")

    return {
        "schema_version": "analysis_narration.v1",
        "source": "deterministic_fallback",
        "summary": summary,
        "bullets": bullets,
        "caveats": _dedupe(caveats),
    }


def fuse_context_field(
    field: str,
    evidence: list[dict[str, Any]],
    *,
    bins: tuple[str, ...],
    default_distribution: dict[str, float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fuse evidence for one latent context field.

    The reported posterior is intentionally conservative. Raw votes are mixed
    with the uniform/default distribution according to total evidence strength,
    so a single weak ambient vote cannot become a false 1.0 posterior.
    """
    weights_by_bin = {key: 0.0 for key in bins}
    weighted_evidence = []
    for item in evidence:
        candidates = [candidate for candidate in item.get("candidates", []) if candidate in bins]
        if not candidates:
            continue
        weight = _evidence_weight(item, cfg)
        if weight <= 0.0:
            continue
        split_weight = weight / len(candidates)
        for candidate in candidates:
            weights_by_bin[candidate] += split_weight
        row = dict(item)
        row["weight"] = round(weight, 6)
        weighted_evidence.append(row)

    total_weight = sum(weights_by_bin.values())
    if total_weight <= 0.0:
        return {
            "estimate": "undetermined",
            "posterior": 0.0,
            "distribution": _rounded_distribution(default_distribution, bins),
            "primary_evidence": f"No reliable {field} evidence",
            "evidence": [],
        }

    normalized_votes = {
        key: weights_by_bin[key] / total_weight
        for key in bins
    }
    evidence_strength = min(1.0, total_weight)
    distribution = {
        key: (
            (float(default_distribution.get(key, 0.0)) * (1.0 - evidence_strength))
            + (normalized_votes[key] * evidence_strength)
        )
        for key in bins
    }
    distribution = _rounded_distribution(distribution, bins)
    top_value = max(bins, key=lambda key: distribution[key])
    posterior = distribution[top_value]
    threshold = float(cfg["fusion"].get("undetermined_threshold", 0.50))
    estimate = top_value if posterior >= threshold else "undetermined"
    if estimate != "undetermined" and _ambient_fallback_too_weak(weighted_evidence, cfg):
        estimate = "undetermined"
    primary = _primary_evidence(weighted_evidence, field)

    return {
        "estimate": estimate,
        "posterior": posterior,
        "distribution": distribution,
        "primary_evidence": primary,
        "evidence": weighted_evidence,
    }


def detect_disagreements(
    *,
    season: dict[str, Any],
    diel: dict[str, Any],
    observations: dict[str, Any],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    disagreements: list[dict[str, Any]] = []
    for field, result in (("diel", diel), ("season", season)):
        source_tops = _source_top_values(result.get("evidence", []))
        ambient = source_tops.get("ambient")
        events = source_tops.get("events")
        if ambient and events and ambient["value"] != events["value"]:
            if events["weight"] >= ambient["weight"] + float(cfg["fusion"].get("conflict_margin", 0.20)):
                resolution = "events_preferred"
                reason = "Event phenology is stronger context evidence than ambient texture."
            else:
                resolution = "low_confidence_range_reported"
                reason = "Ambient and event evidence conflict without a clear winner."
            disagreements.append(
                {
                    "field": field,
                    "ambient": ambient["value"],
                    "events": events["value"],
                    "resolution": resolution,
                    "reason": reason,
                }
            )
        elif ambient and not events and result.get("estimate") != "undetermined":
            disagreements.append(
                {
                    "field": field,
                    "ambient": ambient["value"],
                    "events": "inconclusive",
                    "resolution": "ambient_used_as_fallback",
                    "reason": f"No stronger event evidence was available, so E-A provided the {field} estimate.",
                }
            )
        elif result.get("estimate") == "undetermined" and result.get("evidence"):
            disagreements.append(
                {
                    "field": field,
                    "ambient": ambient["value"] if ambient else "inconclusive",
                    "events": events["value"] if events else "inconclusive",
                    "resolution": "low_confidence_range_reported",
                    "reason": f"{field} evidence was present but too weak or broad for a precise estimate.",
                }
            )
    weather = observations.get("weather") if isinstance(observations.get("weather"), dict) else {}
    if _has_weather_observation(weather):
        disagreements.append(
            {
                "field": "weather",
                "ambient": "not_applicable",
                "events": "not_applicable",
                "resolution": "direct_observation_kept",
                "reason": "Weather is kept from E-B as a direct acoustic observation and is not overwritten by E-A or E-C.",
            }
        )
    return disagreements


def _evidence_weight(item: dict[str, Any], cfg: dict[str, Any]) -> float:
    confidence = _safe_float(item.get("confidence"))
    source = item.get("source_head")
    if source == "ambient":
        cap = float(cfg["fusion"].get("ambient_weight_cap", 0.25))
    elif source == "events":
        cap = float(cfg["fusion"].get("event_weight_cap", 1.0))
    else:
        cap = 0.0
    return round(min(max(confidence, 0.0), cap), 6)


def _ambient_fallback_too_weak(evidence: list[dict[str, Any]], cfg: dict[str, Any]) -> bool:
    if not evidence:
        return False
    if any(item.get("source_head") == "events" for item in evidence):
        return False
    if not all(item.get("source_head") == "ambient" for item in evidence):
        return False
    min_conf = float(cfg["fusion"].get("ambient_fallback_min_confidence", 0.6))
    return max(_safe_float(item.get("confidence")) for item in evidence) < min_conf


def _primary_evidence(evidence: list[dict[str, Any]], field: str) -> str:
    if not evidence:
        return f"No reliable {field} evidence"
    strongest = max(evidence, key=lambda row: float(row.get("weight", 0.0)))
    if strongest.get("source_head") == "events":
        common_name = strongest.get("common_name") or strongest.get("event_label") or "a detected event"
        return f"E-C: {common_name} supports {strongest.get('value')}"
    if strongest.get("source_head") == "ambient":
        return f"E-A: ambient head estimated {strongest.get('value')}"
    return str(strongest.get("reason") or f"Strongest {field} evidence")


def _source_top_values(evidence: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_source: dict[str, dict[str, float]] = {}
    for item in evidence:
        source = str(item.get("source_head") or "")
        by_source.setdefault(source, {})
        weight = float(item.get("weight", 0.0))
        candidates = [str(candidate) for candidate in item.get("candidates", [])]
        if not candidates:
            value = str(item.get("value") or "")
            candidates = [value] if value else []
        for candidate in candidates:
            by_source[source][candidate] = by_source[source].get(candidate, 0.0) + (weight / len(candidates))

    tops: dict[str, dict[str, Any]] = {}
    for source, values in by_source.items():
        if not values:
            continue
        value = max(values, key=values.get)
        tops[source] = {"value": value, "weight": values[value]}
    return tops


def _decision_weather(weather: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": weather.get("derived_label") or "none",
        "confidence": _safe_float(weather.get("confidence")),
        "rain": _weather_component(weather, "rain"),
        "wind": _weather_component(weather, "wind"),
        "thunder": _weather_component(weather, "thunder"),
        "warnings": list(weather.get("warnings") or []),
    }


def _has_weather_observation(weather: dict[str, Any]) -> bool:
    if not weather:
        return False
    if _safe_float(weather.get("confidence")) > 0.0:
        return True
    if weather.get("derived_label") and weather.get("derived_label") != "none":
        return True
    for key in ("rain", "wind", "thunder"):
        component = weather.get(key) if isinstance(weather.get(key), dict) else {}
        summary = component.get("summary") if isinstance(component.get("summary"), dict) else {}
        if _safe_float(summary.get("confidence")) > 0.0:
            return True
        if _safe_float(summary.get("intensity")) > 0.0:
            return True
    return False


def _weather_component(weather: dict[str, Any], key: str) -> dict[str, Any]:
    component = weather.get(key) if isinstance(weather.get(key), dict) else {}
    summary = component.get("summary") if isinstance(component.get("summary"), dict) else {}
    out = {
        "label": summary.get("label") or "unknown",
        "confidence": _safe_float(summary.get("confidence")),
        "intensity": _safe_float(summary.get("intensity")),
        "coverage": _safe_float(summary.get("coverage")),
    }
    if key == "thunder":
        out["events"] = _timeline_events(component.get("events")) if "events" in component else None
        out["mean_interval_s"] = component.get("mean_interval_s")
    return out


def _timeline_events(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    events = []
    for item in value:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row.setdefault("onset_s", row.get("start_s"))
        row.setdefault("offset_s", row.get("end_s"))
        events.append(row)
    return events


def _decision_event(event: dict[str, Any]) -> dict[str, Any]:
    phenology = event.get("phenology") if isinstance(event.get("phenology"), dict) else {}
    return {
        "label": event.get("label"),
        "common_name": event.get("common_name") or phenology.get("common_name"),
        "scientific_name": event.get("scientific_name") or phenology.get("scientific_name"),
        "confidence": _safe_float(event.get("confidence")),
        "onset_s": event.get("onset_s"),
        "offset_s": event.get("offset_s"),
        "diel_signal": phenology.get("diel_signal"),
        "diel_confidence": _safe_float(phenology.get("diel_confidence")),
        "season_signal": phenology.get("season_signal"),
        "season_confidence": _safe_float(phenology.get("season_confidence")),
        "habitat_signal": phenology.get("habitat_signal"),
    }


def _human_value(value: Any) -> str:
    if value in (None, "", "unknown"):
        return "unknown"
    if value == "undetermined":
        return "undetermined"
    return str(value).replace("_", " ")


def _percent(value: Any) -> str:
    return f"{round(_safe_float(value) * 100)}%"


def _merge_params(params: dict[str, Any] | None) -> dict[str, Any]:
    merged = {
        "fusion": dict(DEFAULT_PARAMS["fusion"]),
        "defaults": {
            "season_distribution": dict(DEFAULT_PARAMS["defaults"]["season_distribution"]),
            "diel_distribution": dict(DEFAULT_PARAMS["defaults"]["diel_distribution"]),
        },
        "limitations": list(DEFAULT_PARAMS["limitations"]),
    }
    if not params:
        return merged
    if isinstance(params.get("fusion"), dict):
        merged["fusion"].update(params["fusion"])
    if isinstance(params.get("defaults"), dict):
        defaults = params["defaults"]
        if isinstance(defaults.get("season_distribution"), dict):
            merged["defaults"]["season_distribution"].update(defaults["season_distribution"])
        if isinstance(defaults.get("diel_distribution"), dict):
            merged["defaults"]["diel_distribution"].update(defaults["diel_distribution"])
    if isinstance(params.get("limitations"), list):
        merged["limitations"] = list(params["limitations"])
    return merged


def _rounded_distribution(distribution: dict[str, float], bins: tuple[str, ...]) -> dict[str, float]:
    total = sum(max(0.0, float(distribution.get(key, 0.0))) for key in bins)
    if total <= 0.0:
        return {key: round(1.0 / len(bins), 6) for key in bins}
    return {
        key: round(max(0.0, float(distribution.get(key, 0.0))) / total, 6)
        for key in bins
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
