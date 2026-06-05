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
        "ambient_weight_cap": 0.25,
        "event_weight_cap": 1.0,
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

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "analysis",
        "observations": adapted["observations"],
        "inferred_context": {
            "diel": diel,
            "season": season,
        },
        "disagreements": disagreements,
        "confidence": confidence,
        "limitations": _dedupe(limitations),
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
        elif result.get("estimate") == "undetermined" and result.get("evidence"):
            disagreements.append(
                {
                    "field": field,
                    "ambient": ambient["value"] if ambient else None,
                    "events": events["value"] if events else None,
                    "resolution": "low_confidence_range_reported",
                    "reason": f"{field} evidence was present but too weak or broad for a precise estimate.",
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
