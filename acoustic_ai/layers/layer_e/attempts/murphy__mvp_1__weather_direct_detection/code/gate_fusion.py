"""Rule-based multi-model fusion for Layer E-B weather analysis.

This module turns element-level evidence into a weather decision. It is
intentionally transparent: E-B should report mixed weather by detecting rain,
wind, and thunder independently, then deriving the composite label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WEATHER_ELEMENTS = ("rain", "wind", "thunder")


@dataclass(frozen=True)
class ElementDecision:
    present: bool
    intensity: str
    confidence: float
    coverage: float = 0.0


def _score(channel: dict[str, float] | None, element: str) -> float:
    if not channel:
        return 0.0
    return float(channel.get(element, 0.0))


def top_weather(channel: dict[str, float] | None) -> str:
    if not channel:
        return "none"
    top = max(WEATHER_ELEMENTS, key=lambda element: _score(channel, element))
    if _score(channel, top) <= 0.0:
        return "none"
    return top


def _confidence_band(level: str, raw_confidence: float) -> float:
    """Map uncalibrated model scores into honest MVP confidence bands."""
    if level == "strong":
        return max(0.72, min(0.90, 0.72 + 0.20 * raw_confidence))
    if level == "moderate":
        return max(0.58, min(0.72, 0.52 + 0.18 * raw_confidence))
    if level == "weak":
        return max(0.42, min(0.58, 0.36 + 0.18 * raw_confidence))
    return max(0.0, min(0.42, 0.20 + 0.20 * raw_confidence))


def _intensity(element: str, confidence: float, coverage: float, warnings: set[str]) -> str:
    if confidence <= 0.0:
        return "none"
    if element == "thunder" and "possible_wind_overload" in warnings:
        return "light"
    if confidence >= 0.78 or coverage >= 0.65:
        return "heavy"
    if confidence >= 0.64 or coverage >= 0.35:
        return "medium"
    return "light"


def decide_weather_from_evidence(
    evidence: dict[str, dict[str, float]],
    coverage: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return schema-shaped weather decision from multi-model evidence.

    Expected evidence channels are optional and may include:

    - ``clap``
    - ``panns``
    - ``ast``
    - ``beats``
    - ``features``

    Missing channels are treated as zero evidence. The current MVP policy uses
    CLAP as the sensitive detector and AST/BEATs as conservative guards.
    """
    coverage = coverage or {}
    clap = evidence.get("clap", {})
    panns = evidence.get("panns", {})
    ast = evidence.get("ast", {})
    beats = evidence.get("beats", {})
    features = evidence.get("features", {})
    beats_available = bool(beats)

    present: list[str] = []
    warnings: set[str] = set()
    raw_confidence: dict[str, float] = {}
    confidence_level: dict[str, str] = {element: "absent" for element in WEATHER_ELEMENTS}

    # Rain: CLAP should see rain-like sound, and BEATs should not contradict it.
    # PANNs rain is useful but broad on storm/wind textures, so it is not enough
    # by itself.
    beats_rain_support = (
        top_weather(beats) == "rain"
        or _score(beats, "rain") >= _score(beats, "wind") + 0.03
    )
    panns_rain_support = (
        top_weather(panns) == "rain"
        or (
            _score(panns, "rain") >= 0.35
            and _score(clap, "rain") >= max(_score(clap, "wind"), _score(clap, "thunder"))
        )
    )
    ast_rain_support = top_weather(ast) == "rain" and _score(ast, "rain") >= 0.45
    rain_candidate = _score(clap, "rain") >= 0.52
    raw_confidence["rain"] = (
        0.56 * _score(clap, "rain")
        + 0.16 * _score(panns, "rain")
        + 0.18 * _score(beats, "rain")
        + 0.10 * _score(ast, "rain")
        + 0.05 * _score(features, "rain")
    )
    if rain_candidate and (
        beats_rain_support if beats_available else (panns_rain_support or ast_rain_support)
    ):
        present.append("rain")
        confidence_level["rain"] = "moderate"
        if not beats_available:
            warnings.add("rain_confirmed_without_beats_guard")
    elif rain_candidate and _score(panns, "rain") >= 0.35:
        warnings.add("possible_rain_under_wind")
        confidence_level["rain"] = "weak"

    # Wind: accept if CLAP sees wind and AST/BEATs or CLAP ranking supports it.
    wind_support = (
        top_weather(ast) == "wind"
        or top_weather(beats) == "wind"
        or _score(clap, "wind") >= max(_score(clap, "rain"), _score(clap, "thunder"))
    )
    raw_confidence["wind"] = (
        0.62 * _score(clap, "wind")
        + 0.20 * _score(ast, "wind")
        + 0.18 * _score(beats, "wind")
        + 0.05 * _score(features, "wind")
    )
    if _score(clap, "wind") >= 0.52 and wind_support:
        present.append("wind")
        confidence_level["wind"] = "moderate"

    # Thunder: PANNs is sensitive but overcalls thunder on heavy wind. Require
    # CLAP + PANNs and at least one conservative channel to not reject it.
    strong_thunder = (
        _score(clap, "thunder") >= 0.60
        and _score(panns, "thunder") >= 0.45
        and (top_weather(ast) == "thunder" or top_weather(beats) == "thunder")
    )
    maybe_thunder = (
        _score(clap, "thunder") >= 0.56
        and _score(panns, "thunder") >= 0.55
        and top_weather(ast) == "thunder"
    )
    raw_confidence["thunder"] = (
        0.42 * _score(clap, "thunder")
        + 0.30 * _score(panns, "thunder")
        + 0.18 * _score(ast, "thunder")
        + 0.10 * _score(beats, "thunder")
        + 0.05 * _score(features, "thunder")
    )
    if strong_thunder:
        present.append("thunder")
        confidence_level["thunder"] = "moderate"
    elif maybe_thunder:
        present.append("thunder")
        confidence_level["thunder"] = "weak"
    if _score(clap, "thunder") >= 0.52 or _score(panns, "thunder") >= 0.45:
        warnings.add("possible_wind_overload")

    present = [element for index, element in enumerate(present) if element not in present[:index]]
    if len(present) >= 2:
        warnings.add("weather_mixed_with_ambient")

    elements: dict[str, dict[str, float | bool | str]] = {}
    for element in WEATHER_ELEMENTS:
        element_present = element in present
        confidence = (
            _confidence_band(confidence_level[element], raw_confidence[element])
            if element_present
            else _confidence_band("absent", raw_confidence[element])
        )
        element_coverage = float(coverage.get(element, 0.0))
        elements[element] = {
            "present": element_present,
            "intensity": (
                _intensity(element, confidence, element_coverage, warnings)
                if element_present
                else "none"
            ),
            "confidence": round(float(confidence), 6),
            "coverage": round(element_coverage, 6),
        }

    label = "+".join(present) if present else "none"
    return {
        "overall_label": label,
        "none": not bool(present),
        "elements": elements,
        "warnings": sorted(warnings),
        "debug": {
            "raw_confidence": {
                element: round(float(value), 6)
                for element, value in raw_confidence.items()
            },
            "tops": {
                "clap": top_weather(clap),
                "panns": top_weather(panns),
                "ast": top_weather(ast),
                "beats": top_weather(beats),
            },
        },
    }
