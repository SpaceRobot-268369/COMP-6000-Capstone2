"""Faithfulness guard for the Layer E report writer.

Small models occasionally embellish — naming a species the analysis never
detected. The report writer renders from a *closed* fused JSON, so we can check
that the prose introduces no species outside what the report actually observed.

This is a heuristic guard (see llm_layer_config.md §6): it catches confident
hallucinations of *known* species, not every possible fabrication. The known
vocabulary mirrors the Layer E E-C detector's species set; ideally it is sourced
from config/registry later (TODO).
"""

from __future__ import annotations

import re

# Mirrors the E-C known-species detector vocabulary (registry: songke E-C).
# TODO: source this from the registry/config instead of hardcoding.
KNOWN_SPECIES = [
    "southern boobook",
    "laughing kookaburra",
    "willie wagtail",
    "crested bellbird",
    "tawny frogmouth",
    "red-capped robin",
    "red capped robin",
    "pacific black duck",
    "australian raven",
    "peaceful dove",
    "galah",
    "rainbow bee-eater",
    "rainbow bee eater",
]


def _norm(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _add_event_aliases(out: set[str], items: object) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("label", "common_name"):
            alias = _norm(item.get(key))
            if alias:
                out.add(alias)


def _allowed_species(report: dict) -> set[str]:
    out: set[str] = set()
    source = report or {}
    observations = source.get("observations") if isinstance(source, dict) else {}
    decision = source.get("decision") if isinstance(source, dict) else {}
    llm_input = source.get("llm_input") if isinstance(source, dict) else {}
    llm_decision = llm_input.get("decision") if isinstance(llm_input, dict) else {}

    if isinstance(observations, dict):
        _add_event_aliases(out, observations.get("events"))
    if isinstance(decision, dict):
        _add_event_aliases(out, decision.get("detected_calls"))
    if isinstance(llm_decision, dict):
        _add_event_aliases(out, llm_decision.get("detected_calls"))
    return out


def validate_narrative(narrative: str, report: dict) -> tuple[bool, list[str]]:
    """Return (ok, violations).

    A violation = a known species named in the prose that the report did not
    observe. ``ok`` is True when there are no violations.
    """
    text = _norm(narrative)
    allowed = _allowed_species(report)
    violations: list[str] = []
    for sp in KNOWN_SPECIES:
        species = _norm(sp)
        # word-ish boundary match so "galah" doesn't fire inside another word
        if re.search(rf"\b{re.escape(species)}\b", text) and not any(
            species in a or a in species for a in allowed
        ):
            violations.append(species)
    # de-dup while preserving order
    seen: set[str] = set()
    deduped = [v for v in violations if not (v in seen or seen.add(v))]
    return (len(deduped) == 0, deduped)


__all__ = ["validate_narrative", "KNOWN_SPECIES"]
