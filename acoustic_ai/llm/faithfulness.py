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


def _allowed_species(report: dict) -> set[str]:
    events = (report or {}).get("observations", {}).get("events", []) or []
    out: set[str] = set()
    for e in events:
        label = str(e.get("label", "")).strip().lower()
        if label:
            out.add(label)
    return out


def validate_narrative(narrative: str, report: dict) -> tuple[bool, list[str]]:
    """Return (ok, violations).

    A violation = a known species named in the prose that the report did not
    observe. ``ok`` is True when there are no violations.
    """
    text = (narrative or "").lower()
    allowed = _allowed_species(report)
    violations: list[str] = []
    for sp in KNOWN_SPECIES:
        # word-ish boundary match so "galah" doesn't fire inside another word
        if re.search(rf"\b{re.escape(sp)}\b", text) and not any(sp in a or a in sp for a in allowed):
            violations.append(sp)
    # de-dup while preserving order
    seen: set[str] = set()
    deduped = [v for v in violations if not (v in seen or seen.add(v))]
    return (len(deduped) == 0, deduped)


__all__ = ["validate_narrative", "KNOWN_SPECIES"]
