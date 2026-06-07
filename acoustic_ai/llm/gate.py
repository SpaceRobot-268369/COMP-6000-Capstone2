"""Deterministic validity-gate checks for the Prompt Parser.

Per the policy, the *decision* of what is plausible is deterministic — the LLM
only encodes/narrates it (prompt_parser_policy.md §3, llm_layer_config.md §2).
This module produces structured `gate_findings` that are passed into the
parser's user message; the LLM must honor them.

Scope today: out-of-domain content + climatically-implausible weather for an
arid inland site. The fauna-phenology check is a STUB pending the species
phenology table (analysis_synthesis_policy.md §7) — see plan §10.
"""

from __future__ import annotations

# Things that don't belong in a remote dry-woodland soundscape.
OUT_OF_DOMAIN = [
    "traffic", "siren", "car", "cars", "truck", "engine", "motor", "machinery",
    "machine", "music", "song playing", "crowd", "city", "urban", "train",
    "helicopter", "airplane", "aeroplane", "plane", "factory", "construction",
    "gunshot", "explosion",
]

# Climatically implausible for arid inland Bowra.
IMPLAUSIBLE_WEATHER = ["snow", "snowy", "snowfall", "blizzard", "hail", "sleet", "frost storm"]


def gate_findings(prompt: str) -> list[dict]:
    """Return a list of deterministic gate findings for `prompt`.

    Each finding: {type, term, action, suggestion?}. Empty list = nothing flagged.
    """
    text = (prompt or "").lower()
    findings: list[dict] = []

    for kw in OUT_OF_DOMAIN:
        if kw in text:
            findings.append({
                "type": "out_of_domain",
                "term": kw,
                "action": "swap",
                "suggestion": "remove the anthropogenic element; keep a remote dry-woodland scene",
            })

    for kw in IMPLAUSIBLE_WEATHER:
        if kw in text:
            findings.append({
                "type": "implausible_weather",
                "term": kw,
                "action": "swap",
                "suggestion": "substitute a plausible analogue (heavy rain or dust-laden wind)",
            })

    # TODO(phenology): fauna-plausibility check needs the species phenology
    # table (analysis_synthesis_policy.md §7). Until it exists, fauna requests
    # pass through unchecked here and rely on the LLM's coarse knowledge.

    return findings


__all__ = ["gate_findings", "OUT_OF_DOMAIN", "IMPLAUSIBLE_WEATHER"]
