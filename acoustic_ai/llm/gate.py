"""Deterministic validity-gate checks for the Prompt Parser.

Per the policy, the *decision* of what is plausible is deterministic — the LLM
only encodes/narrates it (prompt_parser_policy.md §3, llm_layer_config.md §2).
This module produces structured `gate_findings` that are passed into the
parser's user message; the LLM must honor them.

Two severities for out-of-domain content (the balance the gate has to strike):

- **swap** — a recoverable prompt that names one or two stray anthropic
  elements over an in-domain scene ("autumn dawn in the city with rain"). The
  flagged element is removed and the rest is kept (correct-and-continue).
- **block** — a prompt *saturated* with out-of-domain content, with no
  in-domain scene left to anchor ("midday city traffic, car horns, sirens, a
  subway train"). There is nothing to correct *into*, so the parser rejects it
  deterministically instead of silently emitting a default empty bed.

The swap/block split is a distinct-concept count (`OUT_OF_DOMAIN_BLOCK_THRESHOLD`)
so a single concept written with several words ("car horns") and negated
mentions ("no traffic") do not push a recoverable prompt over the edge.

Scope today: out-of-domain anthropogenic content, off-site natural biomes
(coast/ocean), and climatically-implausible weather for an arid inland site.
The fauna-phenology check is a STUB pending the species phenology table
(analysis_synthesis_policy.md §7) — see plan §10.
"""

from __future__ import annotations

import re

# Out-of-domain anthropogenic sources, grouped by concept. Each concept counts
# once toward the block threshold no matter how many of its synonyms appear, so
# "car horns" (car + horn) is two concepts, but "cars" and "car" are one.
OUT_OF_DOMAIN_GROUPS: dict[str, tuple[str, ...]] = {
    "traffic": ("traffic",),
    "siren": ("siren", "sirens"),
    "car": ("car", "cars", "automobile"),
    "horn": ("horn", "horns", "honk", "honking"),
    "truck": ("truck", "trucks", "lorry"),
    "engine": ("engine", "motor"),
    "machinery": ("machinery", "machine"),
    "music": ("music", "song playing"),
    "crowd": ("crowd", "crowds"),
    "city": ("city", "urban", "downtown"),
    "train": ("train", "subway", "tram"),
    "aircraft": ("helicopter", "airplane", "aeroplane", "plane", "jet"),
    "factory": ("factory",),
    "construction": ("construction", "jackhammer"),
    "gunshot": ("gunshot", "gunfire"),
    "explosion": ("explosion", "fireworks"),
}

# Flat synonym list kept for backward compatibility / introspection.
OUT_OF_DOMAIN = sorted({term for terms in OUT_OF_DOMAIN_GROUPS.values() for term in terms})

# Climatically implausible for arid inland Bowra.
IMPLAUSIBLE_WEATHER = ["snow", "snowy", "snowfall", "blizzard", "hail", "sleet", "frost storm"]

# Off-site natural elements: real sounds, but from a biome the inland dry-
# woodland site cannot host (the coast). Unlike anthropogenic content these are
# always recoverable — strip the off-biome element and keep the in-domain scene
# (correct-and-continue) — so each is a `swap` and is NOT counted toward the
# anthropogenic saturation block. Grouped by concept like OUT_OF_DOMAIN_GROUPS,
# so "ocean waves breaking on a beach" is one `coastal` concept, not three.
# "waves" is deliberately omitted: too ambiguous (heat/sound waves) to flag on
# its own — ocean/beach/surf/etc. already anchor the coastal case.
OFF_BIOME_GROUPS: dict[str, tuple[str, ...]] = {
    "coastal": (
        "ocean", "oceans", "sea", "seas", "seaside", "seashore", "shore",
        "shoreline", "beach", "beaches", "coast", "coastal", "surf", "tide",
        "tides", "lagoon",
    ),
}

# At/above this many DISTINCT out-of-domain concepts the prompt is treated as
# saturated (unrecoverable) and blocked. Below it, each concept is a swap and
# the in-domain remainder is kept. Tunable — raise to be more lenient, lower to
# be stricter.
OUT_OF_DOMAIN_BLOCK_THRESHOLD = 3

# Negation cues: a flagged term preceded by one of these is the user asking for
# its *absence* ("no traffic", "without cars") — which is already the default,
# so it is neither swapped nor counted toward the block threshold.
_NEGATION_CUES = {
    "no", "not", "without", "absent", "sans", "minus", "lacking",
    "devoid", "free", "zero", "none", "nor", "never",
}
_NEG_WINDOW = 28  # chars of preceding context scanned for a negation cue


def _is_negated(text: str, start: int) -> bool:
    """True if the term beginning at `start` is negated in its left context."""
    prefix_words = re.findall(r"[a-z]+", text[max(0, start - _NEG_WINDOW):start])
    return any(word in _NEGATION_CUES for word in prefix_words[-3:])


def _hit_concept(text: str, terms: tuple[str, ...]) -> bool:
    """True if any (non-negated) synonym of a concept occurs in `text`."""
    for term in terms:
        for m in re.finditer(rf"\b{re.escape(term)}\b", text):
            if not _is_negated(text, m.start()):
                return True
    return False


def gate_findings(prompt: str) -> list[dict]:
    """Return a list of deterministic gate findings for `prompt`.

    Each finding: {type, term, action, suggestion?}. Empty list = nothing flagged.
    When out-of-domain concepts saturate the prompt (>= threshold) every one is
    marked `action: "block"` and a summary `dominant_out_of_domain` finding is
    appended; the parser turns that into a deterministic `rejected`.
    """
    text = (prompt or "").lower()
    findings: list[dict] = []

    hit_concepts = [name for name, terms in OUT_OF_DOMAIN_GROUPS.items()
                    if _hit_concept(text, terms)]
    dominant = len(hit_concepts) >= OUT_OF_DOMAIN_BLOCK_THRESHOLD
    action = "block" if dominant else "swap"

    for name in hit_concepts:
        findings.append({
            "type": "out_of_domain",
            "term": name,
            "action": action,
            "suggestion": "remove the anthropogenic element; keep a remote dry-woodland scene",
        })

    if dominant:
        findings.append({
            "type": "dominant_out_of_domain",
            "term": ", ".join(hit_concepts),
            "action": "block",
            "suggestion": "this prompt is mostly outside the site and has no in-domain scene to keep; "
                          "reject and suggest a remote dry-woodland scene "
                          "(e.g. 'a still autumn dawn with distant birds')",
        })

    for kw in IMPLAUSIBLE_WEATHER:
        if kw in text:
            findings.append({
                "type": "implausible_weather",
                "term": kw,
                "action": "swap",
                "suggestion": "substitute a plausible analogue (heavy rain or dust-laden wind)",
            })

    # Off-site biome elements (coast/ocean): always a recoverable swap, never a
    # block, so a prompt like "...Nightjar calling, with ocean waves" is
    # corrected (drop the coast) rather than accepted as-is or rejected.
    for name, terms in OFF_BIOME_GROUPS.items():
        if _hit_concept(text, terms):
            findings.append({
                "type": "off_biome",
                "term": name,
                "action": "swap",
                "suggestion": "remove the coastal/marine element; keep a remote inland dry-woodland scene",
            })

    # TODO(phenology): fauna-plausibility check needs the species phenology
    # table (analysis_synthesis_policy.md §7). Until it exists, fauna requests
    # pass through unchecked here and rely on the LLM's coarse knowledge.

    return findings


__all__ = [
    "gate_findings",
    "OUT_OF_DOMAIN",
    "OUT_OF_DOMAIN_GROUPS",
    "OUT_OF_DOMAIN_BLOCK_THRESHOLD",
    "IMPLAUSIBLE_WEATHER",
    "OFF_BIOME_GROUPS",
]
