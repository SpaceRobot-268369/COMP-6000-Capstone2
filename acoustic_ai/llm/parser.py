"""Generation Prompt Parser orchestration.

Three stages (prompt_parser_policy.md): pre-fill defaults -> validity gate ->
decode into layer contracts. The LLM does the extraction/encoding/decoding
against the `parser` skill; deterministic gate findings (gate.py) are injected
into the user message and are authoritative. Returns the parse-result schema
(policy §5).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .gate import gate_findings
from .service import get_service
from .skills import load_skill

VALID_SEASONS = {"spring", "summer", "autumn", "winter"}
VALID_DIELS = {"dawn", "morning", "afternoon", "night"}
VALID_WEATHER = {"rain", "wind", "rain+wind"}
VALID_INTENSITIES = {"light", "medium", "heavy"}
VALID_DENSITY = {"sparse", "medium", "dense"}

# Enforced output schema for constrained decoding. The schema is owned by code;
# its human-readable description lives in skills/parser.md (plan §2.1).
PARSE_RESULT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "corrected", "rejected"]},
        "note": {"type": "string"},
        "filled_defaults": {"type": "array", "items": {"type": "string"}},
        "layer_a": {"type": ["object", "null"]},
        "layer_b": {"type": ["object", "null"]},
        "layer_c": {"type": ["object", "null"]},
    },
    "required": ["status", "note", "layer_a", "layer_b", "layer_c"],
}


def _user_message(prompt: str, findings: list[dict]) -> str:
    return json.dumps({"prompt": prompt, "gate_findings": findings}, ensure_ascii=False)


def _blocking_finding(findings: list[dict]) -> Optional[dict]:
    """The first gate finding that demands an outright reject, if any.

    A block decision is deterministic (gate.py owns it), not left to the small
    model — so a saturated out-of-domain prompt is rejected even if the LLM
    tries to 'correct' it into a default empty bed."""
    return next((f for f in findings if f.get("action") == "block"), None)


def _rejected(note: str) -> dict:
    return {"status": "rejected", "note": note, "filled_defaults": [],
            "layer_a": None, "layer_b": None, "layer_c": None}


# Deterministic weather detection — a backstop so an explicitly-requested,
# climatically-plausible weather is never dropped by a small model's slip
# (validation showed a 3B writing the right note but nulling layer_b). Weather
# type/intensity are trivial to detect by keyword; the LLM stays in charge of
# the fuzzier coherence work. Priority storm/rain+wind > rain > wind. The
# current Layer B site pool does not expose thunder directly; thunder/storm
# language is represented by the closest supported site weather type.
_WEATHER_TERMS = [
    ("rain+wind", ("thunder", "thunderstorm", "storm", "lightning", "rain and wind", "wind and rain")),
    ("rain", ("rain", "raining", "rainy", "drizzle", "downpour", "pouring", "showers")),
    ("wind", ("wind", "windy", "breeze", "breezy", "gust", "gusty", "gale")),
]
_INTENSITY_TERMS = [
    ("light", ("light", "gentle", "soft", "faint", "drizzle", "breeze", "slight")),
    ("heavy", ("heavy", "strong", "pouring", "downpour", "torrential", "fierce", "gale", "howling")),
    ("medium", ("medium", "moderate", "steady")),
]


def _detect_weather(prompt: str) -> Optional[dict]:
    text = (prompt or "").lower()

    def _hit(terms: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(t)}\b", text) for t in terms)

    wtype = next((wt for wt, terms in _WEATHER_TERMS if _hit(terms)), None)
    if not wtype:
        return None
    intensity = next((i for i, terms in _INTENSITY_TERMS if _hit(terms)), "medium")
    return {"weather_type": wtype, "intensity": intensity, "duration_s": 10.0}


def _norm_enum(value: Any, allowed: set[str]) -> Optional[str]:
    if isinstance(value, str) and value.lower() in allowed:
        return value.lower()
    return None


def _normalize(raw: dict, *, had_findings: bool) -> dict:
    """Safety net: guarantee the parse-result contract regardless of LLM slips.
    Applies policy defaults (ambient always on; weather off; events empty)."""
    raw = raw if isinstance(raw, dict) else {}
    status = raw.get("status")
    if status not in {"ok", "corrected", "rejected"}:
        status = "corrected" if had_findings else "ok"

    note = str(raw.get("note") or "")
    filled = raw.get("filled_defaults")
    filled = [str(x) for x in filled] if isinstance(filled, list) else []

    if status == "rejected":
        return {"status": "rejected", "note": note,
                "filled_defaults": filled,
                "layer_a": None, "layer_b": None, "layer_c": None}

    # Layer A — always on; season/diel may be null (server falls back to default cell).
    la_in = raw.get("layer_a") or {}
    layer_a = {
        "season": _norm_enum(la_in.get("season"), VALID_SEASONS),
        "diel": _norm_enum(la_in.get("diel"), VALID_DIELS),
    }

    # Layer B — off (null) unless a valid weather request is present.
    lb_in = raw.get("layer_b")
    layer_b = None
    if isinstance(lb_in, dict):
        wtype = _norm_enum(lb_in.get("weather_type"), VALID_WEATHER)
        if wtype:
            intensity = _norm_enum(lb_in.get("intensity"), VALID_INTENSITIES) or "medium"
            try:
                dur = float(lb_in.get("duration_s", 10.0))
            except (TypeError, ValueError):
                dur = 10.0
            layer_b = {"weather_type": wtype, "intensity": intensity, "duration_s": dur}
    if layer_b is None and "weather:none" not in filled:
        filled.append("weather:none")

    # Layer C — empty checklist unless species named.
    lc_in = raw.get("layer_c") or {}
    species = lc_in.get("species")
    species = [str(s) for s in species] if isinstance(species, list) else []
    density = _norm_enum(lc_in.get("density"), VALID_DENSITY) or "sparse"
    layer_c = {"species": species, "density": density}
    if not species and "events:empty" not in filled:
        filled.append("events:empty")

    return {"status": status, "note": note, "filled_defaults": filled,
            "layer_a": layer_a, "layer_b": layer_b, "layer_c": layer_c}


def parse_prompt(prompt: str) -> dict:
    """Parse a raw NL prompt into the parse-result contract. Raises if the LLM
    service is unavailable (the endpoint maps that to 503)."""
    findings = gate_findings(prompt)

    # Deterministic block: a prompt saturated with out-of-domain content has no
    # in-domain scene to correct into, so reject up front without spending an
    # LLM call (and without trusting a small model to reject on its own).
    blocking = _blocking_finding(findings)
    if blocking is not None:
        return _rejected(
            "This is a remote dry-woodland soundscape — it can't voice a city/"
            "machinery scene, and there's nothing in-domain left to keep. "
            "Try something like 'a still autumn dawn with distant birds'."
        )

    messages = [
        {"role": "system", "content": load_skill("parser")},
        {"role": "user", "content": _user_message(prompt, findings)},
    ]
    raw = get_service().complete_json(messages, schema=PARSE_RESULT_SCHEMA)
    result = _normalize(raw, had_findings=bool(findings))

    # Deterministic weather backstop: if the prompt plainly requests a plausible
    # weather the model dropped, restore it. Skip when the request was rejected,
    # the model already set weather, or the gate flagged the weather implausible
    # (the model owns that swap).
    if result["status"] != "rejected" and result.get("layer_b") is None:
        implausible = any(f.get("type") == "implausible_weather" for f in findings)
        detected = _detect_weather(prompt)
        if detected and not implausible:
            result["layer_b"] = detected
            result["filled_defaults"] = [d for d in result["filled_defaults"]
                                         if d != "weather:none"]
    return result


__all__ = ["parse_prompt", "PARSE_RESULT_SCHEMA"]
