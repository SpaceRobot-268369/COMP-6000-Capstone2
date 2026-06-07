"""Generation Prompt Parser orchestration.

Three stages (prompt_parser_policy.md): pre-fill defaults -> validity gate ->
decode into layer contracts. The LLM does the extraction/encoding/decoding
against the `parser` skill; deterministic gate findings (gate.py) are injected
into the user message and are authoritative. Returns the parse-result schema
(policy §5).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .gate import gate_findings
from .service import get_service
from .skills import load_skill

VALID_SEASONS = {"spring", "summer", "autumn", "winter"}
VALID_DIELS = {"dawn", "morning", "afternoon", "night"}
VALID_WEATHER = {"rain", "wind", "thunder"}
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
    messages = [
        {"role": "system", "content": load_skill("parser")},
        {"role": "user", "content": _user_message(prompt, findings)},
    ]
    raw = get_service().complete_json(messages, schema=PARSE_RESULT_SCHEMA)
    return _normalize(raw, had_findings=bool(findings))


__all__ = ["parse_prompt", "PARSE_RESULT_SCHEMA"]
