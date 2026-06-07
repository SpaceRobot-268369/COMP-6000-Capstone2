"""Layer E report writer.

Renders a fused analysis JSON into prose in one of two registers
(analytical / immersive). The LLM only narrates; all weighting already happened
in the deterministic aggregator (analysis_synthesis_policy.md §3, §5). A
faithfulness guard rejects + retries prose that names species absent from the
report (llm_layer_config.md §6).
"""

from __future__ import annotations

import json

from .config import CONFIG
from .faithfulness import validate_narrative
from .service import get_service
from .skills import load_skill, report_skill_name

_VALID_REGISTERS = {"analytical", "immersive"}


def normalize_register(register: str) -> str:
    reg = (register or "analytical").lower()
    return reg if reg in _VALID_REGISTERS else "analytical"


def write_report(report: dict, register: str = "analytical", *, max_retries: int = 1) -> dict:
    """Render `report` (fused analysis JSON) into prose.

    Returns {register, text, faithful, violations}. `faithful` is False (with
    `violations`) if even the final retry named an unobserved species — callers
    can surface that rather than hide it.
    """
    register = normalize_register(register)
    messages = [
        {"role": "system", "content": load_skill(report_skill_name(register))},
        {"role": "user", "content": json.dumps(report, ensure_ascii=False)},
    ]
    # Immersive wants a little warmth; analytical stays nearly deterministic.
    temp = CONFIG.report_temperature if register == "immersive" else 0.2

    svc = get_service()
    text, ok, violations = "", True, []
    for _ in range(max_retries + 1):
        text = svc.complete(messages, temperature=temp)
        ok, violations = validate_narrative(text, report)
        if ok:
            break
    return {"register": register, "text": text, "faithful": ok, "violations": violations}


__all__ = ["write_report", "normalize_register"]
