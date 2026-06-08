"""In-process LLM-OSS service.

One small open-source instruct model, loaded once inside the FastAPI app on
:8000 (no separate server, no extra port), serving both LLM-OSS consumers:

  - the generation **Prompt Parser**  (parser.py / prompt_parser_policy.md)
  - the Layer E **report writer**       (report.py / analysis_synthesis_policy.md §5)

Skills (per-job instruction sets) live as separate files in skills/ and are
loaded as the system message; job data is the user message (plan §2.1).

Decision record + rationale: .claude/context/ai/llm_layer_config.md
Build plan: llm_layer_implementation_plan.md (in this directory)

Public API:

    from llm import get_service, warm, parse_prompt, write_report
"""

from __future__ import annotations

from .config import CONFIG, LLMConfig
from .parser import parse_prompt
from .report import write_report
from .service import LLMService, get_service, warm

__all__ = [
    "CONFIG", "LLMConfig", "LLMService",
    "get_service", "warm", "parse_prompt", "write_report",
]
