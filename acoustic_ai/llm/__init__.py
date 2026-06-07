"""In-process LLM-OSS service.

One small open-source instruct model, loaded once inside the FastAPI app on
:8000 (no separate server, no extra port), serving both LLM-OSS consumers:

  - the generation **Prompt Parser**  (prompt_parser_policy.md)
  - the Layer E **report writer**       (analysis_synthesis_policy.md §5)

Decision record + rationale: .claude/context/ai/llm_layer_config.md

Public API:

    from llm import get_service, warm

    svc = get_service()                      # lazy singleton
    text = svc.complete(messages)            # chat-style completion
    warm()                                   # eager-load for pre-warm
"""

from __future__ import annotations

from .config import LLMConfig
from .service import LLMService, get_service, warm

__all__ = ["LLMConfig", "LLMService", "get_service", "warm"]
