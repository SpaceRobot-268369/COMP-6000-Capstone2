"""System-prompt builders for the two LLM-OSS consumers.

These are stubs that encode the *spirit* of the policies; the policy docs
remain the source of truth and must be kept in sync:

  - parser_system_prompt  -> prompt_parser_policy.md
  - report_system_prompt  -> analysis_synthesis_policy.md §5
"""

from __future__ import annotations

from .parser_system import parser_system_prompt
from .report_system import report_system_prompt

__all__ = ["parser_system_prompt", "report_system_prompt"]
