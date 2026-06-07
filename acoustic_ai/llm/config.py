"""Configuration for the in-process LLM-OSS service.

All values are overridable via environment variables so serverB can be tuned
without code changes. Defaults target serverB's 16 GB GPU, where a 4-bit 3B
sits resident alongside AudioLDM2 + AudioGen (see
.claude/context/ai/llm_layer_config.md §4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    """Resolved at import time from the environment.

    Primary model: Qwen2.5-3B-Instruct (4-bit, ~2.5 GB). License-clean
    alternative: microsoft/Phi-3.5-mini-instruct (MIT). See §5 of the decision
    doc for the trade-off.
    """

    # --- model identity ---
    model_id: str = os.environ.get("AI_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

    # --- loading / footprint ---
    load_in_4bit: bool = _env_bool("AI_LLM_4BIT", True)
    dtype: str = os.environ.get("AI_LLM_DTYPE", "bfloat16")   # compute dtype
    device_map: str = os.environ.get("AI_LLM_DEVICE_MAP", "auto")

    # --- generation defaults ---
    max_new_tokens: int = int(os.environ.get("AI_LLM_MAX_NEW_TOKENS", "768"))
    # The parser must be reproducible -> greedy (temperature 0). The immersive
    # report register may override with a small temperature for warmth.
    parser_temperature: float = float(os.environ.get("AI_LLM_PARSER_TEMPERATURE", "0.0"))
    report_temperature: float = float(os.environ.get("AI_LLM_REPORT_TEMPERATURE", "0.6"))

    @property
    def torch_dtype(self):  # lazy: avoid importing torch at module import
        import torch
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}.get(self.dtype, torch.bfloat16)


# Singleton config resolved once from the environment.
CONFIG = LLMConfig()
