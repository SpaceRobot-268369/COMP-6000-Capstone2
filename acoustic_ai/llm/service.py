"""LLMService — the in-process model wrapper.

Loads a small instruct model with `transformers` (+ optional 4-bit
quantization) and exposes a chat-style `complete()`. Heavy imports (`torch`,
`transformers`) are deferred to `_load()` so importing this module is cheap and
safe in environments without the ML stack (lint, tests that mock the service).

This is intentionally a thin scaffold:
  - `complete()` works today (greedy / sampled chat completion).
  - `complete_json()` has a working prompt-instructed fallback and a clearly
    marked TODO hook for grammar-constrained decoding (lm-format-enforcer /
    outlines) — see .claude/context/ai/llm_layer_config.md §6.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

from .config import CONFIG, LLMConfig

log = logging.getLogger("soundscape.llm")

# Chat message = {"role": "system"|"user"|"assistant", "content": str}
Message = dict[str, str]


class LLMService:
    """Wraps one loaded instruct model. Thread-safe lazy load; one instance
    serves both the Prompt Parser and the Layer E report writer."""

    def __init__(self, config: LLMConfig = CONFIG) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Eager-load the model + tokenizer. Idempotent and thread-safe."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._load()

    def _load(self) -> None:
        import torch  # noqa: F401  (ensures the backend is importable)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        log.info("[llm] loading %s (4bit=%s, device_map=%s)",
                 cfg.model_id, cfg.load_in_4bit, cfg.device_map)

        model_kwargs: dict[str, Any] = {
            "device_map": cfg.device_map,
            "torch_dtype": cfg.torch_dtype,
        }
        if cfg.load_in_4bit:
            # bitsandbytes NF4 — ~2.5 GB for a 3B. Requires `bitsandbytes`
            # installed in the venv (see decision doc §8 open follow-ups).
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=cfg.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
        self._model.eval()
        log.info("[llm] ready: %s", cfg.model_id)

    # -- inference ----------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        prefix_allowed_tokens_fn=None,
    ) -> str:
        """Run a chat-style completion and return the assistant text only.

        `temperature` None -> use the model default; 0.0 -> greedy (use this
        for the parser so contracts are reproducible). `prefix_allowed_tokens_fn`
        constrains decoding (e.g. grammar-enforced JSON); see `complete_json`.
        """
        self.load()
        import torch

        temp = self.config.parser_temperature if temperature is None else temperature
        max_new = max_new_tokens or self.config.max_new_tokens

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new}
        if temp and temp > 0.0:
            gen_kwargs.update(do_sample=True, temperature=temp)
        else:
            gen_kwargs.update(do_sample=False)
        if prefix_allowed_tokens_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        # Strip the prompt tokens; decode only the newly generated tail.
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _json_prefix_fn(self, schema: Optional[dict]):
        """Build a grammar-constrained decoding function for `schema` using
        lm-format-enforcer, if installed. Returns None when unavailable (caller
        falls back to tolerant extraction)."""
        if not schema:
            return None
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )
        except ImportError:
            log.warning("[llm] lm-format-enforcer not installed; JSON is not "
                        "grammar-constrained (using tolerant extraction).")
            return None
        try:
            parser = JsonSchemaParser(schema)
            return build_transformers_prefix_allowed_tokens_fn(self._tokenizer, parser)
        except Exception as exc:  # noqa: BLE001 — schema not enforceable -> fall back
            log.warning("[llm] could not build JSON grammar (%s); falling back "
                        "to tolerant extraction.", exc)
            return None

    def complete_json(
        self,
        messages: list[Message],
        *,
        schema: Optional[dict] = None,
        max_new_tokens: Optional[int] = None,
    ) -> dict:
        """Return parsed JSON from the model. Used by the Prompt Parser, whose
        output must validate against the parse-result schema
        (prompt_parser_policy.md §5).

        When lm-format-enforcer is installed and `schema` is provided, decoding
        is grammar-constrained so the output is valid JSON by construction.
        Otherwise it falls back to a greedy completion + tolerant extraction
        (good enough for a scaffold but not a hard guarantee).
        """
        self.load()  # needed so the tokenizer exists before building the fn
        prefix_fn = self._json_prefix_fn(schema)
        raw = self.complete(
            messages, temperature=0.0, max_new_tokens=max_new_tokens,
            prefix_allowed_tokens_fn=prefix_fn,
        )
        return _extract_json(raw)


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a model completion (fallback path).

    Replace with grammar-constrained decoding for production (see
    complete_json TODO)."""
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start:end + 1])
        raise


# --- module singleton ------------------------------------------------------

_service: Optional[LLMService] = None
_service_lock = threading.Lock()


def get_service() -> LLMService:
    """Lazy singleton — one model instance per process, shared by both
    consumers."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = LLMService()
    return _service


def warm() -> None:
    """Eager-load the model. Call from the server lifespan pre-warm (opt-in via
    AI_LLM_PREWARM) to keep the cold load off the first request — same pattern
    as the audio layer defaults (registry.prewarm_defaults)."""
    get_service().load()
