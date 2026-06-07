# `acoustic_ai/llm/` — In-process LLM-OSS service

One small open-source instruct model, loaded **once inside the FastAPI app on
`:8000`** (no separate server, no extra port), serving both LLM-OSS consumers:

- the generation **Prompt Parser** ([prompt_parser_policy.md](../../.claude/context/ai/prompt_parser_policy.md))
- the Layer E **report writer** ([analysis_synthesis_policy.md §5](../../.claude/context/ai/analysis_synthesis_policy.md))

> **Decision record + rationale (model choice, VRAM budget, why a small model
> is strong enough):**
> [.claude/context/ai/llm_layer_config.md](../../.claude/context/ai/llm_layer_config.md)

This is **not** a registry layer/attempt — it generates no seed-based audio and
has no `handler.py`. It is a shared module imported by the generation
orchestrator and the Layer E aggregator.

## Layout

```
acoustic_ai/llm/
├── __init__.py          # public API: get_service(), warm(), LLMService, LLMConfig
├── config.py            # env-driven config (model id, 4-bit, gen params)
├── service.py           # LLMService: lazy load + complete() / complete_json()
├── prompts/
│   ├── parser_system.py # system prompt for the Prompt Parser
│   └── report_system.py # system prompt for the report writer (2 registers)
└── README.md            # this file
```

## Usage

```python
from llm import get_service
from llm.prompts import parser_system_prompt, report_system_prompt

svc = get_service()  # lazy singleton; first call loads the model

# Prompt parser (JSON contract, greedy for reproducibility)
contract = svc.complete_json([
    {"role": "system", "content": parser_system_prompt()},
    {"role": "user",   "content": "a misty autumn dawn"},
])

# Report writer (immersive register)
prose = svc.complete(
    [
        {"role": "system", "content": report_system_prompt("immersive")},
        {"role": "user",   "content": fused_json_str},
    ],
    temperature=0.6,
)
```

## Config (env vars)

| Var | Default | Meaning |
|---|---|---|
| `AI_LLM_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | HF model id. License-clean alt: `microsoft/Phi-3.5-mini-instruct`. |
| `AI_LLM_4BIT` | `1` | 4-bit NF4 quantization (needs `bitsandbytes`). |
| `AI_LLM_DTYPE` | `bfloat16` | Compute dtype. |
| `AI_LLM_DEVICE_MAP` | `auto` | `transformers` device map. |
| `AI_LLM_MAX_NEW_TOKENS` | `768` | Generation cap. |
| `AI_LLM_PARSER_TEMPERATURE` | `0.0` | Parser = greedy (reproducible contracts). |
| `AI_LLM_REPORT_TEMPERATURE` | `0.6` | Default warmth for the immersive report. |

## Pre-warming

`warm()` eager-loads the model (same pattern as `registry.prewarm_defaults`).
Wire it into the server lifespan, **opt-in** via `AI_LLM_PREWARM` (default off)
so existing boots are unaffected until the model is confirmed on serverB. See
the decision doc §8.

## Not done yet (scaffold)

- **LLM job "skills" are placeholder stubs (deferred — owner: Lucas).** The
  per-job instruction sets sent with each call (`prompts/parser_system.py`,
  `prompts/report_system.py`) are placeholders; the real skills are authored
  later. Wiring may proceed against the stubs and swap them in when ready.
- `bitsandbytes` / `lm-format-enforcer` are **not** in `requirements.txt` yet
  (deferred so this scaffold doesn't pull heavy deps into CI / the serverB
  sync) — add them when wiring the service in.
- `complete_json()` uses a tolerant JSON-extraction fallback. Replace with
  **grammar-constrained decoding** for a hard validity guarantee (see the
  `complete_json` TODO and decision doc §6).
- Faithfulness post-validation for the report writer (reject prose that
  introduces species/numbers absent from the fused JSON) is not implemented.
