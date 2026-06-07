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
├── gate.py              # deterministic validity-gate findings (parser)
├── parser.py            # parse_prompt(): NL prompt -> parse-result contract
├── report.py            # write_report(): fused JSON -> prose (2 registers)
├── faithfulness.py      # guard: prose names no unobserved species
├── skills/              # per-job instruction files (system messages)
│   ├── __init__.py      #   load_skill() loader + cache
│   ├── parser.md        #   parser skill (placeholder — author later)
│   ├── report_analytical.md
│   └── report_immersive.md
└── README.md            # this file
```

## Usage

```python
from llm import parse_prompt, write_report

# Prompt parser — NL prompt -> parse-result contract (JSON, grammar-constrained)
contract = parse_prompt("a misty autumn dawn with light rain")

# Report writer — fused analysis JSON -> prose in a register
narrative = write_report(fused_report_json, register="immersive")
# -> {"register": "immersive", "text": "...", "faithful": True, "violations": []}
```

The service is reached over HTTP via the FastAPI routes `POST /generation/parse`
and `POST /analysis/narrative` (and inline on `/analyze` when
`AI_LLM_NARRATIVE` is on). Skills load from `skills/*.md` as the system message;
the job data is the user message (plan §2.1).

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

## Not done yet

- **Skill content is placeholder (deferred — owner: Lucas).** `skills/*.md` are
  working placeholders; the authored skills (instructions + few-shot) swap in
  later with no code change.
- **serverB-only steps:** install deps in `acoustic_ai/.venv`, pre-download the
  HF model, validate 16 GB VRAM headroom, then flip `AI_LLM_PREWARM=on`. See
  the implementation plan §5–§8.
- Phenology table for the parser's fauna gate is a separate prerequisite
  (plan §10).
- `complete_json()` uses a tolerant JSON-extraction fallback. Replace with
  **grammar-constrained decoding** for a hard validity guarantee (see the
  `complete_json` TODO and decision doc §6).
- Faithfulness post-validation for the report writer (reject prose that
  introduces species/numbers absent from the fused JSON) is not implemented.
