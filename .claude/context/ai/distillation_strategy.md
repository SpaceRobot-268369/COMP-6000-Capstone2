# Generative Model Strategy — Base Model + LoRA vs. Distilled Own Model

## Current approach (MVP and smoke test)

Layers A and C use frozen large base models with LoRA adapters:

| Layer | Base model | Adapter |
|-------|-----------|---------|
| A — Ambient | `cvssp/audioldm2` (~1.5B params, latent diffusion) | LoRA fine-tuned on ~50 Bowra clips |
| C — Events | `facebook/audiogen-medium` (~1.5B params, autoregressive transformer + EnCodec) | Per-species LoRA (40–80 clips each) |

LoRA adds ~0.1–0.5% extra parameters on top of frozen base weights. The full base model must be loaded at inference time.

---

## Future product-level consideration — distilled own model

Knowledge distillation trains a smaller **student** model to mimic the **teacher** (base model) outputs, with the goal of reducing VRAM footprint and inference latency.

Relevant distillation approaches per layer:

- **Layer A (diffusion):** progressive distillation, consistency distillation (LCM), or adversarial distillation (ADD) — compresses 100+ DDIM steps → 1–4 steps, and optionally shrinks the U-Net backbone.
- **Layer C (autoregressive transformer):** sequence-level KD training the student on teacher token probability distributions.

---

## Conditions for pursuing distillation

Distillation becomes worth serious engineering investment only when **all three** hold:

1. The base model + LoRA path is proven to produce high-quality results across multiple species and seasonal contexts (not just smoke tests).
2. Deployment latency or VRAM cost is a demonstrated bottleneck for real users.
3. The team has sufficient ecoacoustic data and ML research capacity to run distillation experiments without stalling product work.

As of 2026-05, condition 1 is partially met (Layer A smoke passed, Layer C pending). Conditions 2 and 3 are not yet met.

---

## Risk and trade-off analysis

### Layer A — distilling AudioLDM2

| Factor | Detail |
|--------|--------|
| **Architecture complexity** | AudioLDM2 is a multi-component stack (CLAP encoder, FLAN-T5, VAE, U-Net, HiFi-GAN vocoder). Each component must be distilled or the pipeline distilled jointly. High engineering effort. |
| **Technique maturity** | Diffusion distillation (LCM, ADD, progressive) exists but is active research. Applying it on top of an ecoacoustic LoRA is not off-the-shelf. |
| **Data requirements** | Consistency/progressive distillation requires thousands of teacher-generated trajectory samples as student supervision — far beyond the 50-clip Bowra dataset. Teacher must first generate large synthetic training data. |
| **Quality floor risk** | Distilled models trade quality for speed. Ecoacoustic ambient texture is subtle; a student optimised to match teacher outputs on average may lose the fine-grained stationarity that makes ambient beds believable. |
| **Inference gain if successful** | LCM-style step reduction: ~25× speedup at sampling (100 steps → 4). VRAM savings require also shrinking the U-Net, which is a separate distillation objective. |
| **Licensing** | `cvssp/audioldm2` is CC BY 4.0. Distillation permitted. No legal blocker. |

### Layer C — distilling AudioGen

| Factor | Detail |
|--------|--------|
| **Architecture complexity** | Lower than Layer A. Single transformer + EnCodec codec. EnCodec can be reused; only the transformer backbone needs distillation. More tractable. |
| **Technique maturity** | Sequence-level KD for transformers is well-established. Practical path exists. |
| **Data requirements** | Still needs large teacher-generated supervision corpora. Per-species datasets of 40–80 clips cannot train a student from near-scratch. |
| **Per-species flexibility lost** | LoRA lets you add a species with 40–80 clips and hours of GPU time. A distilled student would need full retraining or still needs adapters on top — partially defeating the purpose. |
| **EnCodec dependency** | Even a distilled AudioGen student still depends on Meta's EnCodec for token ↔ audio conversion. Fully "own model" requires replacing EnCodec too — a separate major effort. |
| **Licensing** | `facebook/audiogen-medium` is MIT. Distillation permitted. |

---

## Head-to-head comparison

| Dimension | Base model + LoRA (current) | Distilled own model (future) |
|---|---|---|
| **Time to implement** | Done (Layer A smoke passed; Layer C in progress) | 6–18 months of research engineering on top of product work |
| **Data needed per layer** | 40–80 clips per LoRA (proven at smoke scale) | Thousands of teacher-generated samples + original domain data |
| **Quality ceiling** | Teacher model quality minus LoRA approximation error | Below teacher quality by design — trades quality for speed/size |
| **Inference speed** | Slow (100 DDIM steps for A; autoregressive for C) | Faster if distillation succeeds (4–8 steps for A; smaller transformer for C) |
| **VRAM at inference** | Full base model + LoRA (~6–8 GB each) | Smaller student (~1–3 GB target, pending successful distillation) |
| **Adding a new species (Layer C)** | Train one small LoRA, ~2 hrs, ~50 MB | Full student retrain or still needs adapters — no clean win |
| **Maintenance burden** | Low — upstream maintainers patch base models | High — all bugs and regressions are the team's to debug |
| **IP / deployment independence** | Medium — depends on upstream licenses (currently unblocked) | High — own weights, no upstream license dependency |
| **Risk of failure** | Low (validated path) | High — domain is narrow; student may not retain ecoacoustic specificity |
| **Right for MVP / research prototype** | Yes | No |
| **Right for production at scale** | Marginal (latency/VRAM concern at scale) | Potentially, but only after the base model path proves the domain works end-to-end |

---

## Recommended intermediate step

If inference latency becomes a real bottleneck before full distillation is feasible, prefer **LCM/consistency distillation on Layer A only** (step-count reduction without architecture shrink). This is the single highest-leverage distillation target:

- Layer A uses 100 DDIM steps today — LCM can cut to 4, a ~25× speedup.
- Layer C autoregressive generation is already fast enough for the event durations (1–10 s clips).
- Layer C per-species LoRAs should remain as-is indefinitely; retraining a generalist student per species addition is not cost-effective.

Apply full distillation only when conditions 1–3 above are all clearly met.
