# Prerequisites — Concepts to know before touching the AI module

This doc is the **conceptual on-ramp** for the `acoustic_ai/` module. It assumes
no audio-ML background and brings a reader up to the point where the per-layer
docs ([architecture.md](architecture.md), [pipeline_design.md](pipeline_design.md))
read as engineering decisions rather than magic.

Five buckets, ordered from "most universal" to "most project-specific":

1. [Audio fundamentals](#1-audio-fundamentals)
2. [ML model anatomy](#2-ml-model-anatomy)
3. [Generative model families](#3-generative-model-families)
4. [Fine-tuning and transfer](#4-fine-tuning-and-transfer)
5. [Audio analysis and the pre-trained ecosystem](#5-audio-analysis-and-the-pre-trained-ecosystem)
6. [Operational gotchas](#6-operational-gotchas)

---

## 1. Audio fundamentals

### Waveform → spectrogram → mel spectrogram

Raw audio is just a long list of numbers (samples). Models almost never operate
on raw samples directly — they convert to a **spectrogram** (a 2D "picture" of
how much energy is at each frequency over time), and usually a **mel
spectrogram** (with frequency bins warped to match human/animal hearing).

```
   Waveform                STFT             Mel spectrogram
   (1D, time)              ↓                (2D, time × mel-bin)

   ∿∿∿∿∿∿∿∿∿∿∿∿  ──────►  ┌──────────┐  ──► ┌──────────────────┐
   22,050 numbers          │ window + │      │ ░▒▓█▓▒░ ▓█▓▒░    │
   per second              │  FFT per │      │ ▒▓█▓▒░  █▓▒░     │
                           │   frame  │      │ ▓█▓▒░   ▓▒░      │
                           └──────────┘      └──────────────────┘
                                              ~80 bins × ~86 frames/s
                                              ≈ 100× smaller than waveform
```

### Sample rates in this project

| Rate | Where | Why |
|---|---|---|
| 16,000 Hz | AudioGen (Layer C) input/output | base model trained at 16 kHz; native operating range |
| 22,050 Hz | AudioLDM2 (Layer A), VAE, mixer, final output | base model trained at 22.05 kHz; project standard |
| 44,100 Hz | source field recordings | CD-quality; downsampled on ingest |
| 48,000 Hz | some A2O archive material | downsampled on ingest |

**Sample-rate boundaries are silent bugs.** AudioGen (16 kHz) → mixer (22.05 kHz)
*must* resample. If you forget, the audio still plays — just pitch-shifted and
aliased. Always resample explicitly at layer boundaries.

### Loudness vocabulary

| Term | Meaning | When you see it |
|---|---|---|
| **RMS** | root-mean-square — average loudness over a window | layer mixing, ambient gating ("low-RMS frames"), volume matching |
| **dB** | logarithmic scale (10× louder ≈ +10 dB) | "−12 dB event attenuation," "−3 dB headroom" |
| **dBFS** | dB relative to digital full-scale; 0 dBFS = clipping | output level checks |
| **Peak normalisation** | scale so max sample = target | crude; doesn't match *perceived* loudness |
| **RMS normalisation** | scale so RMS = target | what the mixer uses to balance layers |
| **Clipping** | samples exceed ±1.0, get truncated → harsh distortion | a real bug in the mixer if you see it |

### Common artifacts (what's a bug vs what's "the model")

| Artifact | Sounds like | Cause |
|---|---|---|
| Clipping | crackle/buzz on peaks | mixer gain staging; fix in code |
| Aliasing | metallic high-freq whine | missing resample at SR boundary; fix in code |
| Vocoder warble | watery / underwater quality | HiFi-GAN limitation; usually unfixable without retraining |
| EnCodec shimmer | faint metallic ringing in event clips | tokenizer reconstruction limit; tolerated |
| Pulsing / amplitude modulation | rhythmic "breathing" | over-amplified quiet input (e.g. RMS-005 checkpoint); checkpoint problem |
| Phase smearing | hollow / flange-y when layering | naive overlap-add; mixer should crossfade properly |

---

## 2. ML model anatomy

### The universal encoder/decoder shape

```
        ┌──────────┐     ┌──────────┐     ┌──────────┐
  IN ──►│ Encoder  │────►│embedding │────►│ Decoder  │──► OUT
        └──────────┘     │ (vector) │     └──────────┘
                         └──────────┘
        "understand"     compact          "produce"
        the input        meaning          the output
```

- **Encoder** — input → embedding. Throws away surface detail, keeps meaning.
- **Embedding** — vector in the middle. Similar inputs land at nearby vectors.
- **Decoder** — embedding → output. Reverses the squeeze.
- **Vocoder** — a specific kind of decoder: spectrogram (or tokens) → waveform.

Full enumeration of every encoder/decoder/vocoder in this project, with reuse
notes for analysis, is in the section below.

### Latent spaces

A **latent space** is just "the space the embeddings live in." A 256-dimensional
latent = a 256-number vector. The useful property:

```
   semantically similar inputs       →    nearby vectors
   semantically different inputs     →    far-apart vectors
   "meaningful directions" in space  →    e.g. day↔night axis,
                                          quiet↔loud axis
```

This is why k-NN search over embeddings works as "retrieval": find the closest
training examples to a query, treat their metadata as the answer. Layer A's
runtime retrieval and E-A's similarity analysis both run on this principle.

### Joint embedding spaces (CLAP)

A normal encoder maps one modality (audio) to vectors. A **joint** encoder maps
*two* modalities to the *same* space, trained so that matching pairs end up
close:

```
              shared embedding space
              ┌─────────────────────────┐
   "owl ────► │   • text vector         │
    hooting"  │            ≈            │   ← close = matching
              │   • audio vector        │
   🦉 ──────► │                         │
              └─────────────────────────┘
```

Why this matters: you can **compare across modalities for free**. Score an
audio clip against a list of text prompts → instant zero-shot classifier.
That's exactly what powers E-B and E-C fallbacks in analysis mode.

### Discrete vs continuous audio representations

| Representation | Type | Used by | Why |
|---|---|---|---|
| Raw waveform | continuous, very long | nothing trains on it directly | too high-dimensional |
| Mel spectrogram | continuous, 2D | AudioLDM2, VAE, HiFi-GAN | smooth, compact, diffusion-friendly |
| VAE latent | continuous, low-dim | inside AudioLDM2 diffusion | even more compact than mel |
| EnCodec tokens | **discrete**, short sequence | AudioGen AR transformer | autoregressive transformers need a finite vocabulary, like text tokens |

Autoregressive models (predict-one-then-next) **need** discrete tokens.
Diffusion models can live in continuous space. This is why AudioLDM2 and
AudioGen have such different internals despite producing the same kind of
output.

### Every encoder/decoder/vocoder in this project

| # | Name | Type | Lives in | Input → Output | Reused for analysis? |
|---|---|---|---|---|---|
| 1 | T5 text encoder | encoder | AudioLDM2 + AudioGen | text → embedding | ❌ text-only |
| 2 | CLAP text encoder | encoder | AudioLDM2 | text → embedding (joint space) | ✅ pairs with #3 for zero-shot |
| 3 | **CLAP audio encoder** | encoder | AudioLDM2 | audio → embedding (joint space) | ✅✅ **primary analysis encoder** |
| 4 | AudioLDM2 diffusion U-Net | decoder | Layer A base | embedding + noise → latent spec | ❌ generator only; LoRA fine-tunes this |
| 5 | VAE encoder (AudioLDM2's) | encoder | AudioLDM2 | mel spec → latent | ❌ internal only |
| 6 | VAE decoder (AudioLDM2's) | decoder | AudioLDM2 | latent → mel spec | ❌ generator only |
| 7 | HiFi-GAN vocoder (AudioLDM2's) | decoder (vocoder) | AudioLDM2 | mel spec → waveform | ❌ generator only |
| 8 | AudioGen AR transformer | decoder | Layer C base | embedding → token sequence | ❌ generator only; LoRA fine-tunes this |
| 9 | EnCodec encoder | encoder | AudioGen | audio → discrete tokens | ✅ optional event-classifier features |
| 10 | EnCodec decoder | decoder (vocoder) | AudioGen | tokens → waveform | ❌ generator only |
| 11 | Project VAE encoder | encoder | `model/candidates/lucas/vae-site257-30epoch/` | mel spec → 256-dim latent | ✅ legacy E-A option |
| 12 | Project VAE decoder | decoder | same | latent → mel spec | ❌ transformation mode only |
| 13 | Project HiFi-GAN vocoder | decoder (vocoder) | `model/candidates/lucas/vocoder-hifigan-site257/` | mel spec → waveform | ❌ generator only |

**Pattern:** every generator has *at least one encoder and one decoder*, often
chained (AudioLDM2 = U-Net → VAE decoder → HiFi-GAN, three decoder stages).
LoRA always fine-tunes the **decoder side** — which is why fine-tuned
checkpoints don't transfer to analysis.

---

## 3. Generative model families

Two families show up in this project. Different math, different knobs.

### Diffusion (AudioLDM2 — Layer A)

**Idea:** start from pure random noise, iteratively denoise it toward something
that matches the text prompt. Each step nudges the noise a little closer to
"valid audio that means what the prompt says."

```
   step 0          step 25          step 50          step 100
   ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐
   │█▓▒░ │   ──►   │░▒▓░ │   ──►   │ ░▓▒ │   ──►   │ ▒▓░ │
   │▒░▓█ │         │▓▒░▒ │         │▒▓░  │         │▓░▒  │
   │░▓▒█ │         │░▓▒░ │         │ ▒░  │         │ ▒░  │
   └─────┘         └─────┘         └─────┘         └─────┘
   pure noise      noisy           clearer         clean spectrogram
                                                   of "spring night"
```

**Knobs:**

| Knob | What it controls | Project default | Notes |
|---|---|---|---|
| `num_inference_steps` | how many denoising iterations | 100 | fewer = faster but blurrier; LCM distillation reduces this |
| `guidance_scale` (CFG) | how strictly to follow the prompt | 2.0 (Layer A smoke) | higher = on-prompt but less natural; too high = artifacts |
| `seed` | which random noise to start from | dev-exposed | **same seed + same params + same code = same output** |
| `scheduler` | which denoising algorithm | DDIM / DPM++ | mostly a "leave it alone" choice |

### Autoregressive transformer (AudioGen — Layer C)

**Idea:** predict one audio token at a time, like a language model predicts
words. Each token depends on all previous tokens plus the text prompt.

```
   prompt: "boobook call at night"
                │
                ▼
        ┌──────────────┐
        │   AR model   │ ──► token₁
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │   AR model   │ ──► token₂      (sees prompt + token₁)
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │   AR model   │ ──► token₃      (sees prompt + token₁ + token₂)
        └──────┬───────┘
               │
               ▼            … repeat ~150–500 times …
               ▼
        full token sequence ──► EnCodec decoder ──► waveform
```

**Knobs:**

| Knob | What it controls | Project default | Notes |
|---|---|---|---|
| `top_k` | sample only from k most likely next tokens | 250 | lower = safer / repetitive, higher = more variety / risk |
| `temperature` | how peaked the next-token distribution is | 1.0 | <1 = conservative, >1 = wild; **not the same as seed** |
| `cfg_coef` | classifier-free guidance strength | 3.0 | analogous to diffusion CFG but separate hyperparameter |
| `duration` | seconds to generate | 3–5 s | autoregressive cost is linear in duration |
| `seed` | RNG state for sampling | dev-exposed | reproducibility |

### Diffusion vs AR — when each shines

| Property | Diffusion (Layer A) | AR transformer (Layer C) |
|---|---|---|
| Best at | smooth, stationary textures (ambient, drones) | sharp transients, onsets (calls, hits) |
| Internal repr | continuous latents | discrete tokens (EnCodec) |
| Inference cost | many parallel steps; benefits from step distillation | one token at a time; linear in duration |
| Failure mode | blurry / smeared transients | metallic warble, repetition loops |
| Why we chose it for its layer | ambient *is* stationary; sharp onsets would be wrong | events *are* transients; AudioSet animal priors |

### VAEs and GANs (just enough)

- **VAE** = encoder + sampling step + decoder, trained to reconstruct input
  while keeping the latent space smooth. **Smooth latent space** is what
  makes nearest-neighbour interpolation possible. Cost: VAE-only audio output
  sounds blurry, which is why we pair the VAE with a **GAN-trained vocoder**
  (HiFi-GAN) that adds the missing high-frequency detail.
- **GAN** = generator vs discriminator, adversarial training. HiFi-GAN is the
  GAN-trained vocoder; it's why AudioLDM2 output sounds sharp rather than
  watery.

---

## 4. Fine-tuning and transfer

### Why we use frozen base + LoRA (not full fine-tune)

```
   Full fine-tune                     LoRA fine-tune
   ─────────────                      ─────────────
                                      
   ┌─────────────────┐                ┌─────────────────┐
   │  1.5B params    │  ALL update    │  1.5B params    │  FROZEN
   │  base model     │  every step    │  base model     │  (no gradients)
   └─────────────────┘                └─────────────────┘
                                              │
   needs:                                     ├── small "adapter"
   • huge dataset                             │   (~1–5M params)
   • huge VRAM                                │   trained on top
   • days/weeks                               │
   • risks catastrophic                       └── trains in hours
     forgetting                                    on 40–80 clips
                                                   no forgetting risk
```

LoRA adds two small matrices to the attention layers of the base model. Only
those matrices update during training. The base model never moves.

### LoRA knobs

| Knob | Meaning | Project default |
|---|---|---|
| `r` (rank) | size of the low-rank adapter; capacity | 8 |
| `alpha` | scaling factor on the adapter output | 16 |
| `target_modules` | which layers get adapters | `q_proj`, `v_proj` (attention) |
| `dropout` | regularisation inside adapter | usually 0 for our small data |

### Overfitting on tiny datasets

40–80 audited clips per Layer C species is a *very* small dataset. The
defenses you'll see throughout the docs:

| Defense | Why |
|---|---|
| Per-recording cap (≤3 snippets per `audio_recording_id`) | prevents memorising one individual's voice |
| Manual audit of 150 candidates → 40–80 keepers | rejects multi-species overlap, wind contamination |
| Diel-bin filtering | matches training distribution to inference distribution |
| LoRA over full fine-tune | small adapter has limited capacity to memorise |
| 10–15 epochs (not 100) | early stopping by hand |
| Seed audits 42–51 | sample 10 seeds, cherry-pick — assumes some seeds will fail |

### Seed vs temperature vs CFG — three different knobs

| Knob | What it changes | Reproducible? |
|---|---|---|
| **seed** | which random noise draw to use as the starting point | ✅ same seed + same code = same output |
| **temperature** | how peaked the sampling distribution is at each step (AR only) | ❌ even at temp 1.0, different seeds give different outputs |
| **CFG / guidance_scale** | how much the prompt pulls the output away from "unconditional generation" | depends on seed |

The dev frontend exposes **only seed**, not the other two. Reason: the LoRA
checkpoints are trained on tiny data, and exposing temperature / CFG to users
would surface failure modes faster than the LoRA can be improved. Server owns
those choices.

---

## 5. Audio analysis and the pre-trained ecosystem

### What already exists, and what it's good at

| Model | Type | Strong at | Blind spots |
|---|---|---|---|
| **BirdNET** | bird classifier (~6k species) | Australian + global birds | non-birds (frogs, insects, mammals), unusual angles, very distant calls |
| **Google Perch** | bioacoustic embedding | site / biome similarity; works as feature extractor for custom heads | not a classifier out of the box — need a head on top |
| **PANNs (CNN14)** | AudioSet tagger (527 classes) | general audio incl. wind, rain, thunder, vehicles | not species-level for wildlife |
| **YAMNet** | AudioSet tagger (521 classes) | same label space as PANNs, faster | weaker accuracy than PANNs CNN14 |
| **LAION-CLAP** | audio↔text joint encoder | open-vocabulary zero-shot ("light wind in trees") | not specialised for any domain; weaker than dedicated detectors on their home turf |

### k-NN over embeddings — the "model" behind retrieval

The Layer A runtime retrieval and Layer E-A ambient similarity are both this
exact pattern:

```
   Query clip ──► [Encoder] ──► query vector
                                     │
                                     ▼
                    cosine similarity vs every vector in index
                                     │
                                     ▼
                          top-k closest neighbours
                                     │
                                     ▼
                  use their metadata as the answer
                  (env conditions, segment IDs)
```

**Cosine vs L2:** cosine measures direction (similarity), L2 measures
absolute distance. CLAP and most modern embeddings are trained to be
compared with **cosine**. Always normalise vectors first.

**Hard-filter then soft-rank:** for Layer A we first restrict to candidates
matching the requested `diel_bin` + `season` (categorical mismatch sounds
wrong regardless of vector closeness), *then* cosine-rank the survivors on
hour/month encoding. Pure cosine over everything mixes categories.

### Why source separation is hard for natural soundscapes

| Music separation (Demucs, Spleeter) | Natural soundscape separation |
|---|---|
| Sources are distinct (drums vs vocals) | Sources overlap diffusely |
| Massive labelled training data (stems) | No source-isolated ground truth available |
| Stable, repeating structure | Open-class events; non-stationary |
| Works well | No working pre-trained model exists for our domain |

This is why analysis mode uses **direct detection on the mixture**, not
"decompose then detect." See [pipeline_design.md § Design principle](pipeline_design.md#design-principle-direct-detection-no-decomposer).

### Acoustic indices — cheap hand-crafted features

| Index | What it measures | Useful for |
|---|---|---|
| **ACI** (acoustic complexity) | amplitude variation across freq bins over time | biophony presence; high in dawn chorus |
| **Spectral entropy** | how uniformly energy is spread across frequencies | distinguishes tonal calls from broadband noise |
| **Spectral centroid** | "center of mass" of the spectrum | brightness; wind tends low, rain tends high |
| **Spectral flatness** | how noise-like vs tonal | rain is flat, bird calls are tonal |
| **RMS** | loudness | basic energy gate |

These run in CPU-seconds, give you explainability, and serve as a sanity
check on ML predictions. Not primary detectors.

---

## 6. Operational gotchas

These aren't concepts — they're the things that quietly break the system if
you don't internalise them.

### Sample-rate boundaries

Every layer in the mixer must resample to the project standard (22,050 Hz):

```
   Layer A (AudioLDM2)  22.05 kHz  ─┐
   Layer B (assets)     22.05 kHz  ─┼──► mixer @ 22.05 kHz  ──► output
   Layer C (AudioGen)   16 kHz  ────┘
                                ▲
                                └── must resample 16 → 22.05
                                    before overlay
```

Forgetting a resample produces audio that *plays* but is pitch-shifted /
aliased. Silent bug, hard to catch unless you actively listen.

### Python environment matrix

| Stack | venv | Reason |
|---|---|---|
| AudioLDM2, VAE, FastAPI server | `acoustic_ai/.venv` | diffusers + torch 2.x build that works on MPS |
| AudioGen, audiocraft | `acoustic_ai/.venv-audiogen` | audiocraft's torchaudio requirement conflicts with diffusers' |
| DVC + S3 deps | user-site (`pip3 install --user`) | git hooks call `dvc` without venv activation |

Mixing these — `pip install` outside the right venv, or using system
`python3` for AI code — quietly loads the wrong torch build and breaks MPS
kernels with confusing error messages.

### "Annotations are not ground truth"

A2O annotations have **sparse coverage**. *Absence of annotation ≠ absence
of event.* This is why:

- Layer A ambient cleaning uses an **anomaly gate** (3·MAD over per-clip
  baseline), not "annotated frames = bad."
- BirdNET and A2O are used as **post-hoc audits**, not as gates.
- Layer C smoke tests **manually audit** the 150 → 40–80 keepers; never
  trust the annotation tag alone.

### Reproducibility ≠ determinism on MPS

PyTorch on MPS (Apple GPU) is **not** bit-exact between runs even with seed
fixed. Outputs are *perceptually* identical (same prompt + seed + checkpoint
on the same machine sounds the same), but the underlying floats can differ
slightly. CUDA fp16 has similar quirks. Don't write tests that assert
byte-exact output.

---

## Suggested reading order

If you are new to this project and want a fast path to productive:

1. This doc (you are here).
2. [conventions.md](../conventions.md) — repo structure, naming, artifact tiers.
3. [architecture.md](architecture.md) — the five-layer overview.
4. [pipeline_design.md](pipeline_design.md) — per-layer design decisions.
5. One smoke-test runbook (e.g. [layer_a_smoke_1_spring_night.md](runbooks/layer_a_smoke_1_spring_night.md))
   — see all the concepts above applied end-to-end on a concrete attempt.
6. [distillation_strategy.md](distillation_strategy.md) — future direction; safe
   to skip until you've shipped at least one layer.
