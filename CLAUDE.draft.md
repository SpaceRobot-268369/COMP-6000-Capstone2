# COMP-6000 Capstone 2 — Speculative Soundscape Generation

Research prototype: ecoacoustic recordings + environmental data → AI-generated speculative soundscapes.

**Three modes:** Analysis, Generation, Transformation.
**Modeling approach:** layered composition (ambient bed + weather + events + mix), not a single generated waveform.

---

## System at a glance

| # | Component | Tech | Status |
|---|-----------|------|--------|
| 1 | Frontend | React + Vite (`frontend/`) | UI scaffold |
| 2 | Backend | Express.js + PostgreSQL (`backend/`) | Auth endpoints |
| 3 | AI module | Python / PyTorch (`acoustic_ai/`) | Layer A smoke ✓, Layer C smoke ✓ |
| 4 | Metadata DB (optional) | PostgreSQL | Not started |

### AI modules

| Module | Role | Status |
|---|---|---|
| A — Ambient | AudioLDM2 LoRA (base: `cvssp/audioldm2`) for ambient bed | Smoke 1 (spring night) ✓, Smoke 2 (insects) ✓ |
| B — Weather | Curated wind/rain assets + parameter mixing | Placeholder |
| C — Events | AudioGen LoRA per species (base: `facebook/audiogen-medium`) | Smoke 1 (boobook) ✓ |
| D — Mixer | Combine A+B+C → WAV + explanation JSON | Placeholder |
| E — Analysis | Ambient similarity + weather + event detectors | Partial (A working) |

**Generative model strategy:** Layers A and C use frozen large base models + LoRA adapters for the MVP. Distillation to in-house models is a future option, gated on (1) LoRA path proven across species/contexts, (2) demonstrated latency/VRAM bottleneck, (3) team capacity. See [distillation strategy](.claude/context/ai/distillation_strategy.md).

---

## Critical conventions

### Storage rule

All Claude-related context files live under `.claude/` — never at the project root.

| Type | Location |
|------|----------|
| Architecture / design docs | `.claude/context/<topic>/` |
| Runbooks (smoke tests, training workflows) | `.claude/context/ai/runbooks/` |
| Branch-scoped dev logs (ephemeral) | `.claude/context/dev/<branch-slug>/` |
| Settings | `.claude/settings.local.json` |

CLAUDE.md is the **hub**, not the manual. If a section grows past ~5 lines or is linked from elsewhere, it moves to `.claude/context/` and CLAUDE.md links to it.

Branch-scoped logs in `.claude/context/dev/<branch-slug>/` must be deleted in the merge PR, **or** their durable content promoted into a permanent doc first.

### Python environment

Always use `acoustic_ai/.venv` for AI training and inference (`./acoustic_ai/.venv/bin/python`, `./acoustic_ai/.venv/bin/accelerate`, etc.). Do **not** use system / Homebrew `python3`, `pip`, `accelerate`, or `uvicorn` — they load incompatible torch/torchaudio builds.

### Layer A dev-generation contract

The smoke-test LoRAs are trained on tiny datasets, so generation parameters are fixed server-side:

- Frontend exposes **only** a non-negative integer `seed` (range `0`–`2147483647`).
- Express backend forwards **only** `{ seed }`.
- FastAPI AI server owns the prompt, checkpoint, guidance, step count, audio length, RMS, and high-pass.
- Server returns all parameters in response metadata for debugging.

Seed is **not** temperature — it initializes the diffusion noise. Same seed + same params + same code path = same audio.

### Git branch naming

```
<type>/<author>/<short-description>
```

Types: `feat`, `fix`, `data`, `model`, `infra`, `refactor`, `docs`, `exp`.
Example: `model/lucas/layer-c-event-attemp-1`.

Commit subjects use imperative mood, ≤72 chars, no issue numbers in the subject.

### Pre-commit file audit

Mandatory before every commit:
1. Run `git status`.
2. If unintended files appear (binaries, generated outputs, credentials, OS files), don't commit.
3. Add them to `.gitignore` (and `git rm --cached <path>` if already tracked).
4. Verify `git status` is clean of unintended files.

Large binaries (audio, checkpoints, `.npy`) **never** go in git — use DVC.

---

## Running services

| Service | How | URL |
|---------|-----|-----|
| Frontend | Docker | `http://localhost:5173` |
| Backend | Docker | `http://localhost:4000` |
| PostgreSQL | Docker | `localhost:5432` |
| AI server | **Native only** (GPU/MPS) | `http://localhost:8000` |

Docker cannot access macOS MPS — the AI server **must** run natively.

Commands, env vars, and ports: [.claude/context/setup/services.md](.claude/context/setup/services.md)

---

## Where to find everything

### AI

| Topic | File |
|---|---|
| Module architecture (A–E, code layout) | [.claude/context/ai/architecture.md](.claude/context/ai/architecture.md) |
| Generation & analysis pipeline design | [.claude/context/ai/pipeline_design.md](.claude/context/ai/pipeline_design.md) |
| Distillation strategy (future product path) | [.claude/context/ai/distillation_strategy.md](.claude/context/ai/distillation_strategy.md) |
| MVP decision log | [.claude/context/ai/logs/mvp_decision_log.md](.claude/context/ai/logs/mvp_decision_log.md) |
| AudioLDM2 transition log | [.claude/context/ai/logs/audioldm2_transition_log.md](.claude/context/ai/logs/audioldm2_transition_log.md) |

### Runbooks (smoke-test workflows)

| Workflow | File |
|---|---|
| Layer A smoke 1 — spring night ambient (AudioLDM2 LoRA) | [.claude/context/ai/runbooks/layer_a_smoke_1_spring_night.md](.claude/context/ai/runbooks/layer_a_smoke_1_spring_night.md) |
| Layer A smoke 2 — insect/cicada ambient (AudioLDM2 LoRA) | [.claude/context/ai/runbooks/layer_a_smoke_2_insects.md](.claude/context/ai/runbooks/layer_a_smoke_2_insects.md) |
| Layer C smoke 1 — bird events (AudioGen LoRA) | [.claude/context/ai/runbooks/layer_c_smoke_1_birds.md](.claude/context/ai/runbooks/layer_c_smoke_1_birds.md) |

### Data

| Topic | File |
|---|---|
| Data alignment & env features | [.claude/context/data/data_reference.md](.claude/context/data/data_reference.md) |
| S3 bucket layout & DVC remote | [.claude/context/data/s3_bucket_layout.md](.claude/context/data/s3_bucket_layout.md) |
| DVC workflow (git+dvc, hooks, pipeline) | [.claude/context/data/dvc_workflow.md](.claude/context/data/dvc_workflow.md) |
| Generation quality analysis | [.claude/context/data/logs/generation_quality_analysis.md](.claude/context/data/logs/generation_quality_analysis.md) |
| MVP dataset (site 257) details | [.claude/skills/sample_mvp_dataset.md](.claude/skills/sample_mvp_dataset.md) |

### Operations

| Topic | File |
|---|---|
| Known data issues (incl. 12 unrecoverable clips) | [.claude/context/issues/known_issues.md](.claude/context/issues/known_issues.md) |
| Analysis test cases | [.claude/context/testing/analysis_test_cases.md](.claude/context/testing/analysis_test_cases.md) |
| Layer verification & handoff formats | [.claude/context/testing/layer_verification_formats.md](.claude/context/testing/layer_verification_formats.md) |
| Workflow diagrams | [.claude/context/diagrams/workflow_diagrams.md](.claude/context/diagrams/workflow_diagrams.md) |
| Git workflow (full) | [.claude/context/git_workflow.md](.claude/context/git_workflow.md) |
| Hardware requirements | [.claude/context/setup/hardware.md](.claude/context/setup/hardware.md) |

---

## Backend API

Current:
- `GET  /api/health` — DB connectivity check
- `POST /api/register` — user registration
- `POST /api/login` — user login

Planned (Stage 3):
- `POST /api/analysis` — submit audio for soundscape analysis
- `POST /api/generation` — generate soundscape from environmental params
- `POST /api/transformation` — transform audio with new environmental conditions

---

## Notes

- Avoid excessive filtering/denoising — anthropogenic noise is authentic soundscape data.
- Data representations should be **learned** (spectrogram → encoder → embedding), not hand-crafted parameters.
- Prototype stage may use pre-trained models and reduced datasets.
- For exploratory questions, prefer linking to the canonical doc over re-explaining inline.
