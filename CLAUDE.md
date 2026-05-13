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
| C — Events | AudioGen LoRA per species (base: `facebook/audiogen-medium`, 16 kHz native) | Smoke 1 (boobook) ✓ |
| D — Mixer | Combine A+B+C → WAV + explanation JSON | Placeholder |
| E — Analysis | Ambient similarity + weather + event detectors | Partial (A working) |

**Generative model strategy:** Layers A and C use frozen large base models + LoRA adapters for the MVP. Migration to in-house distilled models is a future option, gated on (1) LoRA path proven across species/contexts, (2) demonstrated latency/VRAM bottleneck, (3) team capacity. See [distillation strategy](.claude/context/ai/distillation_strategy.md).

---

## Critical conventions

### Storage rule

All Claude-loadable context lives under `.claude/` — never at the project root.

| Type | Location |
|------|----------|
| Architecture / design docs | `.claude/context/<topic>/` |
| Runbooks (smoke tests, training workflows) | `.claude/context/ai/runbooks/` |
| Dev workflow specs (DVC, S3, git, model README standard) | `.claude/context/dev_specifications/` |
| Setup (services, hardware) | `.claude/context/setup/` |
| Branch-scoped dev logs (ephemeral) | `.claude/context/dev/<branch-slug>/` |
| Settings | `.claude/settings.local.json` |

CLAUDE.md is a **hub**, not a manual. If a section grows past ~5 lines or is referenced from elsewhere, it moves into `.claude/context/` and CLAUDE.md links to it.

Branch-scoped dev logs in `.claude/context/dev/<branch-slug>/` must be deleted in the merge PR, **or** their durable content promoted to a permanent doc first.

### Python environment

Always use `acoustic_ai/.venv` for AI training and inference (`./acoustic_ai/.venv/bin/python`, `./acoustic_ai/.venv/bin/accelerate`, etc.). Do **not** use system / Homebrew `python3`, `pip`, `accelerate`, or `uvicorn` — they load incompatible torch/torchaudio builds.

DVC and its S3 deps are the exception: they live at user-site (`pip3 install --user ...`), **not** in the venv. Git hooks call `dvc` and must work without venv activation. See [.claude/context/dev_specifications/dvc_workflow.md](.claude/context/dev_specifications/dvc_workflow.md).

### Model checkpoint layout

```
model/
├── candidates/<member>/<run-id>/    # per-experiment checkpoints (DVC-tracked)
└── production/<role>/               # promoted checkpoint slots, created only after sign-off
```

At this stage of the project, **nothing is in production**. Every trained
checkpoint — including the VAE and vocoder used by the current inference path
— is a candidate. A `model/production/<role>/` slot will be created only after
an explicit promotion decision (validation, sign-off, release tagging).

Rules (team workflow):
- One folder per member, one folder per run — never overwrite another member's candidates.
- Train into `candidates/<member>/...`.
- Each model folder under `model/candidates/...` or `model/production/...` ships with `README.md` + DVC pointer(s); candidate folders also ship with `params.yaml`, and add `metrics.json` once evals exist.
- Model `README.md` files are required experiment / checkpoint logs. Use [.claude/context/dev_specifications/model_readme_standard.md](.claude/context/dev_specifications/model_readme_standard.md); keep the audit section empty until developers provide evaluation notes or review findings.

Binaries (`.pt`, `.safetensors`, `.bin`, `.ckpt`) are DVC-tracked; metadata (`*.json`, `*.yaml`, `*.md`, `*.dvc`) is git-tracked.

### Pipeline vs. candidate hyperparameters

- Root `params.yaml` → only contains hyperparameters for stages declared in `dvc.yaml` (currently `ambient` and `vocoder`). Changes here trigger `dvc repro` re-runs.
- `model/candidates/<member>/<run-id>/params.yaml` → all per-candidate experiment hyperparameters, sectioned `training:` and `inference:`.

### Layer A dev-generation contract

The Layer A smoke LoRAs are trained on tiny datasets, so the dev generation path is locked down server-side:

- Frontend exposes **only** a non-negative integer `seed` (range `0`–`2147483647`).
- Express backend forwards **only** `{ seed }`.
- FastAPI AI server owns the prompt, checkpoint, guidance, step count, audio length, RMS, and high-pass.
- Server returns all parameters in response metadata for debugging.

Seed is **not** temperature — it initializes the diffusion noise. Same seed + same params + same code path = effectively the same audio.

### Git branch naming

```
<type>/<author>/<short-description>
```

Types: `feat`, `fix`, `data`, `model`, `infra`, `refactor`, `docs`, `exp`.
Example: `model/lucas/layer-c-event-attemp-1`.

Commit subjects use imperative mood, ≤72 chars, no issue numbers in the subject. Full conventions: [.claude/context/dev_specifications/git_workflow.md](.claude/context/dev_specifications/git_workflow.md).

### Pre-commit file audit

Mandatory before every commit:

1. Run `git status`.
2. If unintended files appear (binaries, generated outputs, credentials, OS artefacts), don't commit.
3. Add them to `.gitignore` (and `git rm --cached <path>` if already tracked).
4. Verify `git status` is clean of unintended files.

Large binaries never go to git — use DVC.

---

## Running services

| Service | How | URL |
|---------|-----|-----|
| Frontend | Docker | `http://localhost:5173` |
| Backend | Docker | `http://localhost:4000` |
| PostgreSQL | Docker | `localhost:5432` |
| AI server | **Native only** (GPU/MPS) | `http://localhost:8000` |

Docker cannot access macOS MPS — the AI server **must** run natively.

Commands, env vars, ports: [.claude/context/setup/services.md](.claude/context/setup/services.md).

---

## DVC + S3 (essentials)

- Binaries (checkpoints, audio archives, latents) live in S3 — `s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache/` (region `ap-southeast-2`, profile `capstone2`).
- Git stores only `.dvc` pointer files.
- Remote is already declared in `.dvc/config`; new machines just need the install + AWS profile.
- Full workflow (fresh-clone, daily commands, candidate discipline, troubleshooting): [.claude/context/dev_specifications/dvc_workflow.md](.claude/context/dev_specifications/dvc_workflow.md).

---

## Where to find everything

Full index: [.claude/context/README.md](.claude/context/README.md).

Quick links:

| Need | Doc |
|---|---|
| AI architecture | [.claude/context/ai/architecture.md](.claude/context/ai/architecture.md) |
| Pipeline design (generation + analysis) | [.claude/context/ai/pipeline_design.md](.claude/context/ai/pipeline_design.md) |
| MVP decision log | [.claude/context/ai/logs/mvp_decision_log.md](.claude/context/ai/logs/mvp_decision_log.md) |
| Smoke-test runbooks | [.claude/context/ai/runbooks/](.claude/context/ai/runbooks/) |
| DVC + S3 workflow | [.claude/context/dev_specifications/dvc_workflow.md](.claude/context/dev_specifications/dvc_workflow.md) |
| Model README standard | [.claude/context/dev_specifications/model_readme_standard.md](.claude/context/dev_specifications/model_readme_standard.md) |
| Data alignment & env features | [.claude/context/data/data_reference.md](.claude/context/data/data_reference.md) |
| S3 bucket layout | [.claude/context/dev_specifications/s3_bucket_layout.md](.claude/context/dev_specifications/s3_bucket_layout.md) |
| MVP dataset (site 257) | [.claude/skills/sample_mvp_dataset.md](.claude/skills/sample_mvp_dataset.md) |
| Known data issues | [.claude/context/issues/known_issues.md](.claude/context/issues/known_issues.md) |
| Services / hardware | [.claude/context/setup/](.claude/context/setup/) |
| Git workflow (full) | [.claude/context/dev_specifications/git_workflow.md](.claude/context/dev_specifications/git_workflow.md) |

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
