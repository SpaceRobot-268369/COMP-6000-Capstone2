# COMP-6000 Capstone 2 — Speculative Soundscape Generation

Research prototype: ecoacoustic recordings + environmental data → AI-generated speculative soundscapes.

**Three modes:** Analysis, Generation, Transformation.
**Modeling approach:** layered composition (ambient bed + weather + events + mix), not a single generated waveform.

---

## Contents

1. [System at a glance](#system-at-a-glance)
2. [.claude/ directory map](#claude-directory-map)
3. [Critical conventions](#critical-conventions)
4. [Running services](#running-services)
5. [DVC + S3 (essentials)](#dvc--s3-essentials)
6. [Backend API](#backend-api)
7. [Notes](#notes)

---

## System at a glance

| # | Component | Tech | Status |
|---|-----------|------|--------|
| 1 | Frontend | React + Vite (`frontend/`) | UI scaffold |
| 2 | Backend | Express.js + PostgreSQL (`backend/`) | Auth endpoints |
| 3 | AI module | Python / PyTorch (`acoustic_ai/`) | Layer A smoke ✓, Layer C smoke ✓ |
| 4 | Metadata DB (optional) | PostgreSQL | Not started |

### AI modules (per-layer codes)

| Layer | Role | Status |
|---|---|---|
| layer-a (Ambient) | AudioLDM2 LoRA (base: `cvssp/audioldm2`) for ambient bed | smoke-1 (spring night) ✓, smoke-2 (insects) ✓ |
| layer-b (Weather) | Curated wind/rain assets + parameter mixing | Placeholder |
| layer-c (Events) | AudioGen LoRA per species (base: `facebook/audiogen-medium`, 16 kHz native) | smoke-1 (boobook) ✓ |
| layer-d (Mixer) | Combine A+B+C → WAV + explanation JSON | Placeholder |
| layer-e (Analysis) | Ambient similarity + weather + event detectors | Partial (layer-a working) |

Each layer hosts independent attempts under
`acoustic_ai/layers/<layer-code>/attempts/<member>__<stage>__<slug>/`.
Stage tokens (`smoke-N` / `mvp-N` / `prod-N`) and the full naming rules
live in [.claude/context/conventions.md](.claude/context/conventions.md).
The set of available attempts is declared in `acoustic_ai/registry.yaml`
— the FastAPI server reads it to serve `GET /layers` for the frontend
dropdown.

**Generative model strategy:** Layers A and C use frozen large base models + LoRA adapters for the MVP. Migration to in-house distilled models is a future option, gated on (1) LoRA path proven across species/contexts, (2) demonstrated latency/VRAM bottleneck, (3) team capacity. See [distillation strategy](.claude/context/ai/distillation_strategy.md).

---

## Repo layout

Top-level map. Each entry links to its canonical deep-dive doc where one exists.

```
COMP-6000-Capstone2/
├── CLAUDE.md                # this file — agent guidance + structural index
├── AGENTS.md                # points agents at CLAUDE.md
├── Makefile                 # convenience targets
├── dvc.yaml / dvc.lock      # DVC pipeline definition + lock
├── params.yaml              # hyperparameters for stages declared in dvc.yaml
│
├── frontend/                # React + Vite UI scaffold (Docker, port 5173)
├── backend/                 # Express.js + PostgreSQL (Docker, port 4000); /api/health, /api/register, /api/login
├── services/dev/            # docker-compose.yml + db_init.sql for local frontend+backend+postgres
│
├── acoustic_ai/             # Python AI module (FastAPI server runs natively for MPS)
│   ├── server/              # registry.py + server.py — registry-driven FastAPI app on :8000
│   ├── layers/              # per-layer attempts (layer_a, layer_b, layer_c, …)
│   │                        #   layer_<X>/attempts/<member>__<stage>__<slug>/  — see conventions.md
│   ├── scripts/             # extract_expected_samples.py, regenerate_samples.py
│   ├── registry.yaml        # declares which attempts the server exposes via GET /layers
│   ├── requirements.txt
│   └── .venv/               # gitignored — the ONLY Python interpreter for AI work (see "Python environment")
│
├── model/                   # trained checkpoints
│   ├── candidates/<member>/<stage>__<slug>/   # all current checkpoints (binaries DVC-tracked)
│   └── production/<role>/                     # promoted slots — empty until explicit promotion
│
├── resources/               # source recordings + manifests (DVC-tracked)
│   └── site_257_bowra-dry-a/                  # only site live right now
│
├── script/                  # data prep & download utilities (one-shot scripts, not pipeline stages)
│   ├── dataset/             # manifest builders, segment prep, spectrogram rendering for datasets
│   ├── download/            # site_257 clip/annotation/event downloaders, recording fetcher
│   └── env/                 # NASA env-feature fetcher
│
├── debug/                   # local-only diagnostics workspace (per-layer subfolders; see debug/README.md)
└── .claude/                 # agent context loaded on demand — full tree in next section
```

Conventions for top-level entries:
- **frontend/**, **backend/**, **services/dev/** → containerised; run via Docker Compose in `services/dev/`.
- **acoustic_ai/** → native only (Apple Silicon MPS); never `pip install` outside `acoustic_ai/.venv` (DVC is the documented exception — see "Python environment").
- **model/**, **resources/** → binaries are DVC, metadata (`*.json`, `*.yaml`, `*.md`, `*.dvc`) is git.
- **script/** vs **acoustic_ai/scripts/** → top-level `script/` is for data preparation (one-shot, ad-hoc); `acoustic_ai/scripts/` is for AI-module utilities that read/write artefacts under `acoustic_ai/`.

When a top-level dir is added/removed/renamed, this section must be updated in the same commit (same discipline as the `.claude/` map below).

---

## .claude/ directory map

CLAUDE.md is the **structural index** for `.claude/`. The tree below is the single source of truth — when files are added, moved, renamed, or removed under `.claude/`, this section must be updated in the same commit. Agents should refuse to land structural changes that leave this section stale.

```
.claude/
├── settings.local.json
├── commands/                              # Custom slash-command definitions
├── skills/                                # Reusable agent skills
│   └── training_data_filtering_policy.md
└── context/                               # Project context the agent loads on demand
    ├── conventions.md                     # Canonical doc: repo structure, naming, tracking, artifact tiers, attempt internals, model README
    ├── ai/                                # AI module design, runbooks, decision logs
    │   ├── architecture.md
    │   ├── pipeline_design.md
    │   ├── distillation_strategy.md
    │   ├── runbooks/
    │   │   ├── layer_a_smoke_1_spring_night.md
    │   │   ├── layer_a_smoke_2_insects.md
    │   │   └── layer_c_smoke_1_birds.md
    │   └── logs/
    │       ├── mvp_decision_log.md
    │       └── audioldm2_transition_log.md
    ├── data/                              # Dataset alignment, env features, known data issues
    │   ├── data_reference.md
    │   ├── known_issues.md
    │   └── logs/
    │       └── generation_quality_analysis.md
    ├── dev/                               # Developer workflows: git, DVC, S3, testing
    │   ├── git_workflow.md
    │   ├── dvc_workflow.md
    │   ├── s3_bucket_layout.md
    │   └── testing/
    │       ├── analysis_test_cases.md
    │       └── layer_verification_formats.md
    └── setup/                             # How to run the system
        ├── local/
        │   └── services.md                # Local-mac service topology, ports, env vars
        └── server/
            └── on_demand_ai_worker.md     # Server A/B job orchestration topology
```

### Quick-link table

| Need | Doc |
|---|---|
| **Conventions** (repo structure, naming, tracking, artifact tiers, attempt internals, model README) | [.claude/context/conventions.md](.claude/context/conventions.md) |
| AI architecture | [.claude/context/ai/architecture.md](.claude/context/ai/architecture.md) |
| Pipeline design (generation + analysis) | [.claude/context/ai/pipeline_design.md](.claude/context/ai/pipeline_design.md) |
| Distillation strategy | [.claude/context/ai/distillation_strategy.md](.claude/context/ai/distillation_strategy.md) |
| Smoke-test runbooks | [.claude/context/ai/runbooks/](.claude/context/ai/runbooks/) |
| MVP decision log | [.claude/context/ai/logs/mvp_decision_log.md](.claude/context/ai/logs/mvp_decision_log.md) |
| AudioLDM2 transition log | [.claude/context/ai/logs/audioldm2_transition_log.md](.claude/context/ai/logs/audioldm2_transition_log.md) |
| Data alignment & env features | [.claude/context/data/data_reference.md](.claude/context/data/data_reference.md) |
| Known data issues | [.claude/context/data/known_issues.md](.claude/context/data/known_issues.md) |
| Generation quality analysis | [.claude/context/data/logs/generation_quality_analysis.md](.claude/context/data/logs/generation_quality_analysis.md) |
| Git workflow (full) | [.claude/context/dev/git_workflow.md](.claude/context/dev/git_workflow.md) |
| DVC workflow | [.claude/context/dev/dvc_workflow.md](.claude/context/dev/dvc_workflow.md) |
| S3 bucket layout | [.claude/context/dev/s3_bucket_layout.md](.claude/context/dev/s3_bucket_layout.md) |
| Analysis test cases | [.claude/context/dev/testing/analysis_test_cases.md](.claude/context/dev/testing/analysis_test_cases.md) |
| Layer verification & handoff formats | [.claude/context/dev/testing/layer_verification_formats.md](.claude/context/dev/testing/layer_verification_formats.md) |
| Local services + ports | [.claude/context/setup/local/services.md](.claude/context/setup/local/services.md) |
| On-demand AI worker topology | [.claude/context/setup/server/on_demand_ai_worker.md](.claude/context/setup/server/on_demand_ai_worker.md) |
| Training data filtering policy (site 257 MVP sample) | [.claude/skills/training_data_filtering_policy.md](.claude/skills/training_data_filtering_policy.md) |

---

## Critical conventions

> Canonical doc for project-wide naming, layout, and policy rules:
> [.claude/context/conventions.md](.claude/context/conventions.md). The
> subsections below are the **canonical home** for CLAUDE.md-only rules
> (Storage rule, Python environment, Pipeline-vs-attempt params, Layer A
> dev-generation contract). Everything else — repo structure, attempt
> naming, artifact tiers, model README — lives in `conventions.md`.

### Storage rule

All Claude-loadable context lives under `.claude/` — never at the project root.

| Type | Location |
|------|----------|
| Architecture / design docs | `.claude/context/ai/`, `.claude/context/data/` |
| Runbooks (smoke tests, training workflows) | `.claude/context/ai/runbooks/` |
| Dev workflow specs (DVC, S3, git, model README standard) | `.claude/context/dev/` |
| Sample artifacts (expected/showcase/dev-artifacts-self-testing) | `acoustic_ai/layers/<layer>/attempts/<id>/{expected,showcase,dev-artifacts-self-testing}/` (rules: `.claude/context/conventions.md`) |
| Testing specs | `.claude/context/dev/testing/` |
| Setup (local services; server reserved) | `.claude/context/setup/local/`, `.claude/context/setup/server/` |
| Custom slash commands | `.claude/commands/` |
| Reusable skills | `.claude/skills/` |
| Settings | `.claude/settings.local.json` |

CLAUDE.md is a **hub**, not a manual. If a section grows past ~5 lines or is referenced from elsewhere, it moves into `.claude/context/` and CLAUDE.md links to it.

### Python environment

Always use `acoustic_ai/.venv` for AI training and inference (`./acoustic_ai/.venv/bin/python`, `./acoustic_ai/.venv/bin/accelerate`, etc.). Do **not** use system / Homebrew `python3`, `pip`, `accelerate`, or `uvicorn` — they load incompatible torch/torchaudio builds.

DVC and its S3 deps are the exception: they live at user-site (`pip3 install --user ...`), **not** in the venv. Git hooks call `dvc` and must work without venv activation. See [.claude/context/dev/dvc_workflow.md](.claude/context/dev/dvc_workflow.md).

### Attempts and checkpoints

Layer code lives under `acoustic_ai/layers/<layer-code>/attempts/`, and the
matching checkpoints live under `model/candidates/<member>/`. Both sides share
the same naming convention:

```
acoustic_ai/layers/layer-<X>/attempts/<member>__<stage>__<slug>/   # code
model/candidates/<member>/<stage>__<slug>/                         # checkpoint
model/production/<role>/                                           # promoted slot
```

`<stage>` is one of `smoke-N`, `mvp-N`, `prod-N`. Full rules and examples are
in [.claude/context/conventions.md](.claude/context/conventions.md).

At this stage of the project, **nothing is in production**. Every trained
checkpoint — including the VAE and vocoder used by the smoke-4 inference
path — is a candidate. A `model/production/<role>/` slot will be created
only after an explicit promotion decision (validation, sign-off, release
tagging).

Rules (team workflow):
- One folder per member, one folder per attempt — never overwrite another member's work.
- Attempts are **self-contained**: each `acoustic_ai/layers/.../attempts/<id>/` folder owns its `data/`, `precompute/`, `debug/`, `train.py`, `sample.py`, `handler.py`, `README.md`. No shared `common/` folder; duplication between attempts is intentional.
- Each model folder under `model/candidates/...` or `model/production/...` ships with `README.md` + DVC pointer(s); candidate folders also ship with `params.yaml`, and add `metrics.json` once evals exist.
- Model `README.md` files are required experiment / checkpoint logs. Use [.claude/context/conventions.md § Model checkpoint README](.claude/context/conventions.md#6-model-checkpoint-readme); keep the audit section empty until developers provide evaluation notes or review findings.

Binaries (`.pt`, `.safetensors`, `.bin`, `.ckpt`) are DVC-tracked; metadata (`*.json`, `*.yaml`, `*.md`, `*.dvc`) is git-tracked.

**Sample artifacts** (audio + spectrograms) per attempt live in three tiers
directly under the attempt root:
`acoustic_ai/layers/<layer>/attempts/<id>/{expected,showcase,dev-artifacts-self-testing}/`.
Each case (one source clip or one generated seed) is its own subdirectory
with fixed filenames `{audio.wav, spectrogram.png, metadata.json}`:

- `expected/real_<clip_id>/` — **real-audio** ground truth; PNG + JSON in
  git, WAV via `.wav.dvc`.
- `showcase/seed_<N>_<label>/` — author-curated **generated** samples; all
  three files DVC-tracked.
- `dev-artifacts-self-testing/` — ad-hoc dev scratch; folder tracked via
  `.gitkeep`, all other contents gitignored.

Spectrogram PNGs also carry baked-in metadata: a visible overlay
(header/subline/footer) and lossless PNG `tEXt` chunks mirroring the JSON
sidecar. Python source for the attempt lives under `code/`. Canonical seed
is `42` (showcase + Dev UI only — `expected/` is real audio). Full rules:
[.claude/context/conventions.md](.claude/context/conventions.md). Regenerate via
`acoustic_ai/scripts/regenerate_samples.py`.

### Pipeline vs. attempt hyperparameters

- Root `params.yaml` → only contains hyperparameters for stages declared in `dvc.yaml`. Changes here trigger `dvc repro` re-runs.
- `acoustic_ai/layers/<layer>/attempts/<id>/params.yaml` → per-attempt experiment hyperparameters, sectioned `training:` and `inference:`.
- `model/candidates/<member>/<stage>__<slug>/params.yaml` → frozen snapshot of the params used to train the checkpoint (matches the attempt's `params.yaml` at training time).

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

Commit subjects use imperative mood, ≤72 chars, no issue numbers in the subject. Full conventions: [.claude/context/dev/git_workflow.md](.claude/context/dev/git_workflow.md).

### Pre-commit file audit

Mandatory before every commit:

1. Run `git status`.
2. If unintended files appear (binaries, generated outputs, credentials, OS artefacts), don't commit.
3. Add them to `.gitignore` (and `git rm --cached <path>` if already tracked).
4. Verify `git status` is clean of unintended files.

Large binaries never go to git — use DVC. Full "do not track" table (categories, patterns, where they live instead): [.claude/context/dev/git_workflow.md#do-not-track-in-git](.claude/context/dev/git_workflow.md#do-not-track-in-git).

---

## Running services

| Service | How | URL |
|---------|-----|-----|
| Frontend | Docker | `http://localhost:5173` |
| Backend | Docker | `http://localhost:4000` |
| PostgreSQL | Docker | `localhost:5432` |
| AI server | **Native only** (GPU/MPS) | `http://localhost:8000` |

Docker cannot access macOS MPS — the AI server **must** run natively.

Commands, env vars, ports: [.claude/context/setup/local/services.md](.claude/context/setup/local/services.md).

---

## DVC + S3 (essentials)

- Binaries (checkpoints, audio archives, latents) live in S3 — `s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache/` (region `ap-southeast-2`, profile `capstone2`).
- Git stores only `.dvc` pointer files.
- Remote is already declared in `.dvc/config`; new machines just need the install + AWS profile.
- Full workflow (fresh-clone, daily commands, candidate discipline, troubleshooting): [.claude/context/dev/dvc_workflow.md](.claude/context/dev/dvc_workflow.md).

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
