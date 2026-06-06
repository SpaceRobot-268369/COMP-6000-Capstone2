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
| layer-a (Ambient) | AudioLDM2 LoRA (base: `cvssp/audioldm2`) for ambient bed | smoke-1/2 ✓ · **prod-1 per-cell bank (16 season×diel) promoted** → `model/production/layer_a_ambient/` |
| layer-b (Weather) | Curated wind/rain assets + parameter mixing | Placeholder |
| layer-c (Events) | AudioGen LoRA per species (base: `facebook/audiogen-medium`, 16 kHz native) | smoke-1 (boobook) ✓ |
| layer-d (Mixer) | Combine A+B+C → WAV + explanation JSON | Placeholder |
| layer-e (Analysis) | Three detector heads (ambient similarity + weather + event) on the raw mixture → aggregator fuses latent context (season/diel) → report | Partial (layer-a working) |

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
├── services/dev/            # local docker-compose.yml + db_init.sql + serverB SSH tunnel sidecar
├── services/server-a/       # Server A deployment compose + env template
│
├── acoustic_ai/             # Python AI module (FastAPI app runs natively on serverB for inference)
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
│   └── production/<role>/                     # promoted slots (layer_a_ambient ✓)
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
│   ├── commit_changes.md
│   ├── dvc_push.md
│   └── pre_pr_checklist.md
└── context/                               # Project context the agent loads on demand
    ├── conventions.md                     # Canonical doc: repo structure, naming, tracking, artifact tiers, attempt internals, model README
    ├── ai/                                # AI module design, runbooks, decision logs
    │   ├── prerequisites.md               # Conceptual on-ramp: audio fundamentals, encoder/decoder, LoRA, ecosystem
    │   ├── architecture.md
    │   ├── pipeline_design.md
    │   ├── analysis_synthesis_policy.md    # Layer E: aggregator fusion + LLM-OSS report policy + per-head pass standards
    │   ├── distillation_strategy.md
    │   ├── runbooks/
    │   │   ├── layer_a_smoke_1_spring_night.md
    │   │   ├── layer_a_smoke_2_insects.md
    │   │   └── layer_c_smoke_1_birds.md
    │   └── logs/
    │       ├── mvp_decision_log.md
    │       ├── caption_schema_log.md
    │       └── audioldm2_transition_log.md
    ├── data/                              # Dataset alignment, env features, known data issues
    │   ├── data_reference.md
    │   ├── known_issues.md
    │   └── logs/
    │       └── generation_quality_analysis.md
    ├── dev/                               # Developer workflows: git, DVC, S3
    │   ├── dev_workflow.md                # Stage workflow: smoke → mvp/prod loop
    │   ├── server_training_workflow.md    # Train on serverB → push branch → pull locally → PR
    │   ├── git_workflow.md
    │   ├── dvc_workflow.md
    │   ├── s3_bucket_layout.md
    │   └── cicd_design.md
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
| **AI prerequisites** (audio fundamentals, encoder/decoder, LoRA, pre-trained ecosystem) | [.claude/context/ai/prerequisites.md](.claude/context/ai/prerequisites.md) |
| AI architecture | [.claude/context/ai/architecture.md](.claude/context/ai/architecture.md) |
| Pipeline design (generation + analysis) | [.claude/context/ai/pipeline_design.md](.claude/context/ai/pipeline_design.md) |
| **Analysis synthesis policy** (Layer E aggregator fusion, LLM-OSS report registers, per-head pass standards, phenology table) | [.claude/context/ai/analysis_synthesis_policy.md](.claude/context/ai/analysis_synthesis_policy.md) |
| Distillation strategy | [.claude/context/ai/distillation_strategy.md](.claude/context/ai/distillation_strategy.md) |
| Smoke-test runbooks | [.claude/context/ai/runbooks/](.claude/context/ai/runbooks/) |
| MVP decision log | [.claude/context/ai/logs/mvp_decision_log.md](.claude/context/ai/logs/mvp_decision_log.md) |
| Caption schema log (Layer A) | [.claude/context/ai/logs/caption_schema_log.md](.claude/context/ai/logs/caption_schema_log.md) |
| AudioLDM2 transition log | [.claude/context/ai/logs/audioldm2_transition_log.md](.claude/context/ai/logs/audioldm2_transition_log.md) |
| Data alignment & env features | [.claude/context/data/data_reference.md](.claude/context/data/data_reference.md) |
| Layer B site clip filtering policy | [acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/site_clip_filtering_policy.md](acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/site_clip_filtering_policy.md) |
| Layer B site weather audit v0 | [acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/site_weather_audit_v0.md](acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/site_weather_audit_v0.md) |
| Layer B weather asset schema | [acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/weather_asset_index_schema.md](acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/weather_asset_index_schema.md) |
| Known data issues | [.claude/context/data/known_issues.md](.claude/context/data/known_issues.md) |
| Generation quality analysis | [.claude/context/data/logs/generation_quality_analysis.md](.claude/context/data/logs/generation_quality_analysis.md) |
| **Stage workflow** (smoke → mvp/prod loop, generation mode) | [.claude/context/dev/dev_workflow.md](.claude/context/dev/dev_workflow.md) |
| **Server training workflow** (train on serverB → push branch → pull locally → PR) | [.claude/context/dev/server_training_workflow.md](.claude/context/dev/server_training_workflow.md) |
| Git workflow (full) | [.claude/context/dev/git_workflow.md](.claude/context/dev/git_workflow.md) |
| DVC workflow | [.claude/context/dev/dvc_workflow.md](.claude/context/dev/dvc_workflow.md) |
| S3 bucket layout | [.claude/context/dev/s3_bucket_layout.md](.claude/context/dev/s3_bucket_layout.md) |
| CI/CD design | [.claude/context/dev/cicd_design.md](.claude/context/dev/cicd_design.md) |
| Local services + ports | [.claude/context/setup/local/services.md](.claude/context/setup/local/services.md) |
| Server A deployment compose | [services/server-a/README.md](services/server-a/README.md) |
| On-demand AI worker topology | [.claude/context/setup/server/on_demand_ai_worker.md](.claude/context/setup/server/on_demand_ai_worker.md) |
| Commit changes (git + DVC) skill | [.claude/skills/commit_changes.md](.claude/skills/commit_changes.md) |
| DVC push to S3 skill | [.claude/skills/dvc_push.md](.claude/skills/dvc_push.md) |
| Pre-PR checklist skill | [.claude/skills/pre_pr_checklist.md](.claude/skills/pre_pr_checklist.md) |

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

First production promotion: **`model/production/layer_a_ambient/`** — the
Layer A per-cell ambient bank, promoted from
`lucas__mvp_2__per_cell_loras` (served via attempt
`lucas__prod_1__per_cell_loras`, the layer_a registry default). Promoted
**with documented caveats** — see the production card's audit section.
Everything else — including the VAE and vocoder used by the smoke-4 inference
path — remains a candidate. A `model/production/<role>/` slot is created
only after an explicit promotion decision (validation, sign-off, release
tagging) per [conventions §5.4](.claude/context/conventions.md).

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

The Layer A LoRAs are trained on narrow datasets, so the dev generation path is locked down server-side:

- Frontend exposes **only** a non-negative integer `seed` (range `0`–`2147483647`), plus — for **bank attempts** that declare `uses_cells: true` — a `(season, diel)` cell selector (two dropdowns populated from the attempt's `cells` list).
- Express backend forwards **only** `{ seed }`, plus `{ season, diel }` when both are valid (`season ∈ {spring,summer,autumn,winter}`, `diel ∈ {dawn,morning,afternoon,night}`); invalid/absent selectors are dropped and the server falls back to `default_cell`.
- FastAPI AI server owns the prompt, checkpoint, guidance, step count, audio length, RMS, and high-pass. For bank attempts it routes `(season, diel)` → the matching per-cell LoRA adapter (PEFT `set_adapter`) and uses that cell's locked prompt.
- Server returns all parameters (including the resolved `cell`) in response metadata for debugging.

Seed is **not** temperature — it initializes the diffusion noise. Same seed + same cell + same params + same code path = effectively the same audio.

The first bank attempt is `lucas__mvp_2__per_cell_loras` (16 season×diel adapters). See [.claude/context/ai/runbooks/](.claude/context/ai/runbooks/) once a Phase-3 runbook is added.

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
| AI tunnel | Docker sidecar | `ai-tunnel:8000` inside Compose |
| AI server | Native on serverB | `serverB:127.0.0.1:8000` via SSH tunnel |

The Docker backend reaches serverB through the Compose `ai-tunnel` sidecar.
The FastAPI process itself runs natively on serverB out of
`~/shiny-pikachu/` — a dedicated clone pinned to `main`; never `git
checkout` another branch in that tree. Per-member experiment clones live
beside it (e.g. `~/lucano/COMP-6000-Capstone2/`) and are free to switch
branches. Startup / health / stop commands and the working-tree
convention: [.claude/context/setup/server/on_demand_ai_worker.md](.claude/context/setup/server/on_demand_ai_worker.md).
Keep SSH keys outside the repository and use the key convention in
`services/dev/README.md`.

Commands, env vars, ports: [.claude/context/setup/local/services.md](.claude/context/setup/local/services.md).

---

## DVC + S3 (essentials)

- Binaries (checkpoints, audio archives, latents) live in S3 — `s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache/` (region `ap-southeast-2`, profile `capstone2`).
- Git stores only `.dvc` pointer files.
- Remote is already declared in `.dvc/config`; new machines just need the install + AWS profile.
- Full workflow (fresh-clone, daily commands, candidate discipline, troubleshooting): [.claude/context/dev/dvc_workflow.md](.claude/context/dev/dvc_workflow.md).

### Pre-Commit File Audit

Before every commit, check whether any unintended files are being tracked by git.

1. Run `git status` and inspect the staged and untracked file lists.
2. If any file appears that should not be committed (large binaries, generated outputs, credentials, editor artefacts, OS files, etc.), **do not commit yet**.
3. Add the offending path(s) to `.gitignore` (and run `git rm --cached <path>` if the file is already tracked).
4. Verify `git status` is clean of unintended files before proceeding with the commit.

This check is mandatory — never skip it, even for "quick" commits.

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
