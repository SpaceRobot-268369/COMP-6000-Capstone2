# Project Conventions — Index

Single entry point for naming, layout, and policy rules in this repo. Each
row gives a one-line rule statement and a link to the canonical deep-dive.
**No content is duplicated here** — when a rule changes, edit the linked doc.

If you're adding a new convention, add it to the linked deep-dive (or create
a new one under `.claude/context/dev/`) and register it here with one line.

---

## Repo layout

| Rule | Canonical doc |
|---|---|
| Top-level dirs (`frontend/`, `backend/`, `acoustic_ai/`, `model/`, `resources/`, `script/`, `services/`, `debug/`) | [CLAUDE.md § Repo layout](../../../CLAUDE.md#repo-layout) |
| `.claude/` directory tree | [CLAUDE.md § .claude/ directory map](../../../CLAUDE.md#claude-directory-map) |
| Per-attempt internal layout (`code/`, `data/`, `expected/`, `showcase/`, `dev-artifacts-self-testing/`, …) | [attempt_naming.md](attempt_naming.md) |
| AI module internals (server, layers, scripts, registry) | [../ai/architecture.md](../ai/architecture.md) |

## Naming

| Rule | Canonical doc |
|---|---|
| Git branches: `<type>/<author>/<short-description>` (types: `feat`, `fix`, `data`, `model`, `infra`, `refactor`, `docs`, `exp`) | [git_workflow.md](git_workflow.md) |
| Commit subjects: imperative, ≤72 chars, no issue numbers in subject | [git_workflow.md](git_workflow.md) |
| Attempt folders: `<member>__<stage>__<slug>` under `acoustic_ai/layers/layer_<X>/attempts/` | [attempt_naming.md](attempt_naming.md) |
| Stage tokens: `smoke-N` \| `mvp-N` \| `prod-N` | [attempt_naming.md](attempt_naming.md) |
| Checkpoint folders: `model/candidates/<member>/<stage>__<slug>/`; promotion lands in `model/production/<role>/` | [attempt_naming.md](attempt_naming.md) |
| Expected case dirs (real audio): `expected/real_<source_clip_id>/` | [artifact_policy.md](artifact_policy.md) |
| Showcase case dirs (generated): `showcase/seed_<N>_<short_label>/` | [artifact_policy.md](artifact_policy.md) |
| Fixed filenames inside a case dir: `audio.wav`, `spectrogram.png`, `metadata.json` (plus matching `.dvc` pointers where applicable) | [artifact_policy.md](artifact_policy.md) |

## Storage & tracking

| Rule | Canonical doc |
|---|---|
| What lives under `.claude/` (Claude-loadable context only — never at the project root) | [CLAUDE.md § Storage rule](../../../CLAUDE.md#storage-rule) |
| Artifact tiers (`expected/` real-audio, `showcase/` generated, `dev-artifacts-self-testing/` local) + git/DVC split per tier + PNG metadata baking (overlay + tEXt chunks) | [artifact_policy.md](artifact_policy.md) |
| Model README required sections (model card + run log + audit) | [model_readme_standard.md](model_readme_standard.md) |
| S3 bucket layout (`s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache/`) | [s3_bucket_layout.md](s3_bucket_layout.md) |
| DVC workflow (fresh-clone, daily commands, candidate discipline, troubleshooting) | [dvc_workflow.md](dvc_workflow.md) |
| Binaries (`.pt`, `.safetensors`, `.bin`, `.ckpt`) → DVC; metadata (`*.json`, `*.yaml`, `*.md`, `*.dvc`) → git | [dvc_workflow.md](dvc_workflow.md) |
| Per-attempt `.gitignore` shape (ignore `*.wav`, showcase PNG/JSON; track `dev-artifacts-self-testing/` folder via `.gitkeep`, ignore contents) | [artifact_policy.md](artifact_policy.md) |

## Pipelines & params

| Rule | Canonical doc |
|---|---|
| Root `params.yaml` only contains hyperparameters for stages declared in `dvc.yaml`; per-attempt experiment params live in `<attempt>/params.yaml`; checkpoints ship with a frozen snapshot | [CLAUDE.md § Pipeline vs. attempt hyperparameters](../../../CLAUDE.md#pipeline-vs-attempt-hyperparameters) |
| Canonical seed `42` (applies to **showcase** + the live Dev UI only — not to `expected/`, which is real audio) | [artifact_policy.md](artifact_policy.md) |

## Runtime contracts

| Rule | Canonical doc |
|---|---|
| Python: always use `acoustic_ai/.venv` for AI work (`./acoustic_ai/.venv/bin/python`, `…/accelerate`, …); never system/Homebrew. DVC + its S3 deps are the documented exception (user-site). | [CLAUDE.md § Python environment](../../../CLAUDE.md#python-environment) |
| Layer A dev-generation contract: frontend & backend forward only `{ seed }`; AI server owns prompt/checkpoint/guidance/steps/length/RMS/high-pass; seed is **not** temperature | [CLAUDE.md § Layer A dev-generation contract](../../../CLAUDE.md#layer-a-dev-generation-contract) |
| AI server runs natively only (GPU/MPS); frontend/backend/Postgres run in Docker | [../setup/local/services.md](../setup/local/services.md) |

## Workflow

| Rule | Canonical doc |
|---|---|
| Pre-commit file audit (`git status` → audit unintended files → `.gitignore` / `git rm --cached` if needed) | [git_workflow.md](git_workflow.md) |
| "Do not track in git" categories (large binaries, generated outputs, credentials, OS artefacts — full table) | [git_workflow.md](git_workflow.md) |
| Branch-scoped scratch in `.claude/context/branches/<branch-slug>/` must be deleted in the merge PR (or content promoted first) | [CLAUDE.md § Storage rule](../../../CLAUDE.md#storage-rule) |
| Training-data filtering policy (site 257 MVP sample) | [../../skills/training_data_filtering_policy.md](../../skills/training_data_filtering_policy.md) |
