# `.claude/context/` — Project Context Index

Hub for all Claude-loadable design, architecture, runbook, and reference docs.
CLAUDE.md at the repo root is the always-loaded hub and links here.

**Storage rule:** if a section grows past ~5 lines in CLAUDE.md or is linked
from elsewhere, it moves here and CLAUDE.md links to it.

## Structure

| Folder | Contents |
|---|---|
| `ai/` | AI module architecture, pipeline design, distillation strategy, transition logs |
| `ai/runbooks/` | Step-by-step training + sampling workflows for each smoke test or candidate run |
| `ai/logs/` | Long-form decision logs (MVP, transitions) |
| `data/` | Data alignment and generation-quality analysis |
| `data/logs/` | Detailed analysis runs |
| `dev_specifications/` | Git, DVC, S3, and other developer workflow specifications |
| `setup/` | Service topology, hardware requirements, local dev setup |
| `testing/` | Analysis test cases, layer verification formats |
| `issues/` | Known data issues, won't-fix lists |
| `diagrams/` | Workflow diagrams |
| `dev/<branch-slug>/` | Ephemeral branch-scoped scratch — deleted or promoted on merge |

## Top-level index

### AI

| Topic | File |
|---|---|
| Module architecture (A–E, code layout) | [ai/architecture.md](ai/architecture.md) |
| Generation & analysis pipeline design | [ai/pipeline_design.md](ai/pipeline_design.md) |
| Distillation strategy (future product path) | [ai/distillation_strategy.md](ai/distillation_strategy.md) |
| MVP decision log | [ai/logs/mvp_decision_log.md](ai/logs/mvp_decision_log.md) |
| AudioLDM2 transition log | [ai/logs/audioldm2_transition_log.md](ai/logs/audioldm2_transition_log.md) |

### Runbooks

| Workflow | File |
|---|---|
| Layer A smoke 1 — spring night ambient (AudioLDM2 LoRA) | [ai/runbooks/layer_a_smoke_1_spring_night.md](ai/runbooks/layer_a_smoke_1_spring_night.md) |
| Layer A smoke 2 — insect/cicada ambient (AudioLDM2 LoRA) | [ai/runbooks/layer_a_smoke_2_insects.md](ai/runbooks/layer_a_smoke_2_insects.md) |
| Layer C smoke 1 — bird events (AudioGen LoRA) | [ai/runbooks/layer_c_smoke_1_birds.md](ai/runbooks/layer_c_smoke_1_birds.md) |

### Data

| Topic | File |
|---|---|
| Data alignment & env features | [data/data_reference.md](data/data_reference.md) |
| Generation quality analysis | [data/logs/generation_quality_analysis.md](data/logs/generation_quality_analysis.md) |

### Dev Specifications

| Topic | File |
|---|---|
| DVC workflow (git+dvc, hooks, fresh-clone) | [dev_specifications/dvc_workflow.md](dev_specifications/dvc_workflow.md) |
| S3 bucket layout & DVC remote | [dev_specifications/s3_bucket_layout.md](dev_specifications/s3_bucket_layout.md) |
| Git workflow (full) | [dev_specifications/git_workflow.md](dev_specifications/git_workflow.md) |
| Model README standard | [dev_specifications/model_readme_standard.md](dev_specifications/model_readme_standard.md) |

### Setup

| Topic | File |
|---|---|
| Services & local dev setup | [setup/services.md](setup/services.md) |
| Hardware requirements | [setup/hardware.md](setup/hardware.md) |

### Operations

| Topic | File |
|---|---|
| Known data issues | [issues/known_issues.md](issues/known_issues.md) |
| Analysis test cases | [testing/analysis_test_cases.md](testing/analysis_test_cases.md) |
| Layer verification & handoff formats | [testing/layer_verification_formats.md](testing/layer_verification_formats.md) |
| Workflow diagrams | [diagrams/workflow_diagrams.md](diagrams/workflow_diagrams.md) |
