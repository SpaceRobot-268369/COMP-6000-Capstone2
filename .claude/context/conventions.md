# Project Conventions

Single source of truth for naming, layout, and tracking rules in this repo.
Absorbs the former `attempt_naming.md`, `artifact_policy.md`, and
`model_readme_standard.md`. Workflow how-tos (git, DVC, S3 commands) remain
in their own docs — see [§ Cross-references](#cross-references).

Tracking markers used throughout this doc:

- 🟢 **git** — file lives in git history.
- 🔵 **DVC** — content lives in S3; only the `.dvc` pointer is in git.
- ⚪ **gitignored** — never committed (local-only scratch).

---

## 1. Repo structure

The whole tree from project root down to a single per-case artifact file.
Every leaf carries one of the three tracking markers above.

```
COMP-6000-Capstone2/
├── CLAUDE.md                                   🟢   agent guidance + repo/.claude index
├── AGENTS.md                                   🟢   points agents at CLAUDE.md
├── Makefile                                    🟢
├── dvc.yaml / dvc.lock                         🟢   pipeline definition + lock
├── params.yaml                                 🟢   hyperparameters for stages in dvc.yaml
│
├── frontend/                                   🟢   React + Vite UI (Docker, :5173)
├── backend/                                    🟢   Express + Postgres (Docker, :4000)
├── services/dev/                               🟢   docker-compose + db_init.sql
│
├── acoustic_ai/
│   ├── .venv/                                  ⚪   the ONLY Python interp for AI work
│   ├── server/                                 🟢   FastAPI on :8000 (registry-driven)
│   │   ├── server.py
│   │   └── registry.py
│   ├── registry.yaml                           🟢   declares which attempts the server exposes
│   ├── requirements.txt                        🟢
│   ├── scripts/                                🟢
│   │   ├── extract_expected_samples.py
│   │   └── regenerate_samples.py
│   └── layers/
│       └── layer_<X>/                          (layer_a | layer_b | layer_c | layer_d | layer_e)
│           └── attempts/
│               └── <member>__<stage>__<slug>/  ── one folder per attempt (see § Naming)
│                   ├── README.md               🟢   model card + run log (see § Model README)
│                   ├── params.yaml             🟢   per-attempt hyperparameters
│                   ├── __init__.py             🟢
│                   ├── .gitignore              🟢   *.wav, showcase PNG/JSON, dev-artifacts/* except .gitkeep
│                   │
│                   ├── code/                   🟢   ALL Python source for this attempt
│                   │   ├── handler.py          🟢   required: load() + generate(seed, **kw)
│                   │   ├── train.py            🟢   optional: training entrypoint
│                   │   ├── sample.py           🟢   optional: standalone sampling
│                   │   ├── preprocess.py       🟢   optional
│                   │   ├── dataset.py          🟢   optional
│                   │   └── <layer>_visualization.py 🟢   optional: PNG renderer + metadata bakers
│                   │
│                   ├── data/                   🔵   attempt-local derived data (per .dvc pointers)
│                   ├── precompute/             🟢   attempt-local precompute scripts (optional)
│                   │
│                   ├── expected/               ── real-audio ground truth (2–3 cases per attempt)
│                   │   └── real_<source_clip_id>/    ── one subdir per source clip
│                   │       ├── audio.wav      🔵 (via .dvc)
│                   │       ├── audio.wav.dvc  🟢   pointer → S3 blob
│                   │       ├── spectrogram.png 🟢   renders inline on GitHub PR review
│                   │       └── metadata.json   🟢   source manifest ref + audio stats
│                   │
│                   ├── showcase/               ── author-curated generated samples (any count)
│                   │   └── seed_<N>_<short_label>/   ── one subdir per seed
│                   │       ├── audio.wav            🔵 (via .dvc)
│                   │       ├── audio.wav.dvc        🟢
│                   │       ├── spectrogram.png      🔵 (via .dvc)
│                   │       ├── spectrogram.png.dvc  🟢
│                   │       ├── metadata.json        🔵 (via .dvc)
│                   │       └── metadata.json.dvc    🟢
│                   │
│                   └── dev-artifacts-self-testing/  ── ad-hoc developer scratch
│                       ├── .gitkeep           🟢   keeps the folder around
│                       └── (anything else)    ⚪   gitignored, never committed
│
├── model/
│   ├── candidates/<member>/<stage>__<slug>/    ── all current checkpoints
│   │   ├── README.md                           🟢   required (see § Model README)
│   │   ├── params.yaml                         🟢   frozen training params
│   │   ├── metrics.json                        🟢   once evals exist
│   │   ├── *.pt | *.safetensors | *.bin        🔵 (via matching .dvc files)
│   │   └── *.dvc                               🟢   pointers
│   └── production/<role>/                      ── promoted slot (currently empty — nothing promoted yet)
│
├── resources/
│   └── site_257_bowra-dry-a/                   🔵   source recordings + manifests (DVC-tracked)
│
├── script/                                     🟢   one-shot data prep & download utilities
│   ├── dataset/                                     manifest builders, segment prep, …
│   ├── download/                                    site_257 clip/annotation/event downloaders
│   └── env/                                         NASA env-feature fetcher
│
├── debug/                                      🟢   local-only diagnostics workspace + README
│
└── .claude/
    ├── settings.local.json                     🟢
    ├── commands/                               🟢   custom slash-command definitions
    ├── skills/                                 🟢   reusable agent skills
    └── context/                                🟢   project context loaded on demand
        ├── conventions.md                            ← this file
        ├── ai/                                       architecture, pipeline, runbooks, logs
        ├── data/                                     dataset alignment, env features
        ├── dev/                                      git, dvc, S3, stages, testing, diagrams
        ├── setup/                                    local/server service topology
        └── branches/                                 ephemeral per-branch scratch
```

Conventions baked into the tree above (full rules below):

- **frontend/**, **backend/**, **services/dev/** run via Docker Compose.
- **acoustic_ai/** runs natively only (Apple Silicon MPS); never `pip install` outside `acoustic_ai/.venv` (DVC + S3 deps are the documented exception — see [§ Cross-references](#cross-references)).
- **Binaries** (`.pt`, `.safetensors`, `.bin`, `.ckpt`, `.wav`) → 🔵 DVC.
- **Metadata** (`*.json`, `*.yaml`, `*.md`, `*.dvc`, `*.png` in `expected/`) → 🟢 git.
- **Generated PNGs** in `showcase/` and **all** WAVs → 🔵 DVC.
- **Source for an attempt** lives under `code/` — that's the import root (`layers.<layer>.attempts.<id>.code.handler`).

---

## 2. Naming rules

The conventions for **creating new folders or files** under any of the
structural roots above. Get these right and the registry, frontend, and
checkpoint paths all line up automatically.

### 2.1 Top-level dirs

Don't add new top-level dirs without updating CLAUDE.md's [Repo layout](../../CLAUDE.md#repo-layout) section in the same commit. Reuse the existing roots wherever possible:

| Putting | Goes under |
|---|---|
| New AI code (training, inference, server) | `acoustic_ai/` |
| One-shot data preparation script | `script/<category>/` |
| New service to run in Docker | `services/<name>/` |
| Source recordings + manifests | `resources/<site_or_collection>/` |
| Trained checkpoints | `model/candidates/<member>/<stage>__<slug>/` |
| Agent-loadable docs | `.claude/context/<area>/` |

### 2.2 Git branches

```
<type>/<author>/<short-description>
```

Types: `feat`, `fix`, `data`, `model`, `infra`, `refactor`, `docs`, `exp`.
Example: `model/lucas/layer-c-event-attempt-1`. Full commit-subject /
PR conventions live in [dev/git_workflow.md](dev/git_workflow.md).

### 2.3 Attempt folders

```
acoustic_ai/layers/layer_<X>/attempts/<member>__<stage>__<slug>/
```

Three segments separated by **double underscore** (`__`). Each segment is
`[a-z0-9_]+` so the whole string is a legal Python module name.

| Segment | Value |
|---|---|
| `<member>` | The owning developer's git username, lowercase. One folder per member per attempt — never overwrite another member's work. |
| `<stage>` | Lifecycle stage (next subsection). |
| `<slug>` | Short snake_case description of the **method or distinguishing feature**. Don't repeat the layer letter, the stage token, or the member name. Describe the method (`audioldm2_spring_night`), not the dataset version (`bowra_v3`). |

We sometimes write stage tokens with a dash in conversation ("smoke-1");
the **canonical on-disk form uses an underscore** (`smoke_1`). Same for
layer codes: `layer-a` is shorthand in prose; the folder is `layer_a`.

### 2.4 Stage tokens

| On-disk | Shorthand | Meaning |
|---|---|---|
| `smoke_N` | smoke-N | Exploratory. Tiny dataset, throwaway-OK, no quality guarantees. Used for any pre-MVP experiment, including approaches that were tried and abandoned. |
| `mvp_N` | mvp-N | Targeting the MVP demo. Trained on the full / production-shaped dataset. Must have a working `handler.py` for the registry. |
| `prod_N` | prod-N | Promoted candidate. Sign-off, validation, and a `model/production/<role>/` slot required. |

`N` is a 1-based integer unique **per (member, layer)**. It does **not**
have to be chronological — pick the next available number that isn't
already in use. Once assigned, a number never changes (renaming breaks
the registry, frontend dropdown state, and checkpoint paths).

> The **process** inside each stage (goal → filter → audit → polish →
> train → artifacts → compare → wire-up) lives in
> [dev/dev_workflow.md](dev/dev_workflow.md).

**Examples:**

| ✅ Good | ❌ Bad | Reason |
|---|---|---|
| `lucas__smoke_1__audioldm2_spring_night` | `lucas-smoke1-audioldm2` | single `-` separator, not Python-importable |
| `lucas__smoke_2__audioldm2_insects` | `layer_a_audioldm2_insects_smoke` | wrong order; layer prefix doesn't belong in the slug |
| `alice__mvp_1__audiogen_multi_species` | `alice__audiogen` | missing stage |
| `lucas__prod_1__audioldm2_bowra_v1` | `lucas/prod/audioldm2_bowra` | `/` breaks Python imports |
| `ben__smoke_3__clap_diffusion` | `ben__smoke_1__clap_diffusion` | `smoke_1` already used by ben in this layer |

### 2.5 Checkpoint folders

Each attempt's checkpoint mirrors its name:

```
model/candidates/<member>/<stage>__<slug>/    # smoke + mvp
model/production/<role>/                      # prod (after sign-off only)
```

The `<role>` slot is independent of the attempt slug — name it for the
production role the model serves (e.g. `layer_a_ambient`,
`layer_c_boobook`).

### 2.6 Artifact case dirs

Each "case" (one source clip or one generated seed) is its **own
subdirectory** with **fixed filenames** `audio.wav`, `spectrogram.png`,
`metadata.json` (plus matching `.dvc` pointers).

| Tier | Case-dir pattern | Filenames | Example |
|---|---|---|---|
| `expected/` | `real_<source_clip_id>/` | `audio.wav` + `audio.wav.dvc` + `spectrogram.png` + `metadata.json` | `real_001_5392_clip001_s000/` |
| `showcase/` | `seed_<N>_<short_label>/` (snake_case label) | `audio.wav` + `spectrogram.png` + `metadata.json`, each with `.dvc` pointer | `seed_42_baseline/`, `seed_43_variation_a/` |

Mirrors how source clips are already laid out (`audio.wav` + `meta.json`)
and keeps every case self-contained — you can `cp -r` one case dir
anywhere and it still makes sense.

### 2.7 Where the attempt name appears

Same string in **four** places — they must stay in sync:

1. **Attempt folder:** `acoustic_ai/layers/layer_<X>/attempts/<member>__<stage>__<slug>/`
2. **Checkpoint folder:** `model/candidates/<member>/<stage>__<slug>/`
3. **Registry ID** (key in `acoustic_ai/registry.yaml`):
   ```yaml
   layers:
     layer_a:
       attempts:
         lucas__smoke_1__audioldm2_spring_night:
           label: "Layer A — Spring night (smoke-1)"
   ```
4. **Frontend dropdown value** — the React dropdown sends the same string to the backend, which forwards to `POST /layers/<layer>/attempts/<attempt-id>/generate`.

---

## 3. Tracking rules (git / DVC / gitignore)

| File class | Tracked by | Notes |
|---|---|---|
| Python source, `*.yaml`, `*.json`, `*.md`, `*.dvc` pointers | 🟢 git | Plain text metadata always belongs in git. |
| `*.wav` (anywhere under an attempt) | 🔵 DVC | Audio bloats git history. WAV blob lives in S3, only the `.dvc` pointer is in git. |
| `*.png` in `expected/` | 🟢 git | Renders inline on GitHub diffs / PR review — that's the whole point of the expected tier. |
| `*.png` in `showcase/` | 🔵 DVC | Showcase churns every iteration; keeping in DVC keeps git lean. |
| `metadata.json` in `expected/` | 🟢 git | Reviewers see source provenance inline. |
| `metadata.json` in `showcase/` | 🔵 DVC | Same churn argument as showcase PNGs. |
| Checkpoint binaries (`*.pt`, `*.safetensors`, `*.bin`, `*.ckpt`) | 🔵 DVC | Always. |
| Source recordings under `resources/` | 🔵 DVC | Always. |
| `dev-artifacts-self-testing/.gitkeep` | 🟢 git | Keeps the folder around for teammates. |
| `dev-artifacts-self-testing/*` (everything else) | ⚪ gitignored | Pure local scratch. |
| `acoustic_ai/.venv/`, `__pycache__/`, OS artefacts | ⚪ gitignored | Standard ignores. |

**The asymmetry "PNG+JSON in git for expected, DVC for showcase"** is
deliberate: expected exists *to be reviewed inline*; showcase exists to
*track variations over time*.

**Per-attempt `.gitignore`** has the same shape in every attempt:

```gitignore
# Track the dev-artifacts-self-testing/ folder (via .gitkeep) but never
# commit its contents.
/dev-artifacts-self-testing/*
!/dev-artifacts-self-testing/.gitkeep

# WAVs must be DVC-tracked.
*.wav

# Showcase PNG/JSON are DVC-tracked (expected PNG/JSON go to git).
/showcase/**/*.png
/showcase/**/metadata.json
```

---

## 4. Artifact tiers

Every attempt's outputs fall into exactly one of three tiers:

| Tier | What it is | Lives at |
|---|---|---|
| **expected** | 2–3 **real-audio** ground-truth segments per attempt, extracted from the source recordings the attempt was trained on. NOT model outputs. The comparison baseline in the Dev UI. | `<attempt>/expected/real_<source_clip_id>/` |
| **showcase** | Author-curated **generated** samples (model outputs at chosen seeds) the developer wants teammates to review. | `<attempt>/showcase/seed_<N>_<short_label>/` |
| **dev-artifacts-self-testing** | Ad-hoc developer self-test runs, training-time debug spectrograms, anything experimental. Pure local scratch. | `<attempt>/dev-artifacts-self-testing/` |

### 4.1 Canonical seed

**Project-wide canonical seed is `42`** for generated artifacts (showcase
+ the live Dev UI). The canonical showcase baseline is named
`seed_42_baseline/`. Other showcase seeds use any number you think is
worth showing.

The canonical seed does **not** apply to `expected/` — expected samples
are real recordings, not model output, and don't have a seed.

### 4.2 PNG metadata baking

Every `spectrogram.png` carries **two redundant** metadata channels:

1. **Visible overlay** — small header/subline/footer drawn inside the spectrogram axes, with a translucent backdrop. Survives screenshots; renders inline on GitHub.
2. **PNG `tEXt` chunks** — lossless key/value pairs written after `savefig` via Pillow. Pixels unchanged. Readable via `exiftool` or `PIL.Image.open(p).text`.

Both mirror the fields in `metadata.json`. The JSON sidecar remains the
source of truth; PNG metadata is a convenience layer so tools and humans
can read provenance without the sidecar.

Builders live in each attempt's `code/<layer>_visualization.py`:

- `build_expected_overlay` / `build_expected_png_text`
- `build_showcase_overlay` / `build_showcase_png_text`

### 4.3 Metadata JSON contract

**Expected (real-audio) JSON.** Produced by
`acoustic_ai/scripts/extract_expected_samples.py`. Carries traceability
back to the source manifest, not generation parameters:

```json
{
  "tier": "expected",
  "source": "real_audio",
  "source_kind": "clip_dir | webm_slice",
  "source_clip_id": "001_5392_clip001_s000",
  "source_manifest": "resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv",
  "selection_reason": "canonical 2019 spring-night exemplar (first manifest row)",
  "audio": { "sample_rate": 16000, "duration_s": 13.1, "rms": ..., "peak": ... },
  "source_metadata": {
    "caption": "spring night, ambient soundscape, …",
    "recording_date": "2019-09-01",
    "diel_bin": "night", "season": "spring",
    "env": { ... }
  }
}
```

No `seed`, no `checkpoint`, no `handler_git_sha` — those don't apply to
ground truth.

**Showcase / generated JSON.** Produced by
`acoustic_ai/scripts/regenerate_samples.py`. The handler's `generate()`
dict is dumped verbatim, then enriched with traceability:

```json
{
  "tier": "showcase",
  "showcase_label": "variation_a",
  "seed": 43,
  "prompt": "...",
  "checkpoint": "model/candidates/lucas/layer-a-audioldm2-raw-smoke",
  "checkpoint_dvc_hash": "<md5 from .dvc pointer>",
  "audio": { "sample_rate": 16000, "duration_s": 10.0, "rms": ..., "peak": ... },
  "generated_at": "2026-05-27T12:00:00Z",
  "handler_git_sha": "<short SHA>"
}
```

`checkpoint_dvc_hash` + `handler_git_sha` let reviewers tell at a glance
whether a showcase sample is stale relative to its source.

### 4.4 When to re-extract / regenerate

**Re-extract `expected/`** — rare, only when the source-manifest
selection changes:

```bash
# Edit PICKS in extract_expected_samples.py, then:
./acoustic_ai/.venv/bin/python acoustic_ai/scripts/extract_expected_samples.py

ATT=acoustic_ai/layers/<layer>/attempts/<id>
dvc add  $ATT/expected/*/audio.wav
git add  $ATT/expected/
git commit -m "data: refresh <attempt> expected samples"
git push && dvc push
```

**Regenerate showcase samples** — whenever the handler, params, or
checkpoint changes:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/scripts/regenerate_samples.py \
    <layer> <attempt> --seed 7 --label low_noise
# script prints the exact dvc add / git add commands to run.
```

Not enforced via pre-push hook or CI; the team relies on `handler_git_sha`
in the JSON for after-the-fact spotting of stale showcase samples.

### 4.5 Storage discipline

- **Expected PNG + JSON are the only artefact files that go into git.** Every WAV, every showcase PNG/JSON, every dev-artifact is either DVC or gitignored.
- **Prune showcase samples liberally.** When a smoke attempt is superseded, delete its showcase folder in the same PR that supersedes it.
- **DVC garbage collection** (monthly, or before release):
  ```bash
  dvc gc --cloud --all-branches --all-tags
  ```
  Removes S3 objects unreferenced by any branch tip. Don't run on a shallow clone — you'll over-prune.

### 4.6 What does NOT belong under the artifact tiers

- Training-time loss curves, gradient norms, validation outputs across epochs → keep in `<attempt>/debug/` (gitignored) or attach to wandb/tensorboard.
- Source recordings / training data → those live under `<attempt>/data/` (DVC-tracked).
- Generated audio that doesn't pair with a `metadata.json` — if it can't be reproduced, it shouldn't be a showcase sample.

### 4.7 Placeholder layers

Layers whose registry status is `placeholder` (currently `layer_b`,
`layer_d`, `layer_e`) leave `expected/` empty — they either produce no
audio (`layer_e` outputs detector JSON), use curated assets that *are*
the expected output (`layer_b`), or consume other layers' output
(`layer_d`). Document this in the attempt's `README.md`.

---

## 5. Per-attempt internals

Each attempt is **self-contained** — no shared `common/` folder, no
shared preprocess scripts, no shared data caches. Members work
independently; reproducibility shouldn't depend on files outside the
attempt's own folder.

Two attempts that share a method (e.g. AudioLDM2 spring-night vs.
AudioLDM2 insects) **duplicate** the training/sampling code rather than
import from each other. Each evolves independently; the duplication is
intentional.

### 5.1 Required: `handler.py`

The registry-facing interface that the FastAPI server calls. Minimal
contract:

```python
# layers/layer_<X>/attempts/<id>/code/handler.py
from pathlib import Path

def load(checkpoint_dir: Path, params: dict) -> object:
    """One-time load. Return whatever generate() needs (model, pipeline, …)."""

def generate(state, seed: int | None, **runtime_params) -> dict:
    """Return {'wav_bytes': bytes, 'metadata': dict, 'mel_db': np.ndarray|None}."""
```

`state` is the value returned by `load()`. The server caches it per
attempt-id so `load()` runs once.

### 5.2 The `code/` import root

The handler is loaded by the server as
`layers.<layer>.attempts.<id>.code.handler`. Keep `code/` as the import
root for an attempt's Python sources.

### 5.3 Pipeline vs. attempt hyperparameters

- **Root `params.yaml`** → only contains hyperparameters for stages declared in `dvc.yaml`. Changes here trigger `dvc repro` re-runs.
- **`<attempt>/params.yaml`** → per-attempt experiment hyperparameters, sectioned `training:` and `inference:`.
- **`model/candidates/<member>/<stage>__<slug>/params.yaml`** → frozen snapshot of the params used to train the checkpoint (matches the attempt's `params.yaml` at training time).

### 5.4 Promoting an attempt

Lifecycle moves are explicit folder copies, never renames:

- **`smoke_N` → `mvp_N`:** create a new `<member>__mvp_N__<slug>/`. Don't rename the smoke folder — it stays as a historical record.
- **`mvp_N` → `prod_N`:** create `<member>__prod_N__<slug>/` and, in the same PR, promote the checkpoint to `model/production/<role>/` with sign-off recorded in the README audit section.

The registry's `default:` key for a layer is what the frontend dropdown
selects on first load. Bumping it is part of promotion.

---

## 6. Model checkpoint README

Every trained model folder must include a `README.md` that acts as the
durable model log and lightweight model card. Applies to:

- `model/candidates/<member>/<stage>__<slug>/`
- `model/production/<role>/`

The README is git-tracked metadata. Checkpoint binaries remain DVC-tracked.

### 6.1 Purpose

Let a developer understand what the model is, why it exists, when it was
trained or promoted, which data and settings produced it, and what is
known about its behaviour — **without loading the binary**.

Don't use the README as the full hyperparameter source of truth. Store
detailed candidate hyperparameters in `params.yaml` and summarise only
the important values in the README.

### 6.2 Audit rule

The `Results analysis / audit` section must be **present but empty by
default**. Do not invent audit conclusions. Fill it only after developers
provide evaluation notes, listening-test results, metrics, screenshots,
or review findings.

If no audit has been done, leave the section as:

```markdown
## Results analysis / audit

_Empty until developer evaluation notes are provided._
```

### 6.3 Required sections

```markdown
# <model-folder-name>

## Summary

- Owner:
- Layer / role:
- Status: candidate | production | deprecated
- Base model:
- Source candidate:        <!-- production only; omit for candidates -->
- Trained at:
- Promoted at:             <!-- production only; omit for candidates -->

## Purpose / hypothesis

<!-- Why this model exists, what behaviour it is testing or serving,
     and what success would look like. -->

## Dataset / inputs

- Dataset:
- Source clips / manifests:
- Filtering or preprocessing:
- Known data caveats:

## Training or promotion context

- Training command:
- Code branch / commit:
- Hardware:
- Runtime:
- Important settings:

## Artifacts

- Checkpoint binaries:
- DVC pointer files:
- Params:
- Metrics:
- Sample outputs:
  - Expected: `<attempt>/expected/real_<clip_id>/{spectrogram.png, metadata.json, audio.wav.dvc}` — one subdir per source clip.
  - Showcase: `<attempt>/showcase/seed_<N>_<label>/{audio.wav.dvc, spectrogram.png.dvc, metadata.json.dvc}` — one subdir per seed; all three DVC-tracked.
- Related runbook or log:

## Results / metrics

<!-- Objective metrics, smoke-test outputs, sample paths, or "Not evaluated yet". -->

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

<!-- Failure modes, inappropriate use cases, unresolved issues. -->

## Follow-up actions

<!-- Next evals, fixes, comparisons, promotion tasks, or cleanup. -->
```

### 6.4 Notes

- Keep entries factual and timestamped when possible.
- Link to runbooks, metrics, sample outputs, and issue logs instead of copying long analysis into the README.
- If a checkpoint is deprecated, keep its README and explain why it should not be used.

---

## Cross-references

Workflow how-tos (not naming/policy) live in separate docs:

- [dev/git_workflow.md](dev/git_workflow.md) — branch naming details, commit subject style, pre-commit file audit, "do not track in git" categories.
- [dev/dvc_workflow.md](dev/dvc_workflow.md) — `dvc add` / `dvc pull` / `dvc install`, daily/fresh-clone discipline, candidate vs. production checkpoint handling, branch-switching UX (`dvc install` post-checkout hook).
- [dev/s3_bucket_layout.md](dev/s3_bucket_layout.md) — `s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache/` structure and prefixes.
- [ai/architecture.md](ai/architecture.md) — AI module internals (server, layers, registry).
- [setup/local/services.md](setup/local/services.md) — local service topology, ports, env vars.
- [../../CLAUDE.md](../../CLAUDE.md) — repo-level orientation; canonical home for the storage rule (what lives under `.claude/`), the Python-environment rule, and the Layer A dev-generation contract.
