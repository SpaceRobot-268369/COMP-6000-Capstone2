# Git Workflow & Standards

## Branch Naming

All branches must follow this pattern:

```
<type>/<author>/<short-description>
```

### Fields

- **`<author>`** — Your name in lowercase (e.g., `lucas`, `alex`)
- **`<type>`** — One of the type prefixes below
- **`<short-description>`** — Lowercase, hyphen-separated, max ~4 words

### Type Prefixes

| Type | When to use |
|------|------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `data` | Data pipeline changes (scripts, manifests, DVC stages) |
| `model` | Model architecture, training, or checkpoint changes |
| `infra` | Docker, CI, server config changes |
| `refactor` | Code restructure without behaviour change |
| `docs` | Documentation only |
| `exp` | Throwaway experiments (will not be merged to main) |

### Examples

```
feat/lucas/ambient-retrieval-endpoint
fix/lucas/vocoder-resampling-bug
data/alex/birdnet-annotation-index
model/lucas/vae-beta-annealing
infra/alex/docker-compose-ai-server
exp/lucas/latent-diffusion-prototype
```

---

## Commit Messages

Use the **imperative mood, present tense**. Keep the subject line under 72 characters.

### Examples

```
Add ambient retrieval function to inference.py
Fix days_since_rain UTC/AEST off-by-one error
Train ecoacoustic HiFi-GAN on site 257 clips
Update docker-compose to expose AI server port
```

### Rules

- **Do NOT** reference issue numbers or internal task IDs in the subject line — put those in the body if needed
- Subject line should be a single, clear statement of what the commit does
- Use imperative tense: "Add", "Fix", "Update", not "Added", "Fixed", "Updated"

---

## Do not track in git

Source of truth is `.gitignore` at the repo root. The table below summarises what stays out of git and where it lives instead.

| Category | Examples | Where it lives instead |
|---|---|---|
| Model binaries | `model/**/*.{pt,safetensors,bin,ckpt}` | DVC (S3) |
| Large data artifacts | `acoustic_ai/data/shared/wavs/`, `…/spectrograms/`, `…/module_b/weather_assets/`, `…/module_c/event_snippets/`, `…/module_c/birdnet_labels/`, `acoustic_ai/data/ambient/latents/` | DVC (S3) |
| Raw / generated resources | `resources/**` (allow-listed: site 257 manifest CSVs, `site_257_all_items.json`, `all_items_annotation/README.md` only) | DVC or local-only |
| Audio outputs | `*.wav`, `debug/`, `acoustic_ai/=*` | local-only / DVC if curated |
| Python env & caches | `acoustic_ai/.venv/`, `acoustic_ai/venv/`, `acoustic_ai/.venv-*/`, `**/__pycache__/`, `*.pyc`, `*.egg-info/`, `acoustic_ai/hifigan_cache/` | local-only |
| Node deps & locks | `node_modules/`, `package-lock.json` | local-only |
| Build output | `dist/`, `build/`, `out/`, `.next/`, `coverage/` | local-only |
| DB volume | `services/dev/postgresql/data/` | local Docker volume |
| Env files | `*.local`, `.env.*.local` (keep `.env.example`) | local-only |
| OS / editor artefacts | `.DS_Store`, `Thumbs.db`, `.idea/`, `.vscode/`, `*.swp` | local-only |
| Claude worktrees | `/.claude/worktrees` | local-only |
| Logs | `*.log`, `npm-debug.log*`, `yarn-*.log*` | local-only |

### Rules of thumb

- Binaries (`.pt`, `.safetensors`, `.bin`, `.ckpt`, `.wav`) → **DVC, never git**. Metadata (`*.json`, `*.yaml`, `*.md`, `*.dvc`) → git.
- Anything under `resources/` is ignored **by default** — add an explicit `!` allow-list entry in `.gitignore` if a small metadata file must be tracked.
- Branch-scoped scratch in `.claude/context/branches/<branch-slug>/` is committed but ephemeral — delete or promote before the merge PR.

### Pre-commit file audit

Mandatory before every commit:

1. Run `git status`.
2. If unintended files appear (binaries, generated outputs, credentials, OS artefacts), don't commit.
3. Add them to `.gitignore` (and `git rm --cached <path>` if already tracked).
4. Verify `git status` is clean of unintended files.

Large binaries never go to git — use DVC.

---

## Author

Lucas Tao

**Last updated:** 2026-05-13
