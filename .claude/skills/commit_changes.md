---
name: commit_changes
description: Commit current working-tree changes to git (and DVC where appropriate) following this repo's conventions. Audits files against the do-not-track table, proposes .gitignore updates for unintended files, and raises concerns before staging anything risky. Invoke when the user asks to commit, save, or check in their work.
---

# Skill: Commit Changes (git + DVC)

## Purpose

Commit the current working-tree changes to git, and (when relevant) record DVC
artifact updates, following the conventions in:

- [CLAUDE.md](../../CLAUDE.md) — branch naming, pre-commit audit, storage rule
- [.claude/context/dev/git_workflow.md](../context/dev/git_workflow.md) — full do-not-track table, commit message rules
- [.claude/context/dev/dvc_workflow.md](../context/dev/dvc_workflow.md) — what goes to DVC vs git, `dvc commit -f <stage>` for pipeline outputs

This skill **commits only**. It does not push. Use the `dvc_push` skill to upload
DVC artifacts to S3, and `git push` is left to the user.

---

## Hard guardrails

- **Branch name must conform.** Pattern `<type>/<author>/<short-description>` with `type ∈ {feat, fix, data, model, infra, refactor, docs, exp}`. If the current branch does not match, STOP and ask the user to rename. Do not rename branches automatically.
- **Never** run `git add -A` or `git add .` — always stage explicit paths.
- **Never** commit `.env*` (except `.env.example`), `*.key`, `*.pem`, `id_rsa*`, anything matching `*secret*` / `*credential*`. Stop and ask.
- **Never** commit binaries > 1 MB to git. Route through DVC.
- **Never** use `--no-verify`, `--amend`, or any destructive git command. If the pre-commit hook fails, fix the underlying issue and create a NEW commit.
- Always show the user the staged diff summary + commit message draft before running `git commit`.

---

## Pipeline

### Step 0 — Preconditions

Run in parallel:

```bash
git rev-parse --show-toplevel        # must be the repo root
git rev-parse --abbrev-ref HEAD      # current branch
test -f dvc.yaml && echo "dvc-pipeline"
```

Validate the branch name against:

```
^(feat|fix|data|model|infra|refactor|docs|exp)/[a-z0-9]+/[a-z0-9-]+$
```

If it does not match → **STOP**, report the offending branch, and ask the user
how they'd like to proceed (rename via `git branch -m`, or override). Do not
proceed.

Read the relevant sections of `CLAUDE.md`, `git_workflow.md`, and
`dvc_workflow.md` once at the start so classification in Step 2 is grounded in
the current policy (the table may evolve).

### Step 1 — Inventory

```bash
git status --porcelain               # never -uall
git diff --stat                      # staged + unstaged summary
git diff --cached --stat
dvc status                           # working tree vs .dvc pointers
```

Capture the full list of changed/untracked paths.

### Step 2 — Classify every path

For each path, place it in exactly one bucket:

| Bucket | Examples | Action |
|---|---|---|
| **A. git-track** | source code, `*.md`, `*.yaml`, `*.json`, `*.dvc` pointer, small CSV (<1 MB) | stage in Step 5 |
| **B. dvc-track** | `*.pt`, `*.safetensors`, `*.bin`, `*.ckpt`, `*.wav`, large data dirs (`acoustic_ai/data/**`, `resources/site_257_bowra-dry-a/downloaded_*/`) | `dvc add` (free artifact) or `dvc commit -f <stage>` (pipeline output) |
| **C. ignore** | `__pycache__/`, `*.pyc`, `*.log`, `.DS_Store`, `*.venv/`, `node_modules/`, `dist/`, `build/`, `debug/`, `services/dev/postgresql/data/`, `acoustic_ai/hifigan_cache/`, `*.local`, `.env.*.local` | propose `.gitignore` update |
| **D. ambiguous / suspicious** | `.env` (no `.local`), `*.key`, `*.pem`, files matching `*secret*` / `*credential*`, unknown binaries, generated outputs in new locations | **STOP, raise concern, ask user** |

Ground truth is the do-not-track table in `git_workflow.md`. Do not classify
from memory if the table has been updated.

### Step 3 — Raise concerns

Stop and confirm with the user before any staging if any of the following hold:

- Bucket D matches (above).
- A path looks like a DVC artifact (binary, large) but has no `.dvc` pointer. Propose the right command:
  - Free artifact → `dvc add <path>`
  - Output of a stage declared in `dvc.yaml` → `dvc commit -f <stage-name>` (DVC refuses `dvc add` on these with "overlaps with an output of stage X").
- A bucket C path is not in `.gitignore` yet. Propose the exact lines to add (with a short inline comment) and whether `git rm --cached <path>` is also needed.
- Branch-scoped scratch under `.claude/context/branches/<slug>/` exists on a branch that looks ready to merge — remind the user to delete or promote.
- `dvc status` reports tracked data dirty (needs `dvc commit`).
- Any file > 1 MB is about to be staged to git.
- A structural change under `.claude/` (file added / moved / renamed / removed) is staged but the `.claude/ directory map` section in `CLAUDE.md` was not updated in the same diff — `CLAUDE.md` declares this section the single source of truth and that agents should refuse such commits. Propose the CLAUDE.md edit or stop.

### Step 4 — Apply .gitignore updates (only after explicit user OK)

```bash
# Append agreed patterns to .gitignore with a short inline comment
# For files already tracked that should now be ignored:
git rm --cached <path>
git status                           # verify clean of unintended files
```

### Step 5 — Stage

```bash
# git
git add <explicit path> <explicit path> ...

# DVC artifacts (one of):
dvc add <path>                       # free artifact
dvc commit -f <stage-name>           # pipeline output declared in dvc.yaml
git add <path>.dvc                   # and dvc.lock if a stage was updated
git status                           # show user the final staged set
```

### Step 6 — Draft commit message

Rules (from `git_workflow.md`):

- Imperative mood, present tense ("Add", "Fix", "Update")
- Subject ≤ 72 chars
- No issue numbers / task IDs in the subject (body if needed)
- Single, clear statement of what the commit does
- Body explains **why**, not **what**

Trailer:

```
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

Show the draft to the user. Wait for approval (or edits) before committing.

### Step 7 — Commit

Pass the message via HEREDOC to preserve formatting:

```bash
git commit -m "$(cat <<'EOF'
<subject>

<optional body>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
git status
```

If the pre-commit hook fails:

1. Read the hook output.
2. Fix the underlying issue (do not bypass with `--no-verify`).
3. Re-stage and run a **new** `git commit` (never `--amend`).

The DVC `pre-commit` hook will warn if tracked data was modified but not
`dvc commit`-ed — if it fires, loop back to Step 3.

### Step 8 — Report

Summarise:

- Commit SHA and subject
- Files committed (grouped: source / metadata / `.dvc` pointers)
- `.gitignore` lines added (if any)
- DVC artifacts now needing upload → suggest invoking the `dvc_push` skill
- Reminder that `git push` is the user's call (the `pre-push` hook will also fire `dvc push`)

---

## Out of scope for this skill

- `git push` — user decides when to push
- `dvc push` — separate `dvc_push` skill
- Branch creation / renaming
- Merging, rebasing, cherry-picking
- Any destructive operation
