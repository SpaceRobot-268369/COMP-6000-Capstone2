---
name: pre_pr_checklist
description: Run before opening a pull request in this repo. Walks the pre-PR checklist — sync with main, file audit, DVC split, repo-structure index freshness, AI/model hygiene, sanity checks, branch/commit conventions. Distinguishes checks CI/hooks already hard-gate from the ones nothing automated catches. Invoke when the user is about to open a PR, asks to "send the PR", or wants to verify a branch is mergeable.
---

# Skill: Pre-PR Checklist

## Purpose

A judgment-and-gaps checklist to run before opening a PR. CI and the git
hooks already hard-gate stray binaries/secrets, DVC tracking, and
build/lint — **this skill exists to catch what automation cannot**, and to
let you catch the gated items locally before burning a red-CI round-trip.

Related automation (do not duplicate, just be aware of it):
- CI `hygiene` job — fails on tracked `.pt/.safetensors/.bin/.ckpt/.wav`,
  keys, `node_modules`, `.env*.local`, etc.
- `pre-commit` / `pre-push` git hooks — run `dvc git-hook`, blocking commits
  of files that should be DVC-tracked.
- CI `frontend` / `backend` — `npm run check`.
- CI `acoustic-ai` light check — `compileall` + FastAPI app import only
  (NOT runtime behaviour).

Mechanical commit steps live in [commit_changes](commit_changes.md); DVC
push lives in [dvc_push](dvc_push.md). This skill orchestrates; it does not
re-implement them.

---

## Legend

- 🔒 **Gated** — CI or a hook also enforces this. Running it here only saves
  a red-CI round-trip.
- ⚠️ **Ungated** — nothing automated catches this. These are the items that
  actually matter; never skip them.

---

## Checklist

Stop at the first failure, fix, then resume.

### 1. Sync with main
- [ ] `git fetch origin main`
- [ ] Merge (or rebase) latest `origin/main` into the branch.
- [ ] ⚠️ Resolve all conflicts: `git diff --name-only --diff-filter=U` is empty.
- [ ] `git diff --check` passes (no conflict markers / whitespace errors).

### 2. File audit
- [ ] `git status` inspected (staged + untracked).
- [ ] Diff is only intended files: `git diff --stat origin/main...HEAD`.
- [ ] 🔒 No git-staged binaries: `.pt` `.safetensors` `.bin` `.ckpt` `.wav`
      (except `*.wav.dvc`).
- [ ] 🔒 No credentials, keys, `.env*.local`, `.DS_Store`, local compose
      overrides, logs, debug/generated outputs.
- [ ] ⚠️ `git log --oneline origin/main..HEAD` is only intended commits.

### 3. DVC split
- [ ] 🔒 Every new/changed binary has a `.dvc` pointer (pre-commit hook gates).
- [ ] Metadata (`*.json` `*.yaml` `*.md` `*.dvc`) is in git, not DVC.
- [ ] ⚠️ `dvc push` run, then `dvc status -c` shows cache + remote in sync.
      **CI never pulls DVC** — a forgotten push silently breaks teammates.
      (See [dvc_push](dvc_push.md).)

### 4. Repo structure ⚠️ (entirely ungated — highest-value section)
- [ ] If `.claude/` files moved/added/removed → `.claude/` map in CLAUDE.md
      updated this branch.
- [ ] If a top-level dir changed → Repo layout section updated.
- [ ] If a quick-link target moved → quick-link table updated.
- [ ] Attempt/model folders follow naming + self-contained layout rules
      ([conventions.md](../context/conventions.md)).

### 5. AI / model hygiene (if `acoustic_ai/` or `model/` touched)
- [ ] ⚠️ New checkpoint folder ships `README.md` + `.dvc` pointer
      (+ `params.yaml` for candidates).
- [ ] ⚠️ Attempt `params.yaml` matches the frozen model `params.yaml` snapshot.
- [ ] ⚠️ Model README audit/experiment log updated.
- [ ] ⚠️ If `registry.yaml`, a handler, or a frontend control changed →
      `GET /layers` + the affected generation path tested, not just imported.

### 6. Sanity checks
- [ ] 🔒 Frontend/backend `npm run check` pass.
- [ ] 🔒 Python compiles + FastAPI app imports.
- [ ] ⚠️ Smoke/registry checks run with `acoustic_ai/.venv` — **never** system
      python (incompatible torch/torchaudio).
- [ ] ⚠️ Feature exercised through the real path, not only compile/import.
- [ ] ⚠️ PR body states exact commands run + known limitations — draft it to
      the repo PR standard ([draft_pr](draft_pr.md)).

### 7. Branch + commits
- [ ] Branch name `<type>/<author>/<short-desc>`.
- [ ] Not `exp/...` unless intentionally not mergeable.
- [ ] Commit subjects imperative, ≤72 chars, no issue numbers.

---

> If a 🔒 item is red in CI, fix it there — don't expand this list to
> re-audit what automation already owns. The value of this skill is the ⚠️
> items.
