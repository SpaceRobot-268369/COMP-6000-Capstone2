---
name: resolve_conflicts
description: Diagnose and resolve git conflicts (merge, rebase, cherry-pick, stash-pop) in this repo. First produces a grouped, reasoned report of every conflicted file — clustering files that conflict for the same reason — then auto-resolves trivial conflicts and pauses for the developer on complex ones, defaulting to main's side unless told otherwise. Invoke when an operation hits conflicts or the user says "resolve conflicts" / "fix the merge".
---

# Skill: Resolve Conflicts

## Purpose

Turn a wall of conflict markers into a **grouped, reasoned report** the
developer can act on, then resolve them safely. The flow is always: diagnose
→ report → resolve → verify → hand back. This is the deep-dive behind
[pre_pr_checklist](pre_pr_checklist.md) step 1 ("Resolve all conflicts").

Resolution conventions follow:
- [.claude/context/dev/git_workflow.md](../context/dev/git_workflow.md) — branch/commit rules, do-not-track table.
- [.claude/context/dev/dvc_workflow.md](../context/dev/dvc_workflow.md) — how `.dvc` / `dvc.lock` conflicts differ from source.
- [CLAUDE.md](../../CLAUDE.md) — the `.claude/` map + Repo layout single-source-of-truth rule.

This skill **resolves only**. It does not commit or push — hand off to
[commit_changes](commit_changes.md) once the working tree is clean.

---

## Hard guardrails

- **Report before touching anything.** Always present the grouped report and
  get the developer's read on the complex groups before editing files.
- **Default side = `main`.** When a conflict isn't trivial and the developer
  hasn't weighed in, prefer the version coming from `main` (the side being
  merged/rebased onto) — but surface the choice and ask first. Never silently
  pick a side on a complex conflict.
- **Never** `git checkout --ours <path>` / `--theirs <path>` wholesale on a
  file to "make it go away" without understanding both sides.
- **Never** `git merge --abort` / `git rebase --abort` without explicit
  confirmation — it discards in-progress resolution work.
- **Never** use `--no-verify` to push past hooks, and never resolve a
  delete/modify conflict by blind deletion without confirming intent.
- **Ours/theirs is swapped during rebase / cherry-pick.** In a *merge*,
  `--ours` = your branch, `--theirs` = incoming (main). In a *rebase*,
  HEAD is main's replayed-onto base, so `--ours` = main and `--theirs` =
  your commits. Always confirm direction from Step 1 before reasoning about
  sides.
- Resolved files must contain **zero** conflict markers before completing the
  operation.

---

## Pipeline

### Step 1 — Detect context

```bash
git status                           # shows operation + unmerged paths
git rev-parse --abbrev-ref HEAD      # current branch
ls -d .git/rebase-merge .git/rebase-apply .git/MERGE_HEAD .git/CHERRY_PICK_HEAD 2>/dev/null
```

Determine which operation is in flight — **merge**, **rebase**,
**cherry-pick**, or **stash pop** — and therefore which side is "ours" vs
"theirs" (see guardrail above). State this explicitly before going further;
every later reason depends on it.

### Step 2 — Inventory

```bash
git diff --name-only --diff-filter=U     # all conflicted paths
git diff --diff-filter=U --stat          # rough size of each conflict
```

Capture the full conflicted set. Also note conflict *types* surfaced by
`git status`: "both modified", "deleted by them / us", "added by both".

### Step 3 — Diagnose each file

For every conflicted path, read the conflicted hunks and work out **what
feature / change each side was making** that caused them to collide — the
*intent*, not just "both changed it". Use `git log` on the hunk if the diff
alone doesn't reveal it:

```bash
git log --oneline -3 main -- <path>          # what main was doing here
git log --oneline -3 HEAD..<branch> -- <path># what this branch was doing here
```

For each file (or group) capture three things:
1. **Conflicting features** — what main's change does vs what the branch's
   change does, and why they overlap. E.g. "main renamed the generation
   endpoint; this branch added a retry wrapper around the old name."
2. **If you keep main** — the concrete consequence for the branch's work.
   E.g. "the retry wrapper is dropped — generation calls lose their retry."
3. **If you keep branch** — the concrete consequence for main's work.
   E.g. "the endpoint rename is undone — calls hit the old, now-removed route."

A conflict is **trivial** only when one side fully subsumes the other (e.g.
pure formatting, or main's rename with no real branch logic on top) so neither
consequence loses real work. Otherwise it's **complex**.

Examples of feature-level reasons:
- "main renamed `useGeneration`→`useSoundscape`; branch added loading state to the old hook."
- "Both added a field to the generation request — main added `seed`, branch added `cell`."
- "Deleted on main (file retired); branch kept extending it (delete/modify)."
- "`dvc.lock` hash differs — both regenerated the same stage from different params."

Flag special-cased files:
- **`.dvc` / `dvc.lock`** — never hand-merge hashes; resolve by re-running
  the DVC operation or picking the correct pointer, per
  [dvc_workflow.md](../context/dev/dvc_workflow.md).
- **`CLAUDE.md` `.claude/` map / Repo layout** — both sides may have added
  structural entries; the merge must keep *both* sets of true entries, not
  one side.

### Step 4 — Group

Cluster files that conflict for the **same reason** into one group (the core
deliverable). E.g. "5 frontend components — all break on the same renamed
`useGeneration` export". Single-cause files stand alone. Tag each group
**trivial** (auto-resolvable) or **complex** (needs a decision).

### Step 5 — Present the report

Show this table **before editing anything**. Every row carries the
feature-level reason plus the consequence of each side, from Step 3:

| Group | Files | Conflicting features (main vs branch) | If keep main | If keep branch | Recommendation | Class |
|---|---|---|---|---|---|---|
| 1 | `a.tsx`, `b.tsx`, … | main renamed `foo`→`bar`; branch only reformatted callers | branch reformatting is dropped (cosmetic) | rename undone, calls break | adopt main + reapply formatting | trivial |
| 2 | `mixer.py` | main reworked gain staging; branch added a limiter on old gain path | limiter is lost | gain rework reverted | **needs your call** | complex |

- **Trivial** rows: state the resolution you'll auto-apply (neither side
  loses real work).
- **Complex** rows: state the default (main's side) and **what keeping it
  costs the branch**, then ask the developer for their intent before
  applying. List the concrete options per group.

### Step 6 — Resolve

- **Trivial groups** — auto-resolve, then `git add` each path.
- **Complex groups** — apply only the resolution the developer confirmed
  (default main's side if they defer). Hand-edit the hunks; never wholesale
  `--ours`/`--theirs` unless the developer explicitly chooses an entire side.
- **`.dvc` / `dvc.lock`** — resolve via DVC, not text editing.

### Step 7 — Verify

```bash
git diff --check                         # no leftover markers / whitespace errors
grep -rn '^<<<<<<<\|^=======\|^>>>>>>>' <resolved paths>   # belt-and-suspenders
git diff --name-only --diff-filter=U     # must be empty
```

Run the relevant sanity check for what was touched:
- frontend/backend → `npm run check`
- `acoustic_ai/` → py-compile + FastAPI app import (with `acoustic_ai/.venv`).

### Step 8 — Complete (or hand back)

Stage resolved paths and continue the in-flight operation:

```bash
git add <resolved paths>
git merge --continue        # or: git rebase --continue
                            #     git cherry-pick --continue
```

For a stash pop, just `git add` — there's nothing to "continue".

Then **stop**. Committing and pushing are the developer's call — point at
[commit_changes](commit_changes.md). Report:
- Operation + branch.
- Groups resolved, how, and which side won each.
- Anything left for the developer (deferred decisions, sanity-check results).

---

## Out of scope

- `git commit` / `git push` → [commit_changes](commit_changes.md) + the developer.
- Deciding *whether* to merge or rebase in the first place.
- Auto-`abort` of any operation — only on explicit request.
- Branch creation / renaming.
