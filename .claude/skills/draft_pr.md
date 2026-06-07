---
name: draft_pr
description: Draft a pull-request description to this repo's PR standard. Inspects the branch diff, picks the sections that apply (always-on summary/scope/testing/limitations, plus conditional blocks for AI/model, generation, frontend/backend, and docs/structure changes), and produces a ready-to-paste PR body. Enforces the showcase-cases policy (default 3, justify more) and the "spectrograms inline, audio via dvc pull" rule. Invoke when the user asks to write/draft a PR description, "open the PR", or prep the PR body. This drafts the body only — mergeability gating lives in pre_pr_checklist.
---

# Skill: Draft PR (repo PR standard)

## Purpose

Produce a pull-request **description** that conforms to this repo's PR
standard. The body is assembled from a small set of **always-on** sections
plus **conditional** blocks chosen by what the branch diff actually touches.

This skill **drafts the body only.** It does not gate mergeability (that is
[pre_pr_checklist](pre_pr_checklist.md)), commit ([commit_changes](commit_changes.md)),
or push DVC artifacts ([dvc_push](dvc_push.md)). Run `pre_pr_checklist` before
or alongside this; a clean checklist is what lets several sections below be
written truthfully (e.g. "DVC pushed", "checks green").

Ground the writeup in:
- [CLAUDE.md](../../CLAUDE.md) — system overview, attempt/checkpoint rules, generation contracts
- [conventions.md](../context/conventions.md) — attempt naming, artifact tiers, sample tiers
- [git_workflow.md](../context/dev/git_workflow.md) — branch/commit conventions

---

## Hard guardrails

- **Draft, don't open.** Output the PR body as text (and offer the `gh pr create`
  command); never push or open the PR unless the user explicitly says so.
- **No invented evidence.** Only claim a test ran, a check passed, or DVC was
  pushed if it actually happened in this session or the user confirms it.
  Unverified items go under *Known limitations / not yet done*, not *Testing*.
- **Audio is never inlined as playable.** WAVs are DVC-tracked and do not render
  on GitHub. Always inline the `spectrogram.png` and give the reviewer the
  `dvc pull <path>` command to listen. Never imply the reviewer can hear audio
  from the PR page.
- **Showcase cap is 3 by default.** If the branch ships more than 3 showcase
  cases, STOP and ask the author for a one-line justification; include it in the
  PR body. Do not silently list more than 3.
- **Testing items are mandatory for AI/model + generation PRs.** When block A or
  block B fires, the body must list concrete, itemized test cases/items (block A
  *Test cases / items*; paired with expected vs. sample results). An empty or
  hand-wavy testing section is not acceptable for these PRs. For all other PRs,
  the always-on *How tested* section must still be filled — if nothing was
  tested, say so explicitly there; never leave it blank.
- **Sync with main first — enforced.** Before drafting, the working branch MUST
  have the latest `origin/main` merged in. Run the sync (Pipeline Step 0). **On
  any merge conflict, STOP — do not auto-resolve.** Surface the conflicting paths
  to the dev/author and let them resolve; only resume once the merge is clean.
  Drafting against an un-synced branch is not allowed (the diff and the body
  would not match what actually merges).
- **No secrets / paths-to-local-only files** in the body.

---

## The standard

A PR body is: **always-on sections**, then **whichever conditional blocks the
diff triggers**, in this order.

### Always-on (every PR)

1. **Summary** — what this PR does and **why** (the why matters more than the what).
2. **Scope of change** — bullet list of the areas touched (frontend / backend /
   `acoustic_ai/` / `model/` / docs / infra), one line each.
3. **Changed files (tree)** — the modified-file structure as an **ASCII directory
   tree**, so the reviewer grasps the shape without opening the diff. Generate
   from `git diff --name-status origin/main...HEAD`, collapse shared parent dirs,
   and tag each leaf with its status — `A` added, `M` modified, `D` deleted,
   `R` renamed — plus a short inline "what changed". Append a `(generated)` tag
   to mechanical entries (sample artifacts, `.dvc` pointers, lockfiles) so they
   aren't mistaken for substantive code. For very large diffs, collapse
   repetitive siblings into one line with a count rather than listing each.
   Wrap it in a fenced code block so the tree renders monospaced. Shape:

   ````
   ```
   acoustic_ai/layers/layer_c/attempts/lucas__smoke_1__boobook/
   ├── train.py            M  add early-stop on val loss
   ├── handler.py          M  wire new checkpoint path
   ├── README.md           M  experiment log update
   └── showcase/
       └── seed_42_boobook/
           ├── spectrogram.png   A  (generated)
           └── audio.wav.dvc     A  (generated)
   model/candidates/lucas/smoke-1__boobook/
   ├── README.md           A  checkpoint card
   ├── params.yaml         A  frozen training params
   └── adapter.safetensors.dvc  A  (generated)
   CLAUDE.md               M  layer_c status → smoke-1 ✓
   ```
   ````
4. **Review guide** — how to review this *from the description*: **start here**
   (the entry point / key file), **then** (what deserves real scrutiny), and
   **safe to skim** (boilerplate, regenerated artifacts, pointers). For AI/model
   work, point at the spectrograms + `dvc pull` commands as the review surface so
   the reviewer can judge results without reading training code.
5. **How tested** — exact commands run + outcome. AI work: state the interpreter
   was `acoustic_ai/.venv` (never system python). If something was *not* tested,
   say so here, plainly.
6. **Known limitations / not yet done** — caveats, follow-ups, anything the
   reviewer should not assume works.
7. **Author self-check** — checkboxes the author confirms so the reviewer can
   trust the gated/automated work instead of re-verifying it:
   `[ ]` `pre_pr_checklist` run · `[ ]` `dvc push` done and `dvc status -c` in
   sync (when DVC artifacts changed) · `[ ]` attempt/checkpoint naming +
   self-contained layout follow [conventions.md](../context/conventions.md) ·
   `[ ]` CLAUDE.md structure index updated (when `.claude/` or a top-level dir
   changed). Only tick what is actually true.

### Conditional blocks (include only when triggered)

Decide triggers from the diff (Pipeline Step 1). Skip a block entirely if its
trigger is absent — do not emit empty headers.

#### A. AI / model — trigger: `acoustic_ai/layers/**` or `model/**` changed
- **Attempt + checkpoint identity** — the `<member>__<stage>__<slug>` attempt
  folder and its matching `model/candidates/<member>/<stage>__<slug>/`; name the
  **stage transition** (smoke-N → mvp-N → prod-N).
- **Base + adapter** — base model (e.g. `cvssp/audioldm2`, `facebook/audiogen-medium`)
  and what LoRA/adapter this trains or serves; for bank attempts, the
  `(season, diel)` cells covered.
- **Test cases / items** — what was exercised, as a list.
- **Expected results (real audio)** — the `expected/real_<clip_id>/` ground-truth
  cases: inline each `spectrogram.png`; link the WAV with its `dvc pull` command.
- **Sample results (model-produced)** — the `showcase/seed_<N>_<label>/` cases
  (**≤3 by default**, see guardrail): inline each `spectrogram.png`, give the
  `dvc pull` command for the WAV, and the reproduce recipe (canonical seed `42`,
  the cell, and the `acoustic_ai/scripts/regenerate_samples.py` invocation).
- **Params + log links** — link the frozen `model/candidates/.../params.yaml`
  and the checkpoint `README.md` (the required experiment log); cite
  `metrics.json` if evals exist.
- **DVC state** — confirm `dvc push` ran and `dvc status -c` is in sync. CI never
  pulls DVC; without this the reviewer cannot fetch the artifacts.
- **Registry / promotion** — if `registry.yaml` changed (new attempt on
  `GET /layers`) or a candidate is promoted to `model/production/<role>/`, say so,
  and include a **caveats** subsection for any promotion (precedent: layer_a
  prod-1 was "promoted with documented caveats").

#### B. Generation — trigger: generation path / prompt-parser / layer contracts changed
- **Approach per layer** — state generative-based vs retrieval-based **for each
  layer touched** (A generative AudioLDM2 LoRA, B retrieval/curated assets,
  C generative AudioGen LoRA); a single PR may mix both.
- **Roadmap** — the path taken, including **dead-ends** (what was tried and
  dropped, not just what shipped) when there were multiple attempts.
- **Showcases** — same ≤3 rule and inline-spectrogram / `dvc pull` rule as block A.
- **Contract changes** — if prompt-parser or Layer A/B/C contracts moved, link
  [prompt_parser_policy.md](../context/ai/prompt_parser_policy.md) and name the
  fields that changed.

#### C. Frontend / backend — trigger: `frontend/**` or `backend/**` changed
- **User-visible behaviour** — what changes for the user; for UI, attach a
  screenshot.
- **Checks** — `npm run check` result (per affected package).
- **API surface** — new/changed endpoints (method + path + shape), if any.

#### D. Docs / structure — trigger: `.claude/**`, `CLAUDE.md`, or a top-level dir changed
- **Structure index freshness** — confirm the `.claude/` map / Repo layout /
  quick-link table in `CLAUDE.md` were updated in the same branch (it is the
  single source of truth; stale-index PRs should be refused).

---

## Pipeline

### Step 0 — Sync with main (blocking gate)
Pull the latest remote main and merge it into the working branch **before**
anything else. This is a hard gate, not a courtesy.
```bash
git fetch origin main
git merge origin/main            # merge latest main into the working branch
git diff --name-only --diff-filter=U   # conflicted paths (must be empty)
git diff --check                 # no conflict markers / whitespace errors
```
- If `git merge` reports conflicts → **STOP.** Do **not** auto-resolve. List the
  conflicting paths and hand them to the dev/author to resolve. Resume only once
  the merge is clean (`--diff-filter=U` empty, `git diff --check` clean).
- This mirrors `pre_pr_checklist` §1; running it here guarantees the diff and
  body reflect what will actually merge.

### Step 1 — Inventory the diff
```bash
git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
git log --oneline origin/main..HEAD
```
Classify the changed paths into the trigger buckets above (A/B/C/D). Record
which conditional blocks fire.

### Step 2 — Gather evidence (only what's true)
- For AI/model: locate the attempt folder, the `expected/` and `showcase/`
  tiers, `params.yaml`, the checkpoint `README.md`, and confirm DVC state with
  `dvc status -c` (do not push here — that's `dvc_push`).
- For frontend/backend: note any `npm run check` runs from this session.
- Anything not actually verified → it goes under *Known limitations*, not
  *Testing*.

### Step 3 — Showcase gate
If any triggered block lists > 3 showcase cases → STOP, ask the author for a
one-line justification, and fold it into the body. Otherwise default to ≤3.

### Step 4 — Assemble
Write always-on sections, then the triggered conditional blocks in A→D order.
Drop blocks with no trigger. Inline `spectrogram.png` files; for every WAV give
the `dvc pull <path>` command. Keep it scannable — bullets over prose.

### Step 5 — Output
Show the full Markdown body. Then offer (do not auto-run):
```bash
gh pr create --title "<title>" --body "$(cat <<'EOF'
<body>
EOF
)"
```
Title follows the commit-subject rules (imperative, ≤72 chars, no issue numbers).

---

## Out of scope for this skill

- Mergeability gating → [pre_pr_checklist](pre_pr_checklist.md)
- Committing → [commit_changes](commit_changes.md)
- Pushing DVC artifacts → [dvc_push](dvc_push.md)
- Actually opening/merging the PR (user decides)
- Branch creation / renaming
