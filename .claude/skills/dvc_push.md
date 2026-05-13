---
name: dvc_push
description: Push new or changed DVC artifacts in this repo to the S3 remote (eco-acoustic-data bucket, capstone2 profile). Verifies preconditions, shows what will be pushed, asks for confirmation on large pushes, and verifies sync afterwards. Invoke when the user asks to dvc push, upload artifacts, or sync to S3.
---

# Skill: DVC Push to S3

## Purpose

Upload DVC-tracked artifacts from the local cache to the project S3 remote:

```
s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache/
```

(region `ap-southeast-2`, AWS profile `capstone2`).

See [.claude/context/dev/dvc_workflow.md](../context/dev/dvc_workflow.md) for the
full workflow and [.claude/context/dev/s3_bucket_layout.md](../context/dev/s3_bucket_layout.md)
for the bucket layout.

This skill **pushes only**. Committing changes is the job of the `commit_changes` skill.

---

## Hard guardrails

- **Never** run `dvc push --force`, `dvc push --remote <other>`, or `dvc gc` without an explicit user instruction.
- **Never** modify `.dvc/config` or AWS profiles.
- **Refuse to push** if `dvc status` (working tree) is dirty — that means there are uncommitted DVC changes; route the user to the `commit_changes` skill first.
- **Refuse to push** if `aws sts get-caller-identity --profile capstone2` fails — surface the error verbatim, do not retry blindly.
- For large pushes (> 1 GB total **or** > 50 objects), require explicit user confirmation before running `dvc push`.
- Surface throttling / 403 / network errors verbatim. Do not silently retry.

---

## Pipeline

### Step 0 — Preconditions

Run in parallel:

```bash
which dvc                                                # must resolve (user-site, not venv)
git rev-parse --show-toplevel                            # must be the repo root
grep -A1 "remote" .dvc/config | head -20                 # confirm remote is declared
aws sts get-caller-identity --profile capstone2          # AWS sanity
```

If `aws sts get-caller-identity` fails → **STOP**. Report the error. Likely
causes: missing `[capstone2]` profile in `~/.aws/credentials`, missing region in
`~/.aws/config`, expired credentials. Refer the user to the "Fresh clone setup"
section of `dvc_workflow.md`.

If `which dvc` resolves into a venv (e.g. `acoustic_ai/.venv/bin/dvc`) → warn
the user: DVC is supposed to live at user-site so git hooks work without venv
activation. Ask whether to proceed anyway.

### Step 1 — Pre-push status

```bash
dvc status            # working tree vs .dvc pointers
dvc status -c         # local cache vs S3 — this is what will actually push
```

- If `dvc status` reports anything dirty → **STOP**. There are uncommitted DVC
  changes. Tell the user to run `commit_changes` first (which will `dvc add` /
  `dvc commit -f <stage>` as appropriate), then come back.
- Parse `dvc status -c` to extract the list of objects/paths that are local but
  not on remote. This is the push set.

### Step 2 — Size the push, decide confirmation level

Estimate the push set size. Quickest heuristic: for each path in the push set,
resolve the cache object via `dvc data status` or read the path's size from the
working tree (DVC hard-links cache objects to the working tree, so working-tree
size ≈ cache size).

```bash
# rough size of all paths in the push set
du -sh <path1> <path2> ...
```

Confirmation policy:

| Push set | Action |
|---|---|
| ≤ 100 MB **and** ≤ 10 objects | Announce in one line, proceed |
| > 100 MB **or** > 10 objects, and ≤ 1 GB and ≤ 50 objects | Show the list + total size, proceed unless user objects |
| > 1 GB **or** > 50 objects | **Require explicit user confirmation.** Show paths + total size + estimated objects, wait for "yes" |

### Step 3 — Push

```bash
dvc push                  # foreground; can be long-running for large artifacts
```

Run as a foreground bash call with a generous timeout (up to 10 min) for small
pushes. For large pushes (Step 2 said > 1 GB), run with
`run_in_background: true` and let the user know they will be notified when it
completes; do not poll.

Watch the output for:

- `403 Forbidden` → credentials/permissions issue, surface verbatim
- `SlowDown` / `RequestTimeTooSkewed` → AWS throttling or clock skew, surface verbatim
- Network failures → surface verbatim, do not retry blindly

### Step 4 — Verify

```bash
dvc status -c
```

Expected: `Cache and remote are in sync.`

If still divergent → list the paths that failed, surface the relevant error
lines from Step 3, stop. Do not loop or retry without user direction.

### Step 5 — Report

Summarise:

- Number of objects pushed and approximate total size
- Paths pushed (grouped: model checkpoints / data artifacts / latents / etc.)
- Final sync state (`dvc status -c` output)
- If local git commits aren't on the remote yet, remind the user that
  `git push` will also fire the DVC `pre-push` hook (which will dvc-push again —
  a no-op now).

---

## Out of scope for this skill

- Committing changes (use `commit_changes` skill)
- `dvc pull` from S3
- `dvc gc` or any cache cleanup
- Changing remotes or AWS profiles
- Mirroring to / from the human-browsable S3 prefixes (`dataset/`, `release/`, `logs/`) — that's `aws s3 sync` territory, see [s3_bucket_layout.md](../context/dev/s3_bucket_layout.md)
