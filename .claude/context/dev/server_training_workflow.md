# Server Training Workflow

> Part of the project [conventions](../conventions.md). This is the canonical
> doc for the **train-on-serverB → push branch → review locally → PR** loop.
> Server topology lives in [../setup/server/on_demand_ai_worker.md](../setup/server/on_demand_ai_worker.md);
> DVC commands in [dvc_workflow.md](dvc_workflow.md); stage loop in
> [dev_workflow.md](dev_workflow.md).

Use this flow whenever a training run is heavy enough to need serverB's GPU
(typically `mvp_N` and `prod_N` runs — `smoke_N` stays on the local Mac per
[dev_workflow.md § Stage-specific notes](dev_workflow.md#stage-specific-notes)).

---

## 1. Where to train on serverB

| Clone | Branch | Use for |
|---|---|---|
| `~/shiny-pikachu/` | tracks `origin/main` | **Never** train here — runs the live AI service. Never `git checkout` another branch in this tree. |
| `~/<member>/COMP-6000-Capstone2/` (e.g. `~/lucano/…`) | any | Per-member experiment clone. Switch branches, hold uncommitted work, run training. |

The DVC cache is symlinked across clones (`~/shiny-pikachu/.dvc/cache ->
~/lucano/COMP-6000-Capstone2/.dvc/cache`), so anything `dvc add`-ed on the
experiment clone is physically visible to `shiny-pikachu/` immediately.
Nothing serves a candidate checkpoint until it's wired into
`acoustic_ai/registry.yaml`, so this is safe — but be aware of it.

---

## 2. End-to-end flow

```
┌─ on serverB (~/<member>/COMP-6000-Capstone2/) ─────────────────────────┐
│  1. git fetch origin && git checkout <branch>                          │
│  2. Run training. Outputs land at conventional paths:                  │
│       acoustic_ai/layers/layer_<x>/attempts/<member>__<stage>__<slug>/ │
│       model/candidates/<member>/<stage>__<slug>/                       │
│  3. dvc add model/candidates/<member>/<stage>__<slug>/<binary>         │
│       (one `dvc add` per binary file, not the folder)                  │
│       For dvc.yaml stage outputs use `dvc commit -f <stage>` instead.  │
│  4. git status — mandatory pre-commit file audit                       │
│  5. git add <metadata + .dvc pointers>                                 │
│     git commit -m "model: <stage> <slug> — <one-line result>"          │
│  6. git push   (pre-push hook → `dvc push` uploads blobs to S3)        │
└────────────────────────────────────────────────────────────────────────┘
┌─ on local Mac ─────────────────────────────────────────────────────────┐
│  7. git fetch && git checkout <branch>                                 │
│       post-checkout hook → `dvc checkout` materialises binaries        │
│       cold cache: `dvc pull`                                           │
│  8. Audit the checkpoint. Generate showcase samples locally via        │
│     acoustic_ai/scripts/regenerate_samples.py (canonical seed 42 +     │
│     1–2 variations) per dev_workflow.md step 6.                        │
│  9. Commit the showcase samples (binaries via DVC, .dvc pointers +     │
│     metadata via git), update acoustic_ai/registry.yaml if needed,     │
│     open the PR.                                                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Rules

| Rule | Why |
|---|---|
| Train inside `~/<member>/COMP-6000-Capstone2/`, **never** in `~/shiny-pikachu/`. | `shiny-pikachu/` is pinned to `main` and runs the live service. Checking out a branch there breaks production. |
| One attempt = one folder = one branch. Don't reuse `<slug>` across attempts. | Stage tokens are fixed at creation ([conventions.md § 2.4](../conventions.md)); promotion is a new folder, never a rename. |
| Push the binary (`dvc push`) and the `.dvc` pointer (`git push`) together. | If `git push` lands but `dvc push` fails, teammates' `dvc pull` will error. The pre-push hook bundles them; if hooks are absent, run `dvc push` explicitly. |
| Always run `git status` before commit. Train scratch (`debug/`, raw logs, venv mutations) must not enter git. | Standard pre-commit audit ([git_workflow.md § Pre-commit file audit](git_workflow.md#pre-commit-file-audit)). `.gitignore` covers most cases but not all. |
| Commit checkpoint `README.md` + `params.yaml` in the **same commit** as the `.dvc` pointer. | Reviewers can't tell what a checkpoint is without them. Required per [conventions.md § 6](../conventions.md). |
| Long training runs: checkpoint to `model/candidates/<member>/<stage>__<slug>/` periodically, not only at the end. | The cost-control watchdog ([on_demand_ai_worker.md § Runtime Monitoring](../setup/server/on_demand_ai_worker.md#runtime-monitoring-cron--discord)) may auto-stop serverB. Periodic checkpoints survive an unexpected shutdown. |
| Don't `dvc gc --cloud --all-branches` from serverB while teammates have unmerged branches. | `--all-branches` only sees branches present in this clone. A partial clone can over-prune S3. Run gc from a full clone, monthly. |
| Showcase samples are generated and committed from the **local Mac**, not serverB. | Verifies the served inference path actually works on canonical seed 42 before review. |
| Don't promote (`model/production/<role>/`) from serverB as part of the training commit. | Promotion is a separate, deliberate step with sign-off ([conventions.md § 5.4](../conventions.md)). |

---

## 4. Pre-flight checklist (run once per experiment clone)

Before the first push from a fresh `~/<member>/COMP-6000-Capstone2/`:

```bash
# DVC is installed at user-site, not in the venv (hooks must work without venv activation)
dvc --version

# AWS profile resolves
aws sts get-caller-identity --profile capstone2
aws s3 ls s3://eco-acoustic-data.store.adelaideuni.cloud/ --profile capstone2

# Git hooks are installed (per-clone — they live in .git/, not committed)
dvc install
ls .git/hooks/post-checkout .git/hooks/post-merge .git/hooks/pre-commit .git/hooks/pre-push
```

If any of these fail, see [dvc_workflow.md § Fresh clone setup](dvc_workflow.md#fresh-clone-setup).

---

## 5. Cross-references

- Server topology, idle-shutdown, job queue → [../setup/server/on_demand_ai_worker.md](../setup/server/on_demand_ai_worker.md)
- Stage loop (smoke → mvp/prod) → [dev_workflow.md](dev_workflow.md)
- DVC commands, hooks, fresh-clone setup → [dvc_workflow.md](dvc_workflow.md)
- Branch naming, commit style, pre-commit audit → [git_workflow.md](git_workflow.md)
- Attempt / checkpoint naming → [../conventions.md](../conventions.md)
