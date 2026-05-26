# DVC + S3 Workflow

DVC tracks large binary artifacts (audio clips, model checkpoints, latent
databases) so they stay out of git history. Git stores only the `.dvc` pointer
files (~100 bytes each); the actual bytes live in S3.

## Division of labor

| System | Stores | Why |
|---|---|---|
| Git | `.dvc` pointer files, `dvc.yaml`, `dvc.lock`, `params.yaml`, source code, small CSVs | Cheap, version-controlled text |
| DVC local cache (`.dvc/cache/`) | Content-addressed blobs, hardlinked to working tree | Fast local access, deduplicated across branches |
| S3 remote | Same blobs, shared across team | Persistent, multi-machine |

```
git commit  →  .dvc pointer files committed (tiny text)
               actual binary data stored in
               s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache/

git checkout <branch>  →  post-checkout hook fires
                           dvc checkout runs automatically
                           binary files swapped to match the branch's .dvc pointers
```

## What's tracked where

### DVC

| Artifact | Path | Size |
|---|---|---|
| Downloaded audio clips | `resources/site_257_bowra-dry-a/downloaded_clips/` | 43+ GB |
| Downloaded annotations | `resources/site_257_bowra-dry-a/downloaded_annotations/` | sparse CSVs |
| VAE checkpoint | `model/candidates/lucas/vae-site257-30epoch/best.pt` | 213 MB |
| Vocoder checkpoint | `model/candidates/lucas/vocoder-hifigan-site257/best.pt` | 11 MB |
| Per-clip latent database | `acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/latents/latent_clips.npy` | tens of MB |
| Weather assets | `acoustic_ai/layers/layer_b/attempts/lucas__smoke_1__curated_assets/data/weather/weather_assets/` | curated wind/rain clips |
| Event snippets | `acoustic_ai/layers/layer_c/attempts/lucas__smoke_1__audiogen_boobook/data/events/event_snippets/` | extracted annotation clips |
| Per-candidate LoRA / checkpoint binaries | `model/candidates/<member>/<run-id>/*.{pt,safetensors}` | 11 MB – 280 MB each |

### Git (not DVC)

| File | Why |
|---|---|
| `resources/site_257_bowra-dry-a/site_257_*.csv` | Small metadata |
| `acoustic_ai/layers/layer_b/attempts/lucas__smoke_1__curated_assets/data/weather/asset_index.csv` | Asset index headers |
| All `.dvc` pointer files | Pointers to DVC artifacts |
| `dvc.yaml`, `dvc.lock` | Pipeline stage definitions + lockfile |
| `params.yaml` (root) | Pipeline-stage hyperparameters |
| `model/candidates/<member>/<run-id>/{params.yaml,adapter_config.json,README.md,training_metadata.json,*.dvc}` | Candidate metadata + DVC pointers |

## Daily commands

```bash
# After switching branches or pulling — sync tracked artifacts to match commit
dvc checkout

# Run pipeline stages whose inputs have changed
dvc repro

# Push new or changed artifacts to S3
dvc push

# Pull artifacts from S3 (after cloning or on a new machine)
dvc pull

# Check what pipeline stages are out of date
dvc status

# Verify local cache and S3 are in sync
dvc status -c
```

## Adding a new tracked artifact

For an arbitrary binary that isn't a pipeline output:

```bash
dvc add path/to/large_file.pt
# creates path/to/large_file.pt.dvc — commit that pointer to git
git add path/to/large_file.pt.dvc
git commit -m "Track large_file.pt with DVC"
dvc push
```

For a checkpoint binary that **is** a pipeline output declared in `dvc.yaml`,
do **not** use `dvc add` — DVC will refuse with "overlaps with an output of
stage X". Instead refresh the lock via:

```bash
dvc commit -f <stage-name>   # records current MD5 without rerunning the stage
```

## Model-checkpoint discipline (team rule)

Per the team workflow, every candidate run lives under
`model/candidates/<member>/<run-id>/`. Promoted model slots live under
`model/production/<role>/` only after explicit validation, sign-off, and release
tagging.

Every model checkpoint folder ships with:

- `.dvc` pointer for the binary checkpoint(s)
- `README.md` — model log / model card following [model_readme_standard.md](model_readme_standard.md)

Candidate folders also ship with:

- `params.yaml` — training + inference hyperparameters
- `metrics.json` — when evals exist (TODO for current candidates)
- `training_metadata.json` — when the training script writes it (e.g. AudioGen LoRA)

DVC-add only the binary file inside the candidate folder, not the whole folder:

```bash
dvc add model/candidates/lucas/layer-X-<run-id>/adapter_model.safetensors
```

This leaves the `params.yaml` / `README.md` / `*.json` metadata in git for fast
browsing.

## Adding a new pipeline stage

Edit `dvc.yaml` to define the stage with `cmd`, `deps`, and `outs`. Then:

```bash
dvc repro          # runs only changed stages
git add dvc.yaml dvc.lock
git commit -m "Add <stage-name> pipeline stage"
dvc push
```

## Automatic git hooks

Installed once by `dvc install`. Fire without manual intervention:

| Git action | Hook | DVC action |
|---|---|---|
| `git checkout <branch>` / `git switch` | `post-checkout` | `dvc checkout` — swaps data files to match new branch |
| `git pull` / `git merge` | `post-merge` | `dvc checkout` — syncs data after incoming changes |
| `git commit` | `pre-commit` | warns if tracked data was modified but not `dvc add`-ed |
| `git push` | `pre-push` | `dvc push` — uploads new/changed artifacts to S3 |

## S3 remote

```
s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache/
```

Region `ap-southeast-2`. Profile `capstone2`. Configured once in `.dvc/config`
(already committed) — new machines don't need `dvc remote add`.

The bucket also hosts human-browsable prefixes for source data, blessed
checkpoints, and training logs:

| Prefix | Contents |
|---|---|
| `dataset/metadata/` | Small CSVs (also git-tracked) |
| `dataset/original/` | Raw A2O downloads (~125 GB FLAC pool) — DVC, cold tier |
| `dataset/training_dataset/<layer>/<stage>/` | Curated subsets (smoke / mvp / product) — DVC, hot tier |
| `dvc-cache/` | DVC content-addressed blobs (`dvc push/pull` only) |
| `release/<layer>/<run>-v<N>/` | Blessed checkpoints + metrics + audit samples |
| `logs/<layer>/<run>/<date>/` | Training logs, TB events, debug bundles |

DVC is scoped to the `dvc-cache/` prefix specifically so the opaque hash tree
doesn't pollute the human-browsable prefixes. Full bucket layout, lifecycle
rules, and `aws s3 sync` mirror commands: [s3_bucket_layout.md](s3_bucket_layout.md).

## Typical branch workflow

```bash
# Start a new experiment
git checkout -b experiment/beta-kl-0.05
# post-checkout fires → dvc checkout syncs data for this branch

# Change a hyperparameter and re-run the pipeline
vim params.yaml
dvc repro          # only re-runs stages whose inputs changed
git add .
git commit -m "experiment: higher beta KL"
# pre-commit fires → warns if any DVC-tracked file is dirty
git push
# pre-push fires → dvc push copies new checkpoint to S3

# Switch back — everything restores automatically
git checkout main
# post-checkout fires → dvc checkout restores main's best.pt, latents, etc.
```

## DVC pipeline (`dvc.yaml`)

Defines reproducible stages. `dvc repro` re-runs only stages whose deps or
params changed.

| Stage | Command | Key outputs |
|---|---|---|
| `precompute_spectrograms` | `precompute/precompute_spectrograms.py` | `data/shared/wavs/`, `data/shared/spectrograms/` |
| `train_vae` | `layers/layer_a/attempts/lucas__smoke_4__vae_baseline/train.py` | `model/candidates/lucas/vae-site257-30epoch/best.pt` |
| `precompute_latents` | `precompute/precompute_latents.py` | `data/ambient/latents/latent_clips.npy`, `latent_templates.npy` |
| `train_vocoder` | `layers/layer_a/attempts/lucas__smoke_4__vae_baseline/train_vocoder.py` | `model/candidates/lucas/vocoder-hifigan-site257/best.pt` |

Hyperparameters that affect stage reruns are tracked in root `params.yaml`.
Compare params between branches: `dvc params diff main`.

## Makefile shortcuts

```bash
make branch b=<name>   # git checkout <name> + dvc checkout
make push              # git push + dvc push
make pull              # git pull + dvc pull
make repro             # dvc repro
make diff              # git diff + dvc params diff
make status            # git status + dvc status
make ai                # start AI server locally on port 8000
```

## Fresh clone setup

On a new machine, after `git clone`:

```bash
# 1. Install DVC + S3 driver libs. Two-step is recommended because the
#    bundled 'dvc[s3]' extra triggers a long pip resolver backtracking loop
#    on macOS.
pip3 install --user dvc
pip3 install --user --upgrade boto3 s3fs aiobotocore fsspec
# Verify (both must succeed without error):
dvc --version
python3 -c "import boto3, s3fs; from fsspec.callbacks import DEFAULT_CALLBACK; print('ok')"

# 2. Make sure `dvc` is on PATH. macOS pip3 --user installs go to
#    ~/Library/Python/<ver>/bin — that directory may not be on PATH by default.
echo 'export PATH="$HOME/Library/Python/3.9/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
which dvc   # should print ~/Library/Python/3.9/bin/dvc

# 3. Configure AWS credentials.
#    Add a [capstone2] profile to ~/.aws/credentials with project IAM keys.
#    Add a [profile capstone2] block to ~/.aws/config with region=ap-southeast-2.
aws sts get-caller-identity --profile capstone2
aws s3 ls s3://eco-acoustic-data.store.adelaideuni.cloud/ --profile capstone2

# 4. S3 remote is already declared in .dvc/config — no `dvc remote add` needed.
dvc pull

# 5. Re-install git hooks (hooks live in .git/, not committed).
dvc install
```

### Troubleshooting

- **`import boto3` fails inside dvc** — you installed S3 deps into the wrong
  Python. Check `head -1 $(which dvc)` to see which interpreter dvc actually
  uses, then `<that-python> -m pip install --user boto3 s3fs aiobotocore fsspec`.
- **`cannot import name 'DEFAULT_CALLBACK' from 'fsspec.callbacks'`** — your
  `fsspec` is pinned to an old version by a transitive dep. Run
  `pip3 install --user --upgrade fsspec s3fs aiobotocore`.
- **DVC and its S3 deps live at user-site, not in `acoustic_ai/.venv`.** Git
  hooks call `dvc` and must work without venv activation.
