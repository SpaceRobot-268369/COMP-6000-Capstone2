# S3 Bucket Layout — `eco-acoustic-data.store.adelaideuni.cloud`

This bucket is the shared DVC remote and human-browsable artifact store for the
Capstone 2 project. Region: `ap-southeast-2`.

## Top-level prefixes

```
s3://eco-acoustic-data.store.adelaideuni.cloud/
│
├── dataset/                                          ← source-of-truth data (DVC-tracked from repo)
│   │
│   ├── metadata/                                     ← small CSVs, also git-tracked
│   │   ├── site_257_all_items.csv                    (full archive index, ~5 MB)
│   │   ├── site_257_filtered_items.csv               (MVP sample: 287 recordings)
│   │   ├── site_257_env_data.csv                     (NASA env join)
│   │   └── site_257_training_manifest.csv
│   │
│   ├── original/                                     ← raw downloads from A2O / NASA / etc.
│   │   ├── site_257_bowra-dry-a/
│   │   │   ├── downloaded_clips/                     (~125 GB FLAC pool)
│   │   │   └── downloaded_annotations/               (BirdNET CSVs)
│   │   └── <future_site>/...
│   │
│   └── training_dataset/                             ← curated subsets per layer / per stage
│       │
│       ├── layer-a/
│       │   ├── smoke-1-spring-night/                 (small, manually audited)
│       │   │   ├── manifest.csv
│       │   │   └── segments/
│       │   ├── smoke-2-insects/                      (small, manually audited)
│       │   │   ├── manifest.csv
│       │   │   └── segments/
│       │   ├── mvp/                                  (larger, future)
│       │   │   ├── manifest.csv
│       │   │   └── segments/
│       │   └── product/                              (largest, future)
│       │       ├── manifest.csv
│       │       └── segments/
│       │
│       ├── layer-b/
│       │   ├── weather-assets-smoke/                 (curated wind/rain clips)
│       │   ├── weather-assets-mvp/
│       │   └── weather-assets-product/
│       │
│       └── layer-c/
│           ├── smoke-1-boobook-fairywren/
│           │   ├── manifest.csv
│           │   └── segments/
│           ├── mvp-multi-species/
│           └── product-multi-species/
│
├── dvc-cache/                                        ← DVC remote (opaque content-addressed objects)
│   └── files/md5/<ab>/<cdef123...>                   (managed by `dvc push/pull` ONLY)
│
├── release/                                          ← human-browsable blessed checkpoints
│   │
│   ├── layer-a/
│   │   ├── audioldm2-lora-raw-smoke-v1/
│   │   │   ├── best.pt
│   │   │   ├── metrics.json
│   │   │   ├── params.yaml
│   │   │   ├── training-manifest.csv                 (copy of which dataset was used)
│   │   │   └── sample-audit/                         (a few seed=42/43/44 WAV+PNG)
│   │   ├── audioldm2-lora-insects-smoke-v1/
│   │   ├── audioldm2-lora-mvp-v1/                    (future)
│   │   └── vae-site257-v1/                           (legacy VAE bed)
│   │
│   ├── layer-b/
│   │   └── weather-mixer-v1/                         (config + asset bundle, no model)
│   │
│   ├── layer-c/
│   │   ├── audiogen-lora-boobook-smoke-v1/
│   │   ├── audiogen-lora-splendid-fairywren-smoke-v1/
│   │   └── audiogen-lora-<species>-mvp-v1/           (future)
│   │
│   └── vocoder/
│       └── hifigan-site257-v1/
│
└── logs/                                             ← training runs, tensorboard, audit traces
    ├── layer-a/
    │   ├── audioldm2-lora-raw-smoke/2026-05-06/
    │   └── audioldm2-lora-insects-smoke/2026-05-10/
    ├── layer-b/
    └── layer-c/
        └── audiogen-lora-boobook-smoke/2026-05-11/
```

## What each prefix does

| Prefix | Purpose | Written by | Read by |
|---|---|---|---|
| `dataset/metadata/` | Small CSVs defining the universe of data | Humans / download scripts | Everything |
| `dataset/original/` | Raw downloads (125 GB pool, annotations) | Download scripts | Manifest-builders only |
| `dataset/training_dataset/<layer>/<stage>/` | Curated per-experiment training subsets | Manifest + segment builders | Training scripts |
| `dvc-cache/` | DVC content-addressed blob store | `dvc push` only | `dvc pull` only |
| `release/<layer>/<run-name>-v<N>/` | Blessed checkpoint + metrics + audit samples | Manual promote step | Inference server, teammates, PR review |
| `logs/<layer>/<run-name>/<date>/` | Training logs, TB events, debug bundles | Training scripts | Humans |

## Two important rules

**1. `dataset/`, `release/`, and `logs/` are human-readable mirrors — DVC alone won't populate them.**
DVC stores bytes as opaque hashed objects under `dvc-cache/` only. The readable trees above are populated either by:
- being on disk locally and `dvc add`-tracked (the readable path lives in git working tree + local cache), OR
- explicit `aws s3 sync` after a `dvc push` if you want them browsable on S3 too.

Most teams skip mirroring `dataset/original/` (too big) but mirror `release/` and `dataset/metadata/` (small, useful).

**2. `dataset/training_dataset/<layer>/<stage>/` is the right granularity for DVC tracking.**
Each curated subset is its own DVC artifact. New experiments (e.g. `layer-a/mvp/`) get added without touching prior ones. Smoke / MVP / product progression is just three sibling folders.

## Dual-tracking policy for source data

The 125 GB `dataset/original/` pool and the curated `dataset/training_dataset/` subsets are **different artifacts** — DVC tracks both:

| Artifact | DVC tier | Reason |
|---|---|---|
| `dataset/metadata/*.csv` | Git (small) | Defines the universe |
| `dataset/original/.../downloaded_clips/` | DVC, cold tier (Glacier) | Insurance copy — A2O upstream is already unstable (12 unrecoverable clips) |
| `dataset/original/.../downloaded_annotations/` | DVC, cold tier | Same |
| `dataset/training_dataset/<layer>/<stage>/` | DVC, hot tier | Active training input |
| Per-experiment manifest CSV inside each subset | Git (small) | Source of truth for what got trained |

## DVC remote config

```bash
python3 -m dvc remote add -d s3 \
  s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache

python3 -m dvc remote modify s3 region ap-southeast-2
python3 -m dvc remote modify s3 profile capstone2
```

Scope DVC to the `dvc-cache/` prefix, **not the bucket root** — this prevents the opaque hash tree from polluting the bucket and keeps `dataset/`, `release/`, `logs/` cleanly separated.

## Optional: mirror readable trees alongside DVC

```bash
# After dvc push, also mirror the human-readable view for browsable artifacts
aws s3 sync ./dataset/metadata \
  s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/metadata \
  --profile capstone2

aws s3 sync ./checkpoints/<blessed-run> \
  s3://eco-acoustic-data.store.adelaideuni.cloud/release/<layer>/<run>-v1 \
  --profile capstone2
```

## Bucket-level settings

| Setting | Value | Why |
|---|---|---|
| Versioning | ON | Protects against accidental `dvc gc -c` |
| Block public access | ON | Not public data |
| Encryption | SSE-S3 (or SSE-KMS) | Standard hygiene |
| IAM | One IAM user per team member with `s3:GetObject`/`PutObject` | Audit trail |

## Lifecycle rules

| Prefix | Rule | Reason |
|---|---|---|
| `dataset/original/` | Transition to Glacier Deep Archive after 30 days | 125 GB rarely re-read once segments are extracted |
| `dvc-cache/` | Transition objects > 180 days to Glacier IA | Old experiment artifacts |
| `logs/` | Expire after 90 days | Training logs are disposable |
| `release/` | Standard storage, no expiry | Always need fast access |
| `dataset/training_dataset/` | Standard storage | Active training input |

## What NOT to do

- ❌ Don't point DVC at the bucket root — keep it under `dvc-cache/`.
- ❌ Don't write directly to `dvc-cache/` with `aws s3 cp`. Use `dvc push` only.
- ❌ Don't try to give DVC a human-readable key like `s3://.../checkpoints/foo/best.pt`. DVC will overwrite/ignore your structure.
- ❌ Don't commit AWS credentials. Use `~/.aws/credentials` with a named profile (`capstone2`).
