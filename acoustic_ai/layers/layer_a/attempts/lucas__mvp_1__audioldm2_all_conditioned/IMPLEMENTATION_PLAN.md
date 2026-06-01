# Layer A MVP-1 — Implementation Plan

**Attempt ID:** `lucas__mvp_1__audioldm2_all_conditioned`
**Branch:** `model/lucas/layer-a-mvp-1-all-conditioned`
**Owner:** Lucas
**Status:** Scaffolded, not yet materialised. Awaiting dataset build, audit, and training.

---

## 0. Prerequisites

Two environments to bring up before implementation: **(A)** local Mac for the
dataset build + dry-run, and **(B)** Server B (assumed totally fresh) for
training. Walk each checklist top-to-bottom; every line is a verification
command you can actually run.

### 0.A Local machine (dataset build + dry-run)

The local box only builds the dataset, runs the builder dry-run, and
`dvc push`es the result. It does **not** train.

| # | Check | Verify | If missing |
|---|---|---|---|
| 1 | Repo on the correct branch | `git rev-parse --abbrev-ref HEAD` → `model/lucas/layer-a-mvp-1-all-conditioned` | `git checkout model/lucas/layer-a-mvp-1-all-conditioned` |
| 2 | `acoustic_ai/.venv` exists | `./acoustic_ai/.venv/bin/python --version` | `python3 -m venv acoustic_ai/.venv && ./acoustic_ai/.venv/bin/pip install -r acoustic_ai/requirements.txt` |
| 3 | venv has torch + librosa | `./acoustic_ai/.venv/bin/python -c "import torch, librosa, soundfile, pandas; print('ok')"` | re-run pip install from #2 |
| 4 | DVC at user-site (NOT venv) | `which dvc` → `~/Library/Python/3.9/bin/dvc` and `dvc --version` | follow [dvc_workflow.md § Fresh clone](../../../../../.claude/context/dev/dvc_workflow.md#fresh-clone-setup) steps 1–2 |
| 5 | DVC's boto stack importable | `python3 -c "import boto3, s3fs; from fsspec.callbacks import DEFAULT_CALLBACK; print('ok')"` | `pip3 install --user --upgrade boto3 s3fs aiobotocore fsspec` |
| 6 | AWS profile `capstone2` works | `aws sts get-caller-identity --profile capstone2` and `aws s3 ls s3://eco-acoustic-data.store.adelaideuni.cloud/ --profile capstone2 \| head` | add `[capstone2]` to `~/.aws/credentials` + `[profile capstone2]` with `region=ap-southeast-2` to `~/.aws/config` |
| 7 | Git hooks installed | `ls .git/hooks/post-checkout` exists | `dvc install` |
| 8 | Tier-3 ambient pool present locally | `dvc pull acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_index.csv.dvc` and `wc -l` on the CSV → 1,983 lines (1,982 + header) | re-run `dvc pull`; if S3 auth fails go back to #6 |
| 9 | Annotation CSVs present (rule-5 join input) | `ls resources/site_257_bowra-dry-a/downloaded_annotations/ \| wc -l` → ≥287 | `dvc pull resources/site_257_bowra-dry-a/downloaded_annotations.dvc` |
| 10 | Training manifest present (env-row join) | `test -f resources/site_257_bowra-dry-a/site_257_training_manifest.csv && head -1 $_` | check out via git (it's git-tracked) |
| 11 | Builder dry-runs cleanly | `./acoustic_ai/.venv/bin/python script/dataset/build_mvp1_all_conditioned_dataset.py --per-cell-cap 100 --dry-run` → reproduces the §5.4–§5.5 numbers (1,554 clean → 1,082 after cap) | inspect the builder's drop-reason printout against §5.4 |
| 12 | ffmpeg on PATH (builder calls it for resampling) | `ffmpeg -version` | `brew install ffmpeg` |
| 13 | Free disk for materialised dataset | `df -h .` → ≥10 GB free on the repo's filesystem | free space; dataset is "a few GB" per §7.4 |

After 1–13 pass, the local environment is ready for §5 (materialisation) and §6 (audit).

### 0.B Server B (fresh machine, training only)

Assumes a fresh Linux box with NVIDIA driver + CUDA already installed by IT.
Everything else is set up by you. Order matters — later steps assume earlier
ones.

| # | Check | Verify | If missing |
|---|---|---|---|
| 1 | OS + shell baseline | `uname -a` (Linux x86_64), `bash --version` | n/a |
| 2 | NVIDIA driver + GPU visible | `nvidia-smi` shows ≥1 GPU with ≥16 GB VRAM (24 GB recommended for batch=4) | escalate to IT |
| 3 | CUDA runtime version | `nvidia-smi` "CUDA Version" line ≥ 12.1 (matches torch ≥ 2.2 wheels) | escalate to IT or pin a torch build matching the installed CUDA |
| 4 | Python 3.10–3.12 available | `python3 --version` | `apt install python3.11 python3.11-venv` (or distro equivalent) |
| 5 | git installed + SSH key registered with GitHub | `ssh -T git@github.com` returns success message | `ssh-keygen -t ed25519` → add public key to GitHub |
| 6 | Repo cloned | `git clone git@github.com:…/COMP-6000-Capstone2.git && cd COMP-6000-Capstone2` and `git checkout model/lucas/layer-a-mvp-1-all-conditioned` | — |
| 7 | `acoustic_ai/.venv` built | `python3 -m venv acoustic_ai/.venv && ./acoustic_ai/.venv/bin/pip install --upgrade pip` then `./acoustic_ai/.venv/bin/pip install -r acoustic_ai/requirements.txt` | — |
| 8 | Torch sees CUDA | `./acoustic_ai/.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True <gpu-name>` | reinstall torch with the CUDA-matched index URL (e.g. `--index-url https://download.pytorch.org/whl/cu121`) |
| 9 | `accelerate` on venv PATH | `./acoustic_ai/.venv/bin/accelerate --version` | already in requirements.txt; re-run #7 |
| 10 | `accelerate` configured non-interactively | `./acoustic_ai/.venv/bin/accelerate config default` (writes `~/.cache/huggingface/accelerate/default_config.yaml` for single-GPU bf16) | run the command above |
| 11 | AWS CLI v2 installed | `aws --version` → `aws-cli/2.x` | install per AWS docs (`curl … awscli-exe-linux-x86_64.zip`) |
| 12 | AWS profile `capstone2` works | `aws sts get-caller-identity --profile capstone2` and `aws s3 ls s3://eco-acoustic-data.store.adelaideuni.cloud/ --profile capstone2 \| head` | populate `~/.aws/credentials` + `~/.aws/config` (same content as local 0.A #6) |
| 13 | DVC at user-site (Route A) | `pip3 install --user dvc 'boto3' 's3fs' 'aiobotocore' 'fsspec'` then `dvc --version` and `python3 -c "import boto3, s3fs; from fsspec.callbacks import DEFAULT_CALLBACK; print('ok')"` | follow [dvc_workflow.md § Fresh clone](../../../../../.claude/context/dev/dvc_workflow.md#fresh-clone-setup); ensure `~/.local/bin` is on PATH |
| 14 | DVC remote reachable | `dvc remote list` shows the project remote (already in `.dvc/config`); `dvc pull --dry resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset.dvc` | go back to #12/#13 |
| 15 | HuggingFace Hub reachable for `cvssp/audioldm2` | `./acoustic_ai/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('cvssp/audioldm2', allow_patterns='README.md')"` | model is public — if this fails it's a network / proxy issue, not auth |
| 16 | Free disk (deps + cache + dataset + checkpoint) | `df -h ~` → ≥40 GB free (venv ~6 GB + HF cache ~8 GB + dataset few GB + scratch) | clean caches / mount additional volume |
| 17 | Dataset pulled to local disk (Route A — recommended) | `dvc pull resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset.dvc` and `wc -l resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest.csv` matches local | if Route A fails, fall back to Route B `aws s3 sync` per §7.3 |
| 18 | Output dir for the candidate checkpoint exists | `mkdir -p model/candidates/lucas/mvp_1__audioldm2_all_conditioned` (already in git from the scaffold) | — |
| 19 | Training script smoke-imports without errors | `./acoustic_ai/.venv/bin/python -c "import importlib.util, sys; spec = importlib.util.spec_from_file_location('t', 'acoustic_ai/layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/code/train_audioldm2.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"` exits 0 (or with `--help` if the script supports it) | fix the import error before launching the full run |
| 20 | One-step dry-run training succeeds | run the §7.2 command with `--num_epochs 1 --max_train_steps 2` against the val-only or a 4-row subset of the manifest; confirm loss prints and a `.safetensors` is written | debug before committing to the full ~30–90 min run |

After 1–20 pass, Server B is ready for the full §7.2 training run.

### 0.C Fast pre-flight (single script — optional)

If you want a one-shot check, run this from the repo root on either machine —
it prints `OK` lines for each prerequisite and exits non-zero on the first
failure:

```bash
./acoustic_ai/.venv/bin/python - <<'PY'
import shutil, subprocess, sys
checks = [
    ("venv python",   lambda: subprocess.check_output(["./acoustic_ai/.venv/bin/python", "--version"])),
    ("torch",         lambda: __import__("torch").__version__),
    ("librosa",       lambda: __import__("librosa").__version__),
    ("dvc",           lambda: shutil.which("dvc") or (_ for _ in ()).throw(RuntimeError("dvc not on PATH"))),
    ("aws",           lambda: shutil.which("aws") or (_ for _ in ()).throw(RuntimeError("aws CLI not on PATH"))),
    ("ffmpeg",        lambda: shutil.which("ffmpeg") or (_ for _ in ()).throw(RuntimeError("ffmpeg not on PATH"))),
]
fail = 0
for name, fn in checks:
    try:    print(f"OK   {name}: {fn()}")
    except Exception as e: print(f"FAIL {name}: {e}"); fail += 1
sys.exit(fail)
PY
```

Server B should additionally pass:

```bash
./acoustic_ai/.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('OK cuda:', torch.cuda.get_device_name(0))"
aws sts get-caller-identity --profile capstone2 >/dev/null && echo "OK aws profile"
```

---

## 1. Goal & hypothesis

Smoke tests proved AudioLDM2 + LoRA can reproduce **one** narrow ambient scene each
(smoke_1 = spring night, smoke_2 = summer-afternoon insects). MVP-1 keeps the
**same method** and tests the next question:

> Can a single LoRA cover the entire clean ambient pool from site 257 if captions encode
> `(season, diel_bin, temperature, humidity, wind, date)`?

**Success criterion (qualitative):** changing the caption's condition fields produces
audibly different ambient beds — and the smoke-test scenes remain reproducible from the
new model.

**Failure criterion:** model produces a single "average ambient" insensitive to caption
fields. Fall back to fewer cells or per-cell LoRAs (MVP-2).

---

## 2. Scope

In scope:
- Site 257 only (the only site with data online today).
- Layer A (ambient bed) only.
- Generation mode only.
- One LoRA across all (season, diel) cells.

Out of scope:
- Multi-site training.
- Layer B/C/D/E.
- Analysis or Transformation modes.
- Promoting to `model/production/<role>/` (decided after evaluation).
- Server A on-demand worker orchestration — training will run manually on Server B.

---

## 3. Branch & folder layout

```
acoustic_ai/layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/
├── __init__.py
├── README.md                      ← attempt overview (already written)
├── IMPLEMENTATION_PLAN.md         ← this doc
├── params.yaml                    ← per-attempt hyperparams (already written)
├── code/                          ← copied verbatim from smoke_1
│   ├── __init__.py
│   ├── audioldm2_dataset.py
│   ├── handler.py
│   ├── layer_a_visualization.py
│   ├── sample_audioldm2.py
│   └── train_audioldm2.py
├── expected/                      ← real-audio ground truth (filled at artifact step)
├── showcase/                      ← generated showcase (seed_42 + 2 variations)
└── dev-artifacts-self-testing/    ← gitignored scratch

model/candidates/lucas/mvp_1__audioldm2_all_conditioned/
├── README.md                      ← checkpoint README (conventions § 6.3, already written)
├── params.yaml                    ← frozen snapshot of training params (already written)
└── adapter_model.safetensors.dvc  ← created after training, DVC-pushed

script/dataset/
└── build_mvp1_all_conditioned_dataset.py   ← builder, dry-run verified

resources/site_257_bowra-dry-a/
└── mvp1_all_conditioned_dataset/           ← materialised dataset (created in §5)
    ├── manifest.csv                         (git)
    ├── audit_samples/                       (gitignored, see §6)
    └── clips/<NNNN_segment_id>/             (DVC after build)
        ├── audio.wav
        ├── caption.txt
        └── meta.json
```

---

## 4. Data pipeline summary

The pipeline is three-tier. MVP-1 operates on the third tier (10s ambient segments).

```
┌──────────────────────────────────────────────────────────────┐
│ Tier 1: Raw recordings (~30 min FLACs)                       │
│   resources/site_257_bowra-dry-a/downloaded_clips/           │
│   DVC-tracked, ~125 GB                                       │
└──────────────────────────────────────────────────────────────┘
                          ↓ clipping (script/dataset/build_training_manifest.py)
┌──────────────────────────────────────────────────────────────┐
│ Tier 2: ~60-90s clips                                        │
│   site_257_training_manifest.csv (clip_path + start/end)     │
└──────────────────────────────────────────────────────────────┘
                          ↓ ambient-segment extraction (smoke_4 precompute pipeline)
┌──────────────────────────────────────────────────────────────┐
│ Tier 3: ~10s ambient-pure segments                           │
│   .../smoke_4/data/ambient/ambient_index.csv                 │
│   .../smoke_4/data/ambient/ambient_segments/*.wav  (DVC)     │
│   1,982 segments total                                       │
└──────────────────────────────────────────────────────────────┘
                          ↓ build_mvp1_all_conditioned_dataset.py (NEW, §5)
┌──────────────────────────────────────────────────────────────┐
│ MVP-1 training dataset                                       │
│   resources/.../mvp1_all_conditioned_dataset/                │
│   ~1,082 clips after filter + balance                        │
└──────────────────────────────────────────────────────────────┘
```

The first two tiers are not re-run for MVP-1 — they're already on DVC and on disk.

---

## 5. Dataset build

Script: [`script/dataset/build_mvp1_all_conditioned_dataset.py`](../../../../../script/dataset/build_mvp1_all_conditioned_dataset.py)

### 5.1 Filter policy

**Annotated-event exclusion is the most important rule for an ambient model.** It runs
as a layered defense across three stages — two are already active, one is proposed in §6.1.

#### 5.1.1 Layered defense against foreground events

| Layer | Where | What it does | Status |
|---|---|---|---|
| **L1.** Upstream content anomaly detector | `acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/precompute/build_ambient_index.py` | Walks each 30-min recording with spectral-flux + RMS anomaly detection; only "clean spans" (no sudden energy / spectral jumps) become ambient segments. This is content-based, not annotation-based — it catches loud transients regardless of whether BirdNET labelled them. | **Active.** Produced the 1,982-segment input pool. |
| **L2.** Builder rule 5 — annotation overlap | `build_mvp1_all_conditioned_dataset.py` | For each candidate segment, computes its absolute start/end inside the parent recording and rejects it if it overlaps **any** interval listed in `downloaded_annotations/annotations_<rec_id>.csv`. | **Active in MVP-1 builder.** |
| **L3.** Cheap audio-content filters | proposed §6.1 | RMS / crest-factor / spectral-flatness drops residual loud transients, near-silence, anthropogenic noise that survived L1+L2. | **Proposed; not yet implemented.** |

Verified preconditions for L2 to work:
- All 229 distinct recordings in the ambient pool have annotation files on disk
  (`ls downloaded_annotations/ | wc -l = 287`, ambient ∩ annotated = 229).
- L2's join key matches between `rec_id_from_source_clip()` (the source-clip folder
  name) and the `annotations_<rid>.csv` filename. No join bug.
- Result on the 1,982-segment input pool: **10 segments dropped by L2**, meaning L1
  already caught most annotated events implicitly. The 10 are the residual cases
  where a labelled event was quiet/distant enough not to trigger the content detector.

#### 5.1.2 All rules

Applied per-segment, in order. Any segment failing any rule is **excluded** from the
manifest and never reaches training.

| # | Rule | Threshold | Default arg | Rationale |
|---|---|---|---|---|
| 1 | `duration_s ≥` | 10.0 s | `--min-duration-s 10.0` | AudioLDM2 trains on 10s windows |
| 2 | `wind_speed_ms <` | 4.5 m/s | `--max-wind-speed-ms 4.5` | Strong wind dominates the spectrum (matches smoke_2) |
| 3 | `wind_max_ms <` | 8.0 m/s | `--max-wind-max-ms 8.0` | Catches gusty days even when hourly mean is below threshold |
| 4 | `precipitation_mm <` | 0.1 mm | `--max-precipitation-mm 0.1` | Rain becomes the dominant texture; not an "ambient bed" |
| 5 | **No annotated-event overlap** (L2 of §5.1.1) | binary | (always on) | Reject any segment whose absolute interval overlaps any event in the recording's annotation file |
| 6 | Source segment WAV exists | binary | (always on) | Hygiene; should be 0 drops after `dvc pull` |
| 7 | env row exists (recording_id joined via training_manifest) | binary | (always on) | Required to evaluate rules 2-4 |

#### 5.1.3 What this layered defense does *not* catch

- **Foreground events BirdNET didn't label** — annotation files reflect only what was
  detected at the BirdNET confidence threshold used during annotation. Quiet or
  out-of-vocabulary events may exist in segments that pass L1+L2. Mitigation:
  L3 cheap content filters in §6.1, plus the §6.2 stratified human audit.
- **Anthropogenic noise not present in env data** — distant planes, vehicles, voices.
  These are unannotated. Mitigation: L3 spectral-flatness filter.

### 5.2 Balance policy

After filtering, segments are sorted deterministically by `(season, diel_bin, date, segment_id)` and capped:

| Knob | Default | Effect |
|---|---|---|
| `--per-cell-cap N` | 100 | Max `N` clips per `(season, diel_bin)` cell. `0` = unlimited. |
| `--max-per-date N` | 0 | Max `N` clips per recording date. `0` = unlimited (recommended for MVP). |

Selection within a cell is the deterministic sort order, **not** random — so the
materialisation is reproducible from the source pool.

### 5.2.1 Train/val split

After balancing, each cell is independently split into train/val with a per-cell
deterministic shuffle:

| Knob | Default | Effect |
|---|---|---|
| `--val-fraction F` | 0.1 | Fraction of clips held out as val, stratified per `(season, diel_bin)` cell. `0` = no val split. |
| `--split-seed N` | 42 | RNG seed for per-cell shuffle. Same seed → same val set. |

The split is **stratified**: each cell gets `round(N_cell × val_fraction)` clips
assigned to val (minimum 1 for cells with ≥2 clips; always keeps ≥1 in train).
The manifest has a `split` column (`train` / `val`); val clips are loaded by the
training script with `--no_val` *off* to compute a per-epoch val loss.

Dry-run with `--per-cell-cap 100 --val-fraction 0.1`:

| Cell | train | val |
|---|---:|---:|
| autumn afternoon | 90 | 10 |
| autumn night | 90 | 10 |
| spring night | 90 | 10 |
| summer night | 90 | 10 |
| spring morning | 84 | 9 |
| spring dawn | 77 | 9 |
| spring afternoon | 77 | 8 |
| winter night | 75 | 8 |
| winter dawn | 67 | 8 |
| summer dawn | 57 | 6 |
| autumn dawn | 53 | 6 |
| autumn morning | 34 | 4 |
| summer afternoon | 26 | 3 |
| winter afternoon | 23 | 2 |
| summer morning | 22 | 2 |
| winter morning | 20 | 2 |
| **Total** | **975** | **107** |

### 5.3 Caption schema

Same template used at train time and at inference time:

```
{diel} {season} ambient soundscape, Bowra dry woodland, Australia,
{temp_bucket} ({temp}C), {humidity_bucket}, {wind_bucket},
recorded {YYYY-MM-DD}, no music, no machinery
```

Bucket definitions (all closed-left, open-right intervals):

| Field | Source column | Buckets |
|---|---|---|
| diel | `diel_bin` from ambient_index | `dawn` / `morning` / `afternoon` / `night` |
| season | `season` from ambient_index | `spring` / `summer` / `autumn` / `winter` |
| temp | `temperature_c` from env | `cold` <15, `mild` 15-25, `warm` 25-32, `hot` 32-40, `very hot` ≥40 |
| humidity | `humidity_pct` from env | `dry air` <40, `moderate humidity` 40-70, `humid air` ≥70 |
| wind | `wind_speed_ms` from env | `still` <0.5, `light breeze` 0.5-2, `moderate wind` 2-4.5 |

Combinatorial caption space: `4 × 4 × 5 × 3 × 3 = 720`. In practice the env axes are
heavily correlated with season/diel, so the realised space is ~80-150 distinct strings.

### 5.4 Survey results — what the source pool supports

From the dry-run on 1,982 segments (executed 2026-05-29):

**Filter drops:**

| Filter | Dropped |
|---|---:|
| Strong wind (rule 2 or 3) | 363 |
| Rainy (rule 4) | 55 |
| Annotated-event overlap (rule 5) | 10 |
| **Kept (clean candidates)** | **1,554** |

**Per (season, diel) — clean candidates before balancing:**

| season | diel | clips |
|---|---|---:|
| spring | night | 342 |
| autumn | night | 219 |
| autumn | afternoon | 161 |
| summer | night | 150 |
| spring | morning | 93 |
| spring | dawn | 86 |
| spring | afternoon | 85 |
| winter | night | 83 |
| winter | dawn | 75 |
| summer | dawn | 63 |
| autumn | dawn | 59 |
| autumn | morning | 38 |
| summer | afternoon | 29 |
| winter | afternoon | 25 |
| summer | morning | 24 |
| winter | morning | 22 |
| **Total** | | **1,554** |

Distribution is night-heavy (~51% of total). Smallest 5 cells have only 22-38 clips
each — below the smoke_1/smoke_2 budget of 35-50.

### 5.5 Results under different balance configurations

| `--per-cell-cap` | Total clips | Cells affected | Notes |
|---|---:|---|---|
| 0 (no cap) | 1,554 | none | Maximum data; night-skewed |
| 150 | 1,282 | spring night, autumn night, autumn afternoon, summer night | Mild de-skew |
| **100 (proposed)** | **1,082** | top 4 cells capped | ~30× smoke_2 scale; reasonable balance |
| 80 | 975 | top 5 cells capped | Tighter balance; smaller pool |
| 50 | 696 | top 9 cells capped | Matches smoke training budget per cell |

**Decision:** `--per-cell-cap 100` for the first run. Rationale:
- Keeps the strongest cells well-trained.
- Leaves enough headroom for cross-scene transfer to help thin cells.
- ~30× scale up from smoke_2 — meaningful MVP-vs-smoke comparison.

### 5.6 Build command

```bash
./acoustic_ai/.venv/bin/python script/dataset/build_mvp1_all_conditioned_dataset.py \
  --per-cell-cap 100 \
  --overwrite
```

Output:
- `resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest.csv` (git)
- `resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/clips/` (DVC, see §8)

Runtime estimate: 5-10 min for ffmpeg conversion of 1,082 clips on local Mac.

---

## 6. Pre-training audit

The metadata filters in §5.1 catch wind, rain, and annotated events. They do **not**
inspect audio content. Two gaps:

- BirdNET annotations are sparse — many real foreground events are unlabelled.
- No check for clipping, near-silence, recorder dropouts, or anthropogenic noise that
  doesn't appear in env data.

The audit step covers both.

### 6.1 Automated content filters (proposed addition to the builder)

Cheap per-clip checks added to `build_mvp1_all_conditioned_dataset.py`. Each clip is
evaluated and dropped from the manifest if it fails:

| Filter | Threshold | Reason for drop |
|---|---|---|
| `rms` | `< 0.0005` | Near-silent (recorder off, dropout) |
| `rms` | `> 0.3` | Clipping / loud anomaly |
| `dc_offset` | `> 0.05` | Recorder fault |
| `peak_to_rms_ratio` (crest factor) | `> 30` | Likely contains a sharp foreground transient (gunshot, click) |
| `spectral_flatness_band_500_2000` | `< 0.05` for >60% of frames | Highly tonal — likely motor / engine bleed |

These are deliberately conservative — they catch obvious failures, not subjective
quality. Expected drop rate: low single-digit percent of the 1,082 candidates. Stats
are logged.

**Status:** not yet implemented. ~30-line change to the builder. Apply before running
the materialisation.

### 6.2 Human stratified audit

1,082 clips is too many to listen to one-by-one. Stratified sample:

- Builder writes `mvp1_all_conditioned_dataset/audit_samples/<season>_<diel>/` with 5
  random clips per cell (16 cells × 5 = 80 clips, ~13 min of audio).
- Audit checklist per cell:
  - [ ] No anthropogenic noise (motors, planes, voices) dominant
  - [ ] No mis-classified diel/season (e.g. clearly daytime clip labelled "night")
  - [ ] Volume in expected range (not all silence, not all clipping)
  - [ ] Caption text matches what you hear
- Loop: if a cell's samples are consistently bad → tighten that cell's filter (e.g.
  drop wind threshold to 3.0 m/s for that cell, or drop a date range) → rebuild → re-audit.

Audit notes go into this file under §6.3 once performed.

### 6.3 Audit results

_Pending dataset materialisation._

---

## 7. Training

### 7.0 Verified properties of the (copied) training script

**Concern resolved (caption conditioning):** lines 203-208 + 253-260 of
`train_audioldm2.py` confirm captions are tokenized + encoded per batch by
`pipeline.encode_prompt(batch["caption"], ...)` and the resulting embeddings
flow into the UNet's cross-attention. Per-row captions DO drive the gradient.
Verified directly against smoke_1's script (the MVP-1 copy is a verbatim clone
of that script plus the val-loss addition described below).

### 7.1 Hyperparameters (first-run proposal)

From `params.yaml` (already committed in the attempt folder):

```yaml
training:
  base_model: cvssp/audioldm2
  manifest: resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest.csv
  num_epochs: 3
  batch_size: 4              # CUDA; bump to 8 if VRAM allows
  learning_rate: 1.0e-5
  normalize_audio: false     # raw field-recording levels (smoke_1 negative result)
  lora:
    r: 8
    alpha: 32
    dropout: 0.1
    target_modules: [to_q, to_k, to_v, to_out.0]
```

Step-count comparison:

| Run | Clips | Batch | Epochs | Steps/epoch | Total steps |
|---|---:|---:|---:|---:|---:|
| smoke_1 | 50 | 1 | 5 | 50 | 250 |
| smoke_2 | 35 | 1 | 5 | 35 | 175 |
| **MVP-1 first try** | **1,082** | **4** | **3** | **270** | **810** |

MVP-1 sees ~3-5× more total steps than smoke_2 but on 30× more data. If the first run
under-converges (showcase samples sound undertrained), bump `num_epochs` to 5 → 1,350
total steps. LoRA `r/alpha` are kept identical to smoke to isolate the data variable.

### 7.2 Training command

From `acoustic_ai/` on a CUDA host:

```bash
./.venv/bin/accelerate launch \
  layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/code/train_audioldm2.py \
  --manifest_path ../resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest.csv \
  --output_dir ../model/candidates/lucas/mvp_1__audioldm2_all_conditioned \
  --batch_size 4 \
  --num_epochs 3 \
  --learning_rate 1e-5
```

The training script will:
- Load the manifest's `train` rows (split filter built into the modified
  `AudioLDM2Dataset`).
- Run the standard diffusion training loop on those.
- At the end of each epoch, switch UNet to `eval()`, run the `val` rows under
  `torch.no_grad()` with `--val_eval_seed 12345` (fixed) so the resulting
  `val_loss` is comparable across epochs, and log:
  ```
  Epoch 1/3: train_loss=0.1234  val_loss=0.1320
  Epoch 2/3: train_loss=0.0921  val_loss=0.1015
  Epoch 3/3: train_loss=0.0712  val_loss=0.0890
  ```
- Pass `--no_val` to skip the val pass entirely (e.g. when training on a
  smoke manifest that doesn't carry a `split` column).

Runtime estimate: ~30-90 min on a single A100/3090-class GPU (810 steps).

### 7.3 Server B data transfer

Training reads `audio_path` as a **local file** — no S3-streaming layer. Server B
must have the dataset on local disk before training. Two routes:

**Route A — DVC (canonical, conventions-compliant):**

```bash
# Locally after build, before transfer
dvc add resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset
git add resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset.dvc .gitignore
git commit -m "data(layer-a-mvp-1): add conditioned ambient dataset"
dvc push

# On Server B
git clone <repo> && cd COMP-6000-Capstone2
# configure AWS profile capstone2
dvc pull resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset.dvc
```

**Route B — direct S3 sync (faster if Server B isn't a full DVC host):**

```bash
# Locally
aws s3 sync \
  resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset \
  s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/training_dataset/layer-a/mvp/ \
  --profile capstone2

# On Server B
aws s3 sync \
  s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/training_dataset/layer-a/mvp/ \
  ./mvp1_dataset/
# Edit manifest paths or pass --manifest_path absolute
```

**Recommendation:** Route A. It's the documented project workflow, preserves provenance,
and means Server B is identical to a teammate's checkout. Use Route B only if Server B
can't have DVC installed.

### 7.4 What needs to live on Server B

| Path | Source | Size | Notes |
|---|---|---:|---|
| Repo working tree | `git clone` | ~50 MB | Code, configs |
| `acoustic_ai/.venv/` | `pip install -r requirements.txt` | ~6 GB | Python deps + torch |
| `resources/.../mvp1_all_conditioned_dataset/clips/` | `dvc pull` or `aws s3 sync` | a few GB | 1,082 WAVs |
| HuggingFace cache for `cvssp/audioldm2` | downloaded on first run | ~8 GB | Base model |
| Output dir for checkpoint | created by training | small | LoRA adapter is ~10-50 MB |

**Not needed on Server B:**
- `dataset/original/downloaded_clips/` (125 GB) — only used at dataset-build time, locally.
- `dataset/original/downloaded_annotations/` — same.
- `ambient_segments/` (1,982 raw segments) — only needed by the builder, locally.

### 7.5 Running training detached from SSH

The real training run takes ~30-90 min, so the SSH session must not own the
process — closing the terminal would `SIGHUP` it. Pick one:

**Option A — `tmux` (recommended, reattachable):**

```bash
tmux new -s train
# inside the tmux session:
cd acoustic_ai
./.venv/bin/accelerate launch layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/code/train_audioldm2.py \
  --manifest_path ... --output_dir ... 2>&1 | tee train.log
# detach: Ctrl-b then d   → safe to `exit` SSH
# reattach later:  ssh server  →  tmux attach -t train
```

**Option B — `nohup` + `&` (fire-and-forget):**

```bash
cd acoustic_ai
nohup ./.venv/bin/accelerate launch \
  layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/code/train_audioldm2.py \
  --manifest_path ... --output_dir ... \
  > train.log 2>&1 &
echo $! > train.pid
disown
exit   # SSH disconnect is now safe
# later: ssh server && tail -f acoustic_ai/train.log
# kill if needed: kill "$(cat acoustic_ai/train.pid)"
```

Notes:
- Always tee/redirect to a log file — once detached, stdout is gone otherwise.
- If Server B is a Mac (MPS), also wrap the command with `caffeinate -i` so
  system sleep doesn't pause the job.
- `accelerate launch` spawns workers; killing the parent PID is enough — it
  propagates `SIGTERM` to the group.

---

## 8. Post-training artifacts

After training succeeds:

1. **Checkpoint binary**:
   ```bash
   dvc add model/candidates/lucas/mvp_1__audioldm2_all_conditioned/adapter_model.safetensors
   dvc push
   git add model/candidates/lucas/mvp_1__audioldm2_all_conditioned/adapter_model.safetensors.dvc
   ```
2. **Expected samples** — 2-3 real audio clips from the manifest, one per representative cell:
   - `expected/real_<spring_night_clip>/{audio.wav.dvc, spectrogram.png, metadata.json}`
   - `expected/real_<summer_afternoon_clip>/...`
   - `expected/real_<autumn_night_clip>/...`
3. **Showcase samples** — generated from the trained checkpoint:
   - `showcase/seed_42_baseline/` (canonical seed, the "default" caption)
   - `showcase/seed_43_summer_afternoon/` (caption variation)
   - `showcase/seed_44_autumn_night/` (caption variation)
   - All three files (audio.wav, spectrogram.png, metadata.json) DVC-tracked.
4. **Registry entry** in `acoustic_ai/registry.yaml`:
   ```yaml
   layers:
     layer_a:
       attempts:
         lucas__mvp_1__audioldm2_all_conditioned:
           label: "Layer A — Site 257 conditioned ambient (mvp_1)"
           checkpoint: model/candidates/lucas/mvp_1__audioldm2_all_conditioned/
   ```
   Do **not** flip `default:` to mvp_1 until A/B compared against smoke_2 by ear.
5. **Update checkpoint README** §6 audit section with results.

---

## 9. Inference / dev endpoint contract — tentative

The smoke checkpoints lock the prompt server-side and expose only `seed`. MVP-1
needs a **conditioned** contract since the model is driven by caption fields:

```
{ seed, season, diel_bin, temperature_c, humidity_pct, wind_speed_ms }
```

The server would re-use the §5.3 caption template to assemble the prompt from these
fields. Validation:
- `seed`: non-negative integer 0–2147483647 (smoke contract)
- `season`: enum `spring|summer|autumn|winter`
- `diel_bin`: enum `dawn|morning|afternoon|night`
- `temperature_c`, `humidity_pct`, `wind_speed_ms`: numeric, server-side bucketed before caption build

**Status: deferred** until after the first training run validates that captions
actually drive the model output. Until then, `code/handler.py` is the smoke_1 verbatim
copy (seed-only, fixed prompt). The conditioned handler is a follow-up task once
showcase samples confirm the conditioning works.

---

## 10. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Model produces "average ambient" insensitive to caption fields | Medium | High | Pre-decided fallback: split into per-cell LoRAs (MVP-2) |
| Thin cells (winter morning 22, summer morning 24, winter afternoon 25) produce muddier output | Medium | Low | Accept; cross-scene transfer helps |
| Rule 5 under-catches because BirdNET annotations are sparse (no rec-id bug — verified) | Medium | Medium | Layered defense: L1 content anomaly detector upstream + L3 cheap content filters (§6.1) downstream |
| LoRA r=8 is too small to encode 16 cells + 3 weather axes | Medium | Medium | First-run signal: if loss plateaus high, bump `r` to 16/32 in MVP-1.1 |
| Server B data transfer wastes hours on the wrong route | Low | Medium | Route B (direct S3 sync) is the escape hatch if DVC has issues |
| Showcase samples look fine but degrade smoke_2's insect scene | Medium | Medium | A/B compare smoke_2 vs mvp_1 on the insect prompt before flipping `default:` |
| Hyperparams undertrain (3 epochs too few) | Medium | Low | Trivial to retrain at 5 epochs; checkpoint cost is cheap |

---

## 11. Open decisions

Need a call from the owner before proceeding past §5:

1. **Implement §6.1 content filters now or after first training run?**
   - Recommendation: **now**. ~30-line change, catches obvious failure modes before they pollute training.
2. **Implement §6 audit-sample sidecar in the builder?**
   - Recommendation: **yes** — minimal cost, makes the audit step concrete and reproducible.
3. **Server B transfer route — A (DVC) or B (S3 sync)?**
   - Recommendation: **A**, unless Server B doesn't have DVC.
4. **`--per-cell-cap` value?**
   - Tentatively chosen: **100**.
5. **Drop the 5 thin cells (22-38 clips) entirely?**
   - Open. Saves ~140 clips of likely-weak training data, removes 5 weak scenes from the supported set. My lean: keep them — cross-scene transfer should help and the cost is small.

**Already settled** (implemented in this scaffold):
- Concern (a) — captions consumed by training. **Resolved by code-read** (§7.0).
- Concern (b) — train/val split + per-epoch val loss. **Implemented**: builder writes `split` column (§5.2.1), `AudioLDM2Dataset` accepts `split=` filter, `train_audioldm2.py` runs a fixed-seed `no_grad` val pass each epoch and logs `train_loss`+`val_loss` (§7.2).

---

## 12. Step-by-step execution checklist

- [ ] **Prerequisites — local machine (§0.A 1–13).** Block on any FAIL.
- [ ] (Optional) Implement §6.1 content filters and §6.2 audit sidecar in the builder.
- [ ] Materialise dataset: `python script/dataset/build_mvp1_all_conditioned_dataset.py --per-cell-cap 100 --overwrite`
- [ ] Stratified human audit (§6.2). Loop if needed.
- [ ] `dvc add` + `dvc push` the dataset; `git add` the `.dvc` pointer + manifest.
- [ ] Commit scaffold + dataset on `model/lucas/layer-a-mvp-1-all-conditioned`.
- [ ] Transfer dataset to Server B (§7.3, Route A).
- [ ] **Prerequisites — Server B (§0.B 1–20).** Block on any FAIL.
- [ ] Train (§7.2).
- [ ] Generate `showcase/seed_42_baseline/` + 2 variations.
- [ ] Extract 2-3 `expected/real_<clip_id>/` from the manifest.
- [ ] A/B compare against smoke_1 and smoke_2 on their own prompts.
- [ ] Update `model/candidates/lucas/mvp_1__audioldm2_all_conditioned/README.md` § Results.
- [ ] Add `acoustic_ai/registry.yaml` entry.
- [ ] (Conditional) Implement §9 conditioned dev contract if first run validates conditioning.
- [ ] Add runbook `.claude/context/ai/runbooks/layer_a_mvp_1_all_conditioned.md`.
- [ ] Open PR.

---

## Cross-references

- [Attempt README](README.md)
- [Checkpoint README](../../../../../model/candidates/lucas/mvp_1__audioldm2_all_conditioned/README.md)
- [Dataset builder](../../../../../script/dataset/build_mvp1_all_conditioned_dataset.py)
- [Project conventions](../../../../../.claude/context/conventions.md)
- [Dev workflow (smoke → mvp loop)](../../../../../.claude/context/dev/dev_workflow.md)
- [S3 bucket layout](../../../../../.claude/context/dev/s3_bucket_layout.md)
- [DVC workflow](../../../../../.claude/context/dev/dvc_workflow.md)
- [smoke_1 runbook](../../../../../.claude/context/ai/runbooks/layer_a_smoke_1_spring_night.md)
- [smoke_2 runbook](../../../../../.claude/context/ai/runbooks/layer_a_smoke_2_insects.md)
