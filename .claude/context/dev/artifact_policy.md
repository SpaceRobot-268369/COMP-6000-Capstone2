# Artifact Policy

What sample outputs (audio, spectrograms, metadata) live in the repo, where,
and how they're stored. Goal: switching to any teammate's branch shows their
results without re-running training/inference.

Cross-references:
- [attempt_naming.md](attempt_naming.md) — the attempt folder structure these
  rules extend.
- [dvc_workflow.md](dvc_workflow.md) — `dvc add`, `dvc pull`, `dvc install`,
  retention.
- [model_readme_standard.md](model_readme_standard.md) — model README must link
  to the reference sample.

---

## Three tiers

Every attempt's outputs fall into exactly one tier. Wrong tier → either the
repo bloats or reviewers can't see the result.

| Tier | What it is | Where | Tracked by |
|---|---|---|---|
| **expected** | 2–3 **real-audio** ground-truth segments per attempt, extracted from the source recordings the attempt was trained on. Used as the comparison baseline in the Dev UI. NOT model outputs. | `<attempt>/expected/` | PNG + JSON in **git**, WAV in **DVC** |
| **showcase** | Author-curated **generated** samples (model outputs at specific seeds) that the developer wants other teammates to review. | `<attempt>/showcase/` | all in **DVC** (PNG + JSON + WAV) |
| **dev-artifacts-self-testing** | Ad-hoc developer runs for self-testing, training-time debug spectrograms, anything experimental. | `<attempt>/dev-artifacts-self-testing/` | **gitignored entirely** — never committed |

The asymmetry "PNG in git, WAV in DVC" is deliberate:
- PNGs render inline on GitHub diffs / PR review.
- JSON renders inline and is the audit trail.
- WAVs don't render in browsers anyway and bloat git history forever — DVC keeps git lean.

---

## Canonical seed

**Project-wide canonical seed is `42`** for generated artifacts (showcase and
the live Dev UI). It does not apply to `expected/` content — expected samples
are real recordings, not model output, and don't have a seed.

`showcase/` may use any seed the developer thinks is worth showing. Stem
format is `seed_<N>_<short_label>` (lowercase, snake_case label).

---

## File naming inside the attempt tiers

```
<attempt>/
├── expected/                        # real-audio ground truth (2–3 per attempt)
│   ├── real_<source_clip_id>.png            # mel spectrogram (rendered via the layer's own viz)
│   ├── real_<source_clip_id>.metadata.json  # source manifest ref, selection reason, audio stats
│   └── real_<source_clip_id>.wav.dvc        # → DVC blob of the WAV
├── showcase/                        # dev-curated generated samples for teammate review
│   ├── seed_<N>_<short_label>.png.dvc
│   ├── seed_<N>_<short_label>.metadata.json.dvc
│   └── seed_<N>_<short_label>.wav.dvc
├── dev-artifacts-self-testing/      # gitignored — ad-hoc developer self-test runs
└── .gitignore                       # excludes dev-artifacts-self-testing/  + *.wav  + *.png/.json in showcase/
```

Conventions:
- **Expected stem:** `real_<source_clip_id>` — traceable back to a row in the
  attempt's training manifest. No "seed" because the audio is not generated.
- **Showcase stem:** `seed_<N>_<short_label>` (lowercase, snake_case).
- Triplet `.wav` + `.png` + `.metadata.json` must always share the same stem.
- Expected WAV uses a `.wav.dvc` pointer so the blob is fetched on demand;
  PNG + JSON next to it are plain git so reviewers see them inline on GitHub.
- Showcase triplet is fully DVC-tracked (PNG + JSON + WAV).

### Placeholder layers

Layers whose registry status is `placeholder` (currently `layer_b`, `layer_d`,
`layer_e`) leave `expected/` empty — they either produce no audio (`layer_e`
outputs detector JSON), use curated assets that *are* the expected output
(`layer_b`), or consume other layers' output (`layer_d`). Document this in
the attempt's `README.md`.

---

## Metadata JSON contract

### Expected (real-audio) JSON

Produced by `acoustic_ai/scripts/extract_expected_samples.py`. Carries the
traceability back to the source manifest, not generation parameters:

```json
{
  "tier": "expected",
  "source": "real_audio",
  "source_kind": "clip_dir | webm_slice",
  "source_clip_id": "001_5392_clip001_s000",
  "source_manifest": "resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv",
  "selection_reason": "canonical 2019 spring-night exemplar (first manifest row)",
  "audio": { "sample_rate": 16000, "duration_s": 13.1, "rms": ..., "peak": ... },
  "source_metadata": {
    "caption": "spring night, ambient soundscape, …",
    "recording_date": "2019-09-01",
    "diel_bin": "night", "season": "spring",
    "env": { ... }  // weather etc. from the source manifest
  }
}
```

No `seed`, no `checkpoint`, no `handler_git_sha` — those don't apply to ground
truth.

### Showcase / generated JSON

Produced by `acoustic_ai/scripts/regenerate_samples.py`. The handler's
`generate()` dict is dumped verbatim, then enriched with traceability:

```json
{
  "tier": "showcase",
  "showcase_label": "variation_a",
  "seed": 43,
  "prompt": "...",
  "checkpoint": "model/candidates/lucas/layer-a-audioldm2-raw-smoke",
  "checkpoint_dvc_hash": "<md5 from .dvc pointer>",
  "audio": { "sample_rate": 16000, "duration_s": 10.0, "rms": ..., "peak": ... },
  "generated_at": "2026-05-27T12:00:00Z",
  "handler_git_sha": "<short SHA>"
}
```

The `checkpoint_dvc_hash` + `handler_git_sha` fields let reviewers tell at a
glance whether a showcase sample is stale relative to its source.

---

## When to regenerate / re-extract

**Re-extract `<attempt>/expected/`** (rare — only when the source manifest
selection changes):

```bash
# Edit PICKS in extract_expected_samples.py, then:
./acoustic_ai/.venv/bin/python acoustic_ai/scripts/extract_expected_samples.py
dvc add acoustic_ai/layers/<layer>/attempts/<id>/expected/*.wav
git add acoustic_ai/layers/<layer>/attempts/<id>/expected/   # PNG + JSON + .wav.dvc
git commit -m "data: refresh <attempt> expected samples"
git push && dvc push
```

**Regenerate showcase samples** whenever the model changes (handler, params,
checkpoint):

```bash
./acoustic_ai/scripts/regenerate_samples.py <layer> <attempt> \
    --showcase --seed 7 --label low_noise
# script prints the exact dvc add / git add commands to run.
```

The project does not currently enforce regeneration via a pre-push hook or
CI; the team relies on the `handler_git_sha` field in the JSON for after-the-
fact spotting.

---

## Branch-switching UX

Two one-time setup steps each developer must do after `git clone`:

```bash
# in project root
dvc install   # installs git post-checkout hook → auto-runs `dvc checkout`
```

Without `dvc install`, `git checkout other-branch` updates `*.dvc` pointers but
leaves the materialised files (`.wav`, big PNGs) pointing at the old hash on
disk. With it, branch switches are seamless.

`dvc install` is idempotent and writes to `.git/hooks/`. Re-run it after any
git hook upheaval.

---

## Storage discipline

- **Expected samples are the only WAVs allowed in git working tree under an attempt.** Showcase WAVs
  must be DVC-tracked too; nothing under `<attempt>/dev-artifacts-self-testing/` is ever committed.
- **Prune showcase samples liberally.** When a smoke attempt is superseded,
  delete its showcase folder in the same PR that supersedes it.
- **DVC garbage collection** (run monthly or before release):
  ```bash
  dvc gc --cloud --all-branches --all-tags   # cleans S3 of unreachable blobs
  ```
  This removes S3 objects no longer referenced by any branch tip. Don't run on
  a shallow clone — you'll over-prune.

---

## What does NOT belong under the artifact tiers

- Training-time loss curves, gradient norms, validation outputs across epochs
  — keep in `<attempt>/debug/` (gitignored) or attach to a wandb/tensorboard
  run.
- Source recordings or training data — those live under `<attempt>/data/`
  (DVC-tracked) per the existing convention.
- Generated audio that doesn't pair with a `metadata.json` — if it can't be
  reproduced, it shouldn't be an expected sample.
