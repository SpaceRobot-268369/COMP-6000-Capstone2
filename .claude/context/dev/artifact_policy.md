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
| **reference** | Exactly one canonical PNG + JSON (+ optional WAV) per attempt, at the canonical seed. Regenerated whenever code/params/checkpoint change. | `<attempt>/samples/reference/` | PNG + JSON in **git**, WAV in **DVC** |
| **showcase** | Up to 3 author-curated extra seeds that reviewers should hear. | `<attempt>/samples/showcase/` | all in **DVC** (PNG + JSON + WAV) |
| **dev** | Ad-hoc developer runs, training-time debug spectrograms, anything experimental. | `<attempt>/samples/dev/` | **gitignored entirely** — never committed |

The asymmetry "PNG in git, WAV in DVC" is deliberate:
- PNGs render inline on GitHub diffs / PR review.
- JSON renders inline and is the audit trail.
- WAVs don't render in browsers anyway and bloat git history forever — DVC keeps git lean.

---

## Canonical seed

**Project-wide canonical seed is `42`.** Every attempt's `samples/reference/`
uses this seed. Do not vary the canonical seed per attempt or per developer —
the whole point of "canonical" is that reviewers know exactly what to compare
against.

If you need different starting conditions for review, add them under
`samples/showcase/` with a self-describing filename:
`samples/showcase/seed_7_quiet.{wav,png,metadata.json}`.

---

## File naming inside `samples/`

```
samples/
├── reference/
│   ├── seed_42.png                  # mel spectrogram
│   ├── seed_42.metadata.json        # prompt, params, audio stats, checkpoint hash
│   └── seed_42.wav.dvc              # → DVC blob of the WAV
├── showcase/
│   ├── seed_<N>_<short_label>.png.dvc
│   ├── seed_<N>_<short_label>.metadata.json.dvc
│   └── seed_<N>_<short_label>.wav.dvc
├── dev/                             # gitignored
└── .gitignore                       # excludes dev/  + *.wav  + *.png in showcase/
```

Conventions:
- Stem is `seed_<N>` for reference, `seed_<N>_<short_label>` for showcase. Lowercase, snake_case label.
- Triplet `.wav` + `.png` + `.metadata.json` must always share the same stem.
- Reference WAV uses a `.wav.dvc` pointer so the WAV blob is fetched on demand; the PNG/JSON next to it are plain git.

---

## Metadata JSON contract

The handler's `generate()` already returns a `metadata` dict — the regeneration
script just dumps it next to the PNG/WAV. At minimum it must carry:

```json
{
  "attempt":    { "layer": "layer_a", "id": "lucas__smoke_1__...", "stage": "smoke_1", ... },
  "prompt":     "...",
  "seed":       42,
  "checkpoint": "model/candidates/lucas/layer-a-audioldm2-raw-smoke",
  "checkpoint_dvc_hash": "<md5 from .dvc pointer>",
  "audio":      { "sample_rate": 16000, "duration_s": 10.0, "rms": ..., "peak": ... },
  "generated_at": "2026-05-27T12:00:00Z",
  "handler_git_sha": "<short SHA of the commit handler.py was at>"
}
```

The `checkpoint_dvc_hash` + `handler_git_sha` fields let reviewers tell at a
glance whether a reference sample is stale relative to its source.

---

## When to regenerate

Regenerate `samples/reference/` for an attempt whenever **any** of the
following change:

1. The attempt's `handler.py` or any module it imports inside its own folder.
2. The attempt's entry in `registry.yaml` (`params:` block, `checkpoint:`).
3. The checkpoint pointed at by `checkpoint:` (i.e. you re-trained and pushed a new `.dvc` pointer).

Workflow:

```bash
# from project root, with acoustic_ai/.venv active
./acoustic_ai/scripts/regenerate_samples.py <layer_id> <attempt_id>
git add acoustic_ai/layers/<layer>/attempts/<id>/samples/reference/
dvc add acoustic_ai/layers/<layer>/attempts/<id>/samples/reference/seed_42.wav
git add acoustic_ai/layers/<layer>/attempts/<id>/samples/reference/seed_42.wav.dvc
git commit -m "model: refresh smoke_1 reference sample"
git push && dvc push
```

For the **showcase tier**, generate with any seed/label and `dvc add` the
triplet by hand.

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

- **Reference samples are the only WAVs allowed in `samples/`.** Showcase WAVs
  must be DVC-tracked too; nothing under `samples/dev/` is ever committed.
- **Prune showcase samples liberally.** When a smoke attempt is superseded,
  delete its showcase folder in the same PR that supersedes it.
- **DVC garbage collection** (run monthly or before release):
  ```bash
  dvc gc --cloud --all-branches --all-tags   # cleans S3 of unreachable blobs
  ```
  This removes S3 objects no longer referenced by any branch tip. Don't run on
  a shallow clone — you'll over-prune.

---

## What does NOT belong under `samples/`

- Training-time loss curves, gradient norms, validation outputs across epochs
  — keep in `<attempt>/debug/` (gitignored) or attach to a wandb/tensorboard
  run.
- Source recordings or training data — those live under `<attempt>/data/`
  (DVC-tracked) per the existing convention.
- Generated audio that doesn't pair with a `metadata.json` — if it can't be
  reproduced, it shouldn't be a reference.
