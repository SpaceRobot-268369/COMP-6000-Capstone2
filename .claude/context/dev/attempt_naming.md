# Attempt Naming Rules

> Part of the project [conventions index](conventions.md). This is the
> canonical doc for attempt + checkpoint folder naming.

Single source of truth for how AI-module attempts are named on disk, in the
model registry, and on checkpoint folders under `model/candidates/`. This
applies to every layer (A–E) under `acoustic_ai/layers/`.

---

## The format

```
<member>__<stage>__<slug>
```

Three segments separated by **double underscore** (`__`). Each segment uses
`[a-z0-9_]+` so the whole string is a legal Python module name — the registry
loads `handler.py` via normal `import` and sibling modules inside an attempt
work the same way. (Single underscores are still meaningful inside a segment
because the separator is the *double* underscore.)

We sometimes write stage tokens with a dash in conversation ("smoke-1"); the
**canonical on-disk form uses an underscore** (`smoke_1`). Same for layer
codes: `layer-a` is the shorthand we use in prose; the folder under
`acoustic_ai/layers/` is `layer_a`. The registry IDs and URL paths follow the
on-disk form.

### `<member>`

The owning developer's git username, lowercase. e.g. `lucas`, `alice`, `ben`.

One folder per member, one folder per attempt — never overwrite another
member's attempts (matches the existing rule in CLAUDE.md → "Model checkpoint
layout").

### `<stage>`

The attempt's lifecycle stage. One of:

| Token (on disk) | Shorthand | Meaning |
|---|---|---|
| `smoke_N` | smoke-N | Exploratory work. Tiny dataset, throwaway-OK, no quality guarantees. Used for any pre-MVP experiment, including approaches that were tried and abandoned. |
| `mvp_N`   | mvp-N   | Targeting the MVP demo. Trained on the full / production-shaped dataset. Must have a working `handler.py` for the registry. |
| `prod_N`  | prod-N  | Promoted candidate. Sign-off, validation, and a `model/production/<role>/` slot required. See CLAUDE.md → "Attempts and checkpoints". |

`N` is a 1-based integer that uniquely numbers attempts within a layer for
that developer. It does **not** have to be chronological — pick the next
available number that isn't already in use. Once assigned, a number never
changes (renaming would break the registry, frontend dropdown state, and
checkpoint paths).

### `<slug>`

Short snake_case description of the method/variant. Rules:

- Lowercase `[a-z0-9_]+`.
- Don't repeat the layer letter (the parent folder already says `layer_a/`).
- Don't repeat the stage token (`smoke`, `mvp`, `prod`).
- Don't repeat the member name.
- Describe the **method** or **distinguishing feature**, not the dataset
  version — e.g. `audioldm2_spring_night`, not `bowra_v3`.

---

## Examples

| ✅ Good | ❌ Bad | Reason |
|---|---|---|
| `lucas__smoke_1__audioldm2_spring_night` | `lucas-smoke1-audioldm2` | single `-` separator, not Python-importable |
| `lucas__smoke_2__audioldm2_insects` | `layer_a_audioldm2_insects_smoke` | wrong order; layer prefix doesn't belong in the slug |
| `alice__mvp_1__audiogen_multi_species` | `alice__audiogen` | missing stage |
| `lucas__prod_1__audioldm2_bowra_v1` | `lucas/prod/audioldm2_bowra` | `/` breaks Python imports |
| `ben__smoke_3__clap_diffusion` | `ben__smoke_1__clap_diffusion` | `smoke_1` already used by ben in this layer |

---

## Where the name appears

The same string is used in **four** places. They must stay in sync.

1. **Attempt folder.**
   ```
   acoustic_ai/layers/layer-<X>/attempts/<member>__<stage>__<slug>/
   ```

2. **Checkpoint folder under `model/`.** For `smoke-*` and `mvp-*`:
   ```
   model/candidates/<member>/<stage>__<slug>/
   ```
   For `prod-*`, the checkpoint additionally gets a promoted slot at
   `model/production/<role>/` after sign-off — that slot has its own naming
   (`<role>`), independent of the attempt slug.

3. **Registry ID.** In `acoustic_ai/registry.yaml`, the YAML key for the
   attempt is the folder name verbatim:
   ```yaml
   layers:
     layer-a:
       attempts:
         lucas__smoke-1__audioldm2-spring-night:
           label: "Layer A — Spring night (smoke-1)"
           ...
   ```

4. **Frontend dropdown value.** The React dropdown sends the same string to
   the backend, which forwards to
   `POST /layers/<layer>/attempts/<attempt-id>/generate`.

---

## What lives inside an attempt folder

Each attempt is **self-contained** — no shared `common/` folder, no shared
preprocess scripts, no shared data caches. The reason: members work
independently, and an attempt's reproducibility shouldn't depend on
files outside its own folder.

```
layers/layer_<X>/attempts/<member>__<stage>__<slug>/
├── README.md           # required: model card + run log + link to a showcase spectrogram
├── params.yaml         # optional: per-attempt hyperparameters
├── __init__.py         # makes the attempt an importable package
├── code/               # required: all Python source for this attempt
│   ├── __init__.py
│   ├── handler.py      # required: implements load() + generate(seed, **kw)
│   ├── train.py        # optional: training entrypoint
│   ├── sample.py       # optional: standalone sampling/debug entrypoint
│   ├── preprocess.py   # optional
│   └── dataset.py      # optional
├── data/               # attempt-local derived data (DVC-tracked)
├── precompute/         # attempt-local precompute scripts (optional)
├── debug/              # attempt-local diagnostics (optional, often gitignored)
├── expected/           # real-audio ground truth; one subdir per case (real_<clip_id>/)
│   └── real_<clip_id>/{audio.wav.dvc, spectrogram.png, metadata.json}
├── showcase/           # author-curated generated samples; one subdir per seed
│   └── seed_<N>_<label>/{audio.wav, spectrogram.png, metadata.json} (all DVC-tracked)
├── dev-artifacts-self-testing/   # ad-hoc dev scratch; folder tracked via .gitkeep, contents gitignored
│   └── .gitkeep
└── .gitignore          # ignores dev-artifacts-self-testing/* (except .gitkeep), *.wav, showcase PNG/JSON
```

The `expected/` / `showcase/` / `dev-artifacts-self-testing/` tiers, the
per-case subdir convention, and the PNG metadata baking (overlay + tEXt
chunks) are all governed by [artifact_policy.md](artifact_policy.md).
Canonical seed across the project is `42` (applies to showcase + the
live Dev UI, **not** to expected — that's real audio). Regenerate via
`acoustic_ai/scripts/regenerate_samples.py`.

The handler is loaded by the server as
`layers.<layer>.attempts.<id>.code.handler` — keep `code/` as the import root
for the attempt's Python sources.

Two attempts that share a method (e.g. AudioLDM2 spring-night vs.
AudioLDM2 insects) **duplicate** the training/sampling code rather than
import from each other. Each attempt evolves independently; duplication
is intentional.

### Required: `handler.py`

The registry-facing interface that the FastAPI server calls. Minimal
contract:

```python
# layers/layer-a/attempts/<id>/handler.py
from pathlib import Path

def load(checkpoint_dir: Path, params: dict) -> object:
    """One-time load. Return whatever generate() needs (model, pipeline, …)."""

def generate(state, seed: int | None, **runtime_params) -> dict:
    """Return {'wav_bytes': bytes, 'metadata': dict, 'mel_db': np.ndarray|None}."""
```

`state` is the value returned by `load()`. The server caches it per
attempt-id so `load()` runs once.

### Required: `README.md`

Follows [model_readme_standard.md](model_readme_standard.md). Use it as
the experiment + checkpoint log for this specific attempt.

---

## Promoting an attempt

Lifecycle moves are explicit folder copies, never renames:

- **`smoke-N` → `mvp-N`:** create a new `<member>__mvp-N__<slug>/`. Don't
  rename the smoke folder — the smoke attempt stays as a historical record.
- **`mvp-N` → `prod-N`:** create `<member>__prod-N__<slug>/` and, in the
  same PR, promote the checkpoint to `model/production/<role>/` with
  sign-off recorded in the README audit section.

The registry's `default:` key for a layer is what the frontend dropdown
selects on first load. Bumping it is part of promotion.

---

## Migration note (initial restructure)

The original `acoustic_ai/modules/<layer-name>/` layout did not encode
stage tokens. During the initial restructure, the existing AudioLDM2
smoke runs keep their `smoke-1` / `smoke-2` numbering (matching the
existing project docs), and older abandoned approaches (CLAP, VAE
baseline) get higher numbers (`smoke-3`, `smoke-4`) per CLAUDE.md →
"Notes / Legacy attempts".

`model/candidates/<member>/<old-name>/` directories created before this
restructure do not match `<stage>__<slug>` — renaming them requires
coordinated `dvc mv`. Until that lands, the registry's `checkpoint:`
field points at the existing paths.
