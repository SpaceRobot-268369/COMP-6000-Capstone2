---
name: validate_model_attempt
description: Validate that a generation-layer model attempt's on-disk structure matches its declared kind (generative or retrieval) per conventions §5.5. Checks the registry entry, the model/ artifact layout, the retrieval index.json contract (JSON only, id/audio_path/attributes, every audio_path DVC-tracked), and git/DVC tracking. Invoke before opening a PR that adds or changes a model attempt, when reviewing an attempt's layout, or when the user asks to "check the attempt structure" / "validate a model".
---

# Skill: Validate Model Attempt

## Purpose

Confirm a **generation-layer** (Layer A / B / C) model attempt is laid out
correctly for its `kind`. This is the structural gate behind the
[pre_pr_checklist](pre_pr_checklist.md) §5 model-hygiene items, pulled out so
it can run on its own. Canonical rules:
[conventions §5.5](../context/conventions.md#55-generative-vs-retrieval-attempts)
and the [§1 repo tree](../context/conventions.md#1-repo-structure).

Scope: **generation layers only**. Layer D (mixer) and Layer E (analysis,
keyed by `head:`) are not generation models — they carry no `kind` and this
skill does not apply to them.

---

## Inputs

- An attempt id (e.g. `murphy__mvp_1__weather_stem_selector`) and its layer,
  **or** "all" to sweep every generation-layer attempt in `registry.yaml`.

Resolve paths from the id (conventions §2.7 — the same string in all places):
- code: `acoustic_ai/layers/<layer>/attempts/<id>/`
- artifact: the `checkpoint:` (generative) or `asset_bank:` (retrieval) path,
  which must live under `model/candidates/...` or `model/production/...`.

---

## Procedure

Stop at the first ❌, report it with the offending path, then continue the
remaining independent checks so the user sees the full picture.

### 0. Registry entry
- [ ] The attempt is a key under `layers.<layer>.attempts` in
      `acoustic_ai/registry.yaml`.
- [ ] It declares `kind: generative | retrieval`.
      *(Pre-migration retrieval entries may still set `checkpoint: null` and
      point at an in-attempt CSV bank — flag that as a migration TODO, not a
      hard fail, but the `kind` itself must be present.)*

### 1. Attempt code (both kinds)
- [ ] `code/handler.py` exists and exports `load()` + `generate()`.
- [ ] `README.md`, `params.yaml`, `__init__.py`, `.gitignore` present.

### 2. Artifact layout — branch on `kind`

**If `kind: generative`:**
- [ ] `checkpoint:` resolves under `model/candidates/...` or `model/production/...`.
- [ ] That folder has `README.md` (§6) + `params.yaml`.
- [ ] Weights (`*.pt|*.safetensors|*.bin|*.ckpt`) each have a matching `.dvc`.
- [ ] No raw weight binary is git-tracked (`git ls-files` shows only `*.dvc`).

**If `kind: retrieval`:**
- [ ] `asset_bank:` resolves under `model/candidates/...` or `model/production/...`.
      *(Until migrated, an attempt may instead point at an in-attempt
      `asset_index`/`retrieval_index` path — report as a migration TODO.)*
- [ ] Bank folder has `README.md` + `index.json`.
- [ ] **`index.json` is JSON — no `*.csv` index/manifest** is the bank's index.
- [ ] Audio lives under `media_asset_bank/` (not `audio/`), DVC-tracked, with
      no raw audio git-tracked.

### 3. Retrieval index contract (`kind: retrieval` only)
For each record in `index.json` `assets[]`:
- [ ] has `id` (unique within the bank), `audio_path`, and an `attributes` object.
- [ ] `audio_path` is **relative to the bank root** and resolves to a file
      under `media_asset_bank/` that is DVC-tracked (a `.dvc` covers it).

A ready-to-adapt check (run with `acoustic_ai/.venv/bin/python`):

```python
import json, sys
from pathlib import Path

bank = Path(sys.argv[1])                      # model/.../<attempt>  (bank root)
idx = json.loads((bank / "index.json").read_text())
bad = []
for a in idx["assets"]:
    for k in ("id", "audio_path", "attributes"):
        if k not in a:
            bad.append(f"{a.get('id','?')}: missing {k}")
    p = bank / a["audio_path"]
    if not p.exists() and not Path(str(p) + ".dvc").exists():
        # also acceptable: an ancestor dir is DVC-tracked (folder-level .dvc)
        if not any(Path(str(d) + ".dvc").exists() for d in p.parents):
            bad.append(f"{a['id']}: audio_path not DVC-tracked → {a['audio_path']}")
print("OK" if not bad else "\n".join(bad))
```

### 4. Tracking sanity (both kinds)
- [ ] Metadata (`*.json`, `*.yaml`, `*.md`, `*.dvc`) is git; binaries/audio are DVC.
- [ ] `dvc status -c` for the artifact is clean (pushed), so teammates can pull.

---

## Output

Report per attempt: `PASS` or a bullet list of ❌ failures + ⚠️ migration TODOs,
each with its path. On an "all" sweep, end with a one-line tally
(`N pass, M fail, K migration-pending`).

> This skill checks **structure**, not model quality. Listening tests, metrics,
> and audit notes live in the model README (§6) and are out of scope here.
