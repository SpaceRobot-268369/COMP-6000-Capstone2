# Dev Stage Workflow — smoke → mvp/prod

Applies to **generation mode**. Analysis and Transformation modes — TODO (see [§ Out of scope](#out-of-scope-todo)).

Every attempt moves through stages declared by its folder name (`smoke_N` / `mvp_N` / `prod_N` — see [conventions.md § 2.4](../conventions.md)). Stage tokens are fixed at creation; promotion is a **new folder**, never a rename.

The core loop (goal → filter → audit → polish → train → artifacts → compare → wire-up) is the **same across stages**. What changes between stages is dataset scale, where training runs, and the quality bar — not the steps.

| Stage | Dataset | Trains on | Quality bar | Lives under |
|---|---|---|---|---|
| **smoke_N** | Tens of clips, one focused scene/event | Local Mac (MPS) | Recognizable target; throwaway-OK | `model/candidates/<member>/` |
| **mvp_N** | Full site dataset, 1–2 sites (today: `site_257_bowra-dry-a` only) | Server A (on-demand worker) | Listenable; teammate-reviewed | `model/candidates/<member>/` **or** `model/production/<role>/` if best current candidate |
| **prod_N** | Same as mvp_N (or curated superset) | Server A | Signed-off, release-tagged | `model/production/<role>/` |

> **Current project goal is mvp-stage models.** A successful mvp may be
> promoted into `model/production/<role>/` when it's the best available
> candidate for that role — full prod_N rigour (release tag, formal
> sign-off) is not required yet.

---

## The loop (same for every stage)

```
  ┌──────────────────────────────────────────────────────────────┐
  │  1. Goal set                                                  │
  │     - Pick ONE specific scene or event class.                 │
  │     - Write the expected result in plain words: what should   │
  │       a listener hear? What should the spectrogram look like? │
  │     - Record in the attempt README "Purpose / hypothesis".    │
  ├──────────────────────────────────────────────────────────────┤
  │  2. Data filtering policy                                     │
  │     - Codify selection rules in the manifest builder script   │
  │       (score threshold, duration, diel/season, species,       │
  │       site, exclusions). No ad-hoc notebook filtering.        │
  │     - Smoke: tens of clips, sharply focused.                  │
  │     - MVP : full dataset for the chosen site(s).              │
  ├──────────────────────────────────────────────────────────────┤
  │  3. Data audit                                                │
  │     - Sample-listen + render spectrograms.                    │
  │     - Check for misclassifications, anthropogenic bleed,      │
  │       clipping, silence, off-target species.                  │
  │     - Save audit notes in the attempt README.                 │
  ├──────────────────────────────────────────────────────────────┤
  │  4. Polish filtering policy (if needed) ───┐                  │
  │     - Tighten thresholds, drop bad clips.  │ loop to (3)      │
  │     - Re-build manifest, re-audit.         │ until clean      │
  ├────────────────────────────────────────────┴─────────────────┤
  │  5. Train                                                     │
  │     - Smoke → local (acoustic_ai/.venv, MPS).                 │
  │     - MVP   → submit job to Server A on-demand worker         │
  │       (see setup/server/on_demand_ai_worker.md).              │
  │     - Use canonical seed 42 for the baseline.                 │
  │     - Save checkpoint to                                      │
  │       model/candidates/<member>/<stage>__<slug>/.             │
  ├──────────────────────────────────────────────────────────────┤
  │  6. Generate artifacts                                        │
  │     - Extract 2–3 expected/ real-audio cases from the source  │
  │       manifest (ground-truth baseline).                       │
  │     - Generate showcase samples:                              │
  │         REQUIRED: seed_42_baseline + 1–2 variations           │
  │         (total 2–3) before opening a PR. Teammates review     │
  │         these directly, not by re-running the model.          │
  │     - Both via scripts in acoustic_ai/scripts/.               │
  ├──────────────────────────────────────────────────────────────┤
  │  7. Compare to expected                                       │
  │     - Side-by-side spectrograms + listen.                     │
  │     - Did the model produce the target?                       │
  │     - Record findings in the attempt README "Results" and the │
  │       checkpoint README "Results analysis / audit".           │
  ├──────────────────────────────────────────────────────────────┤
  │  8. Successful? Wire it up so the frontend can drive it.      │
  │     - Add/update the attempt's checkpoint under model/        │
  │       (candidates/ or production/<role>/ for promoted mvp).   │
  │     - Add/update the entry in acoustic_ai/registry.yaml       │
  │       (REQUIRED at every stage — smoke included). Set/keep    │
  │       the layer's `default:` if this attempt should be the    │
  │       one the frontend dropdown selects on first load.        │
  │     - Update acoustic_ai/requirements.txt if any new Python   │
  │       dep was introduced.                                     │
  │     - Update any layer-specific docs / runbooks.              │
  │     - Verify end-to-end: launch the FastAPI server natively,  │
  │       pick the attempt in the frontend dropdown, send a       │
  │       generate request with a fresh seed, confirm a WAV       │
  │       comes back.                                             │
  │                                                               │
  │     If NOT successful:                                        │
  │     - Bad audio quality → back to (2)/(4): re-filter,         │
  │       re-audit.                                               │
  │     - Wrong method → abandon this attempt and start a new     │
  │       one: <member>__<stage>_{N+1}__<new_slug>. Do NOT        │
  │       overwrite the failed attempt — it stays as a            │
  │       historical record.                                      │
  └──────────────────────────────────────────────────────────────┘
```

### Stage-specific notes

**Smoke**
- One question per smoke. A different species or scene = a new smoke (`smoke_{N+1}`).
- Failed smokes stay on disk — they're the team's record of what was tried.
- Don't optimize for quality. Recognizable target = done; push to mvp.

**MVP**
- Dataset covers the full chosen site(s). Today the project has only `site_257_bowra-dry-a`; expand to a second site when one comes online.
- Training jobs go to **Server A**'s on-demand AI worker (see [setup/server/on_demand_ai_worker.md](../setup/server/on_demand_ai_worker.md)) — local Mac is for smoke only.
- Filtering policy must be frozen before the headline training run.
- Showcase review by at least one teammate before merge.
- A successful mvp checkpoint may be placed under `model/production/<role>/` when it's the best current candidate for that role — that's the project's near-term ceiling.

**Prod**
- Same loop as mvp; difference is formal sign-off, release tagging, and an explicit `prod_N` attempt folder. Not a current project goal.

---

## Registry entry (required at every stage)

Every attempt — smoke included — must be declared in
`acoustic_ai/registry.yaml` so the FastAPI server can load its
`code/handler.py` and the frontend dropdown can target it.

Minimal entry:

```yaml
layers:
  layer_<x>:
    default: <member>__<stage>__<slug>      # optional; sets dropdown default
    attempts:
      <member>__<stage>__<slug>:
        label: "Layer <X> — <human description> (<stage>)"
        checkpoint: model/candidates/<member>/<stage>__<slug>/
        # or, for promoted mvp: model/production/<role>/
```

The attempt ID, folder name, and checkpoint folder name all share the
same `<member>__<stage>__<slug>` string — see [conventions.md § 2.7](../conventions.md).

---

## Out of scope (TODO)

This workflow covers **generation mode** only. The other two modes need
their own loops, written up when work on them begins:

- **Analysis mode** — TODO. Likely centres on detector training/eval and
  ground-truth annotation rather than generated-sample review.
- **Transformation mode** — TODO. Likely a hybrid: analysis on input
  audio + generation conditioned on transformed environmental params.

---

## Cross-references

- Naming, paths, tracking → [conventions.md](../conventions.md)
- Git/branch/commit rules → [git_workflow.md](git_workflow.md)
- DVC commands → [dvc_workflow.md](dvc_workflow.md)
- Server A topology → [../setup/server/on_demand_ai_worker.md](../setup/server/on_demand_ai_worker.md)
- Worked smoke examples → [../ai/runbooks/](../ai/runbooks/)
