# Model README Standard

Every trained model checkpoint folder must include a `README.md` that acts as
the durable model log and lightweight model card.

This policy applies to:

- `model/candidates/<member>/<run-id>/`
- `model/production/<role>/`

The README is git-tracked metadata. Checkpoint binaries remain DVC-tracked.

## Purpose

The README should let a developer understand what the model is, why it exists,
when it was trained or promoted, which data and settings produced it, and what
is known about its behavior without loading the binary checkpoint.

Do not use the README as the full hyperparameter source of truth. Store detailed
candidate hyperparameters in `params.yaml` and summarize only the important
values in the README.

## Audit rule

The `Results analysis / audit` section must be present but empty by default.
Do not invent audit conclusions. Fill it only after developers provide
evaluation notes, listening-test results, metrics, screenshots, review findings,
or other explicit analysis.

If no audit has been done, leave the section as:

```markdown
## Results analysis / audit

_Empty until developer evaluation notes are provided._
```

## Required sections

Use these sections for candidate and production model READMEs. For production
models, include the source candidate and promotion context.

```markdown
# <model-folder-name>

## Summary

- Owner:
- Layer / role:
- Status: candidate | production | deprecated
- Base model:
- Source candidate: <!-- production only; omit for candidates -->
- Trained at:
- Promoted at: <!-- production only; omit for candidates -->

## Purpose / hypothesis

<!-- Why this model exists, what behavior it is testing or serving, and what
success would look like. -->

## Dataset / inputs

- Dataset:
- Source clips / manifests:
- Filtering or preprocessing:
- Known data caveats:

## Training or promotion context

- Training command:
- Code branch / commit:
- Hardware:
- Runtime:
- Important settings:

For candidates, keep detailed settings in `params.yaml`. For production models,
record the source candidate and the validation / sign-off evidence that justified
promotion.

## Artifacts

- Checkpoint binaries:
- DVC pointer files:
- Params:
- Metrics:
- Sample outputs:
  - Reference (canonical seed): `<attempt>/samples/reference/seed_42.png` · `seed_42.metadata.json` · `seed_42.wav.dvc`
  - Showcase (optional extras): `<attempt>/samples/showcase/seed_<N>_<label>.{png,metadata.json,wav}.dvc`
  - Policy: [artifact_policy.md](artifact_policy.md)
- Related runbook or log:

## Results / metrics

<!-- Objective metrics, smoke-test outputs, sample paths, or "Not evaluated yet". -->

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

<!-- Failure modes, inappropriate use cases, unresolved issues, or deprecated
behavior. -->

## Follow-up actions

<!-- Next evals, fixes, comparisons, promotion tasks, or cleanup. -->
```

## Notes

- Keep entries factual and timestamped when possible.
- Link to runbooks, metrics, sample outputs, and issue logs instead of copying
  long analysis into the README.
- If a checkpoint is deprecated, keep its README and explain why it should not be
  used.
