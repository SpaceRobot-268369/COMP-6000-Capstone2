# Caption Schema Log — Layer A

Append-only log of the caption template used at training and inference time for
each attempt. Whenever the template changes, add a new entry with the rationale
and the attempt it landed in. Source of truth for "why does this attempt use
*this* caption" arguments.

---

## v1 — `lucas__smoke_1__audioldm2_spring_night`, `lucas__smoke_2__audioldm2_insects`

Fixed prompt per attempt, hardcoded server-side, no template variables.

- smoke_1: `night spring ambient soundscape, Bowra dry woodland, Australia, no music, no machinery`
- smoke_2: `summer afternoon insect chorus, Bowra dry woodland, Australia, no music, no machinery`

**Rationale.** One scene per LoRA — caption never needed conditioning.

---

## v2 — `lucas__mvp_1__audioldm2_all_conditioned`

Per-row caption assembled at dataset-build time from env data:

```
{diel} {season} ambient soundscape, Bowra dry woodland, Australia,
{temp_bucket} ({temp}C), {humidity_bucket}, {wind_bucket},
recorded {YYYY-MM-DD}, no music, no machinery
```

Axes: 4 diel × 4 season × 5 temp × 3 humidity × 3 wind × ~140 dates.
Combinatorial space ~80–150 distinct strings after weather correlation.

**Rationale.** First conditioned attempt — included every observable axis to
let the model discover which ones matter.

**Outcome.** Generated samples blurred per-cell character (see
`acoustic_ai/layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/DEVLOG.md`).
Suspected contributors: shared-LoRA capacity dilution AND high-cardinality date
token acting as noise.

---

## v3 — planned for `lucas__mvp_2__per_cell_loras` (and Phase 1B shared-LoRA bake-off)

Drop the date token:

```
{diel} {season} ambient soundscape, Bowra dry woodland, Australia,
{temp_bucket} ({temp}C), {humidity_bucket}, {wind_bucket},
no music, no machinery
```

**Rationale.**
- ~140 unique date tokens against 1,082 clips ≈ one date appears in ~8 clips
  on average — the text encoder has very weak signal to learn what a date
  means.
- At inference time we always pass an arbitrary placeholder date, so the
  signal can't help even if learned.
- Removing it shrinks the effective caption space to ~40-60 strings, which
  is a better fit for the data scale.

**Action.** Rebuild the conditioned dataset under a new builder flag
`--no-date-in-caption`, producing a new DVC artifact alongside (not replacing)
v2. Attempts using v3 are roll-backable to v2 by checking out the prior
`.dvc` pointer.

**Phase 1A (`lucas__mvp_1_1__spring_night_replica`) keeps v2** intentionally
— that attempt's purpose is to validate the data filter pipeline as-is, not
to confound with a caption-template change. Date-drop lands in Phase 1B and
MVP-2.
