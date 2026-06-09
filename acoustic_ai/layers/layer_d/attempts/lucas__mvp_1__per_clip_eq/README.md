# Layer D — Per-Clip EQ & Gain

## Status

**Design only — not implemented, not registered.** This attempt extends the
multi-clip mixer
([`songke__mvp_2__multi_clip_mix`](../songke__mvp_2__multi_clip_mix/README.md))
with a **per-clip equalizer (EQ)** stage, complementing the per-clip `gain_db`
that v2 already supports. Read the v2 card first; every decision there
(timeline placement, gain staging, peak ceiling, random fallback, explanation
JSON) carries forward unchanged unless this card overrides it.

This document is a future-works specification. No `code/`, `params.yaml`, or
registry entry exists yet — those land per the checklist in section 8.

---

## 1. Motivation

Layers A/B/C are generated/retrieved in isolation, so their clips don't always
"sit" in the mix the way a real field recording would:

- A synthetic bird call can be spectrally too bright/dry compared to the
  ambient bed it's placed over.
- A weather bed and the ambient bed can fight for the same low-mid energy and
  turn muddy when summed.
- Distance/proximity cues (a far call vs. a near one) aren't expressed — in a
  real recording, distance rolls off highs and reduces level.

Gain alone (v2) controls *level*. EQ controls *spectral shape*. Together they
let the mixer make each clip read as belonging to the same acoustic space,
which is the realism goal of this attempt.

---

## 2. Division of Responsibility (unchanged rule)

> The LLM/orchestrator computes specifics; the mixer consumes specifics.

The mixer does **not** decide that a bird is "distant" or that a bed is
"muddy." Upstream emits concrete EQ band parameters; Layer D applies them.
Layer D stays dumb — it EQs, gains, places, sums. All "which clip needs what
spectral shape" reasoning happens upstream (LLM parser / orchestrator), exactly
as cadence and onsets do in v2.

---

## 3. Input Contract (additive, backward compatible)

Each clip object (`weather_clips[]`, `event_clips[]`) gains one optional field,
`eq`, alongside the existing `gain_db`:

```jsonc
{
  "wav": "<bytes>",
  "species": "tawny_frogmouth",
  "onsets_s": [5.0, 12.5],
  "gain_db": -8.0,
  "eq": {
    "bands": [
      { "type": "high_pass",  "freq_hz": 120,  "q": 0.707 },
      { "type": "peaking",    "freq_hz": 3000, "q": 1.2, "gain_db": -3.0 },
      { "type": "high_shelf", "freq_hz": 8000, "gain_db": -4.0 }
    ]
  }
}
```

### Field Semantics

- `eq: null` or omitted → **no EQ applied**, byte-identical to v2 behavior.
  This is what keeps the contract backward compatible: every existing v2
  request remains valid and unchanged.
- `eq.bands[]` → an ordered list of biquad filters applied **in listed order**
  to that clip before placement and summing.
- Band `type` ∈ `{ high_pass, low_pass, low_shelf, high_shelf, peaking }`.
- `freq_hz` — required for every band (center/corner frequency).
- `q` — quality factor, default `0.707` (Butterworth). Ignored by shelf types
  that don't use it.
- `gain_db` — required for `low_shelf` / `high_shelf` / `peaking`; ignored for
  `high_pass` / `low_pass`.

`ambient_wav_bytes` MAY also accept an optional sibling `ambient_eq` (same
`{ bands: [...] }` shape) so the bed can be carved to make room for events.
Default off; TBD whether to ship in this attempt or defer.

---

## 4. Processing Order (per clip)

```
raw clip → EQ bands (in listed order) → per-clip gain_db → place at onsets → sum → peak ceiling (0.95)
```

- EQ runs **before** gain, so `gain_db` still means "final level of this clip"
  and is not perturbed by shelf boosts/cuts.
- This sits in front of the existing v1 layer-gain mixer path that v2 reuses;
  the v2 gain pre-scaling trick (section 4 of the v2 card) is untouched.
- The v2 **event band-pass** and **activity envelope** still apply per event
  clip. `eq` stacks *after* that existing band-pass — it does not replace it.
- Runtime stays **22,050 Hz mono float32**; export stays **PCM16 WAV**.

---

## 5. Realism Presets (orchestrator-side, not mixer-side)

To keep the LLM's job tractable, the orchestrator expands a small named
vocabulary into concrete `bands`. **The mixer never sees these names — only the
resolved bands.** Initial proposed presets:

| Preset | Intent | Rough EQ |
|---|---|---|
| `distant` | far source | high-shelf cut ~6–8 dB @ 6 kHz + gentle low-pass |
| `muffled` | occluded / behind foliage | low-pass ~4 kHz |
| `near` / `bright` | close source | mild high-shelf boost |
| `de_mud` | carve low-mid to fit the bed | peaking cut ~2–4 dB @ 250–500 Hz |
| `rumble_cut` | remove sub-rumble from synthetic clips | high-pass @ 80–120 Hz |

This table is the canonical home only as a sketch; the binding version lives in
[`prompt_parser_policy.md`](../../../../../.claude/context/ai/prompt_parser_policy.md)
once the parser learns to emit `eq`.

---

## 6. Output Contract

Unchanged top-level handler payload (`wav_bytes`, `mel_db`,
`metadata.layer_d`). The v2 `placed_clips` explanation rows gain:

- `applied_eq` — the resolved `bands` actually applied (post-default-fill).
- `eq_preset` — the upstream preset name, if one was used (else `null`).

A reviewer must be able to reconstruct *what spectral change was made to each
clip and why* — the same traceability standard v2 holds for placement.

---

## 7. Limits / Non-Goals

- **Static EQ only.** No time-varying / automated EQ sweeps (those would pair
  with the reserved v2 `change` transition field — out of scope here).
- No dynamics processing (compression, limiting) beyond the existing peak
  ceiling.
- **No per-onset EQ variation** — `eq` applies to the whole clip across all its
  onsets. A clip that needs two distances is two clips.
- Filters are standard biquads (e.g. `scipy.signal` / `torchaudio`
  implementations); no linear-phase / FFT EQ in this attempt.

---

## 8. Implementation Checklist (all TODO)

1. TODO — fork `code/` from `songke__mvp_2__multi_clip_mix`; add an
   `audio_eq.py` biquad module (HP/LP/low-shelf/high-shelf/peaking).
2. TODO — apply `eq.bands` per clip in `audio_mixer.py`, before gain staging.
3. TODO — optional `ambient_eq` support (decide ship vs. defer).
4. TODO — extend explanation JSON with `applied_eq` / `eq_preset`.
5. TODO — unit coverage: per-band frequency-response sanity, `null`/omitted =
   passthrough, band ordering, ambient_eq, peak ceiling still respected.
6. TODO — add `params.yaml` (EQ defaults: default `q`, preset band tables).
7. TODO — register in `acoustic_ai/registry.yaml`, kept **non-default**.
8. TODO — orchestrator: preset → bands expansion specced in
   [`prompt_parser_policy.md`](../../../../../.claude/context/ai/prompt_parser_policy.md).
9. TODO — listen to real A/B/C mixes with EQ on; tune presets before any
   promotion to the Layer D default.
