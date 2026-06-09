# Layer D - Multi-Clip Mix

## Status

**Implemented at handler level; not yet the default Layer D attempt.** This
attempt implements the Layer D mixer contract for multiple placed clips per
layer. It accepts one ambient bed, zero or more weather clips, and zero or more
event clips, places discrete clips on an explicit timeline, applies per-layer
and per-clip gain staging, preserves the v1 event cleanup path, and returns
traceable mix metadata.

This attempt is registered in `acoustic_ai/registry.yaml` and
`registry.orchestrate_generation` can route to it when
`layer_d_attempt="songke__mvp_2__multi_clip_mix"` is selected. It is still not
the default Layer D attempt. The LLM parser and frontend still need follow-up
work before this becomes the normal user-facing path.

For compatibility while the rest of the pipeline catches up, the handler still
accepts the v1 single-stem parameters (`weather_wav_bytes`, `event_wav_bytes`,
and `event_start_s`) in addition to the v2 list contract.

The MVP mixer this supersedes is documented in
[`songke__mvp_1__layered_mix/README.md`](../songke__mvp_1__layered_mix/README.md).
Read that first for the existing fixed-format / gain-staging decisions; they
carry forward unchanged unless this card says otherwise.

---

## 1. Purpose

Layer D mixes the audio produced by Layers A/B/C into one coherent file and
makes the result feel like a real recording. The new capability over the MVP is
arrangement: instead of one weather stem and one event stem, Layer D can lay
down a timeline of clips at caller-specified times.

| Layer | What it hands Layer D | Multiplicity |
|---|---|---|
| A - Ambient | One ambient bed | exactly 1 |
| B - Weather | Wind / rain continuous beds and/or thunder discrete clips | 0..N, optional |
| C - Events | Species calls | 0..N, optional |

Layer D stays dumb on purpose: it does not interpret prompts, environmental
conditions, species, or weather semantics. It places and mixes the bytes it is
given. All "where / how often / which species" decisions are made upstream and
passed to Layer D as concrete numbers.

---

## 2. Division of Responsibility

The main rule of this contract:

> The LLM computes specifics; the mixer consumes specifics.

- "Frequency" / repetition is not a parameter the mixer reasons about. The LLM
  or orchestrator expands cadence into an explicit list of onset times. A clip
  that recurs 4 times is one clip with 4 onsets.
- Overlap is allowed and expected. The mixer sums overlapping audio and relies
  on the final peak ceiling to protect against clipping. It does not run a
  collision solver and does not reshuffle onsets.
- The mixer owns format normalization, fitting beds to duration, placing
  discrete clips at given onsets, summing, gain staging, peak protection,
  random-onset fallback when a list is `null`, export, and explanation JSON.

---

## 3. Input Contract

Layer D's handler receives one mix request. Audio is passed as in-memory WAV
bytes. The v2 list contract is:

```jsonc
{
  "duration_s": 30,
  "placement_seed": 42,

  "ambient_wav_bytes": "<bytes>",

  "weather_clips": [
    {
      "wav": "<bytes>",
      "weather_type": "thunder",
      "continuous": false,
      "onsets_s": [8.0, 21.0],
      "gain_db": null,
      "change": null
    }
  ],

  "event_clips": [
    {
      "wav": "<bytes>",
      "species": "tawny_frogmouth",
      "onsets_s": [5.0, 12.5, 19.0],
      "gain_db": null
    }
  ]
}
```

### Field Semantics

- `ambient_wav_bytes`: exactly one bed. It is looped or trimmed to
  `duration_s` exactly as the MVP does today.
- `weather_clips[].continuous`:
  - `true`: a bed such as wind or steady rain. It is looped/crossfaded to the
    full duration. `onsets_s` is ignored.
  - `false`: a discrete weather sound such as thunder. It is placed at each
    resolved onset and is not looped to fill duration.
- `weather_clips[].onsets_s` / `event_clips[].onsets_s`: list of start times
  in seconds on the final timeline. `null` triggers the random fallback in
  section 5. Copies that run past `duration_s` are trimmed at the end.
- `event_clips`: always discrete. There is no `continuous` flag.
- `gain_db`: optional per-clip gain override. `null` means use the layer
  default.
- `change`: reserved placeholder for future weather transitions. It is carried
  through metadata but no transition behavior is implemented.

Legacy compatibility parameters still accepted by the handler:

- `weather_wav_bytes`
- `event_wav_bytes`
- `event_start_s`

---

## 4. Gain Staging

Current defaults:

| Layer / role | Starting gain |
|---|---|
| Ambient | 0 dB |
| Weather | -2 dB |
| Event | -8 dB |

Every clip gets that layer default unless its `gain_db` field is set. When set,
the per-clip value overrides the layer default for that clip only.

Implementation detail: the v2 handler pre-scales override clips by the delta
between clip gain and layer gain, then passes the timeline through the existing
v1 layer-gain mixer. This preserves the existing v1 mixer path while producing
the intended final per-clip gain.

The same final peak ceiling (`0.95`) and runtime/export format from the MVP
apply after summing:

- Runtime: 22,050 Hz mono float32
- Export: PCM16 WAV

Event band-pass and the event activity envelope from the MVP carry forward and
apply per discrete event clip through the existing event timeline path.

---

## 5. Random Onset Fallback

When an `onsets_s` list is `null`, Layer D assigns the onset itself:

- Events: place one copy at a random onset within
  `[0, duration_s - clip_length]`.
- Discrete weather: same single-random-onset behavior.
- If the clip is longer than `duration_s`, onset is fixed at `0.0`.
- Random draws use `placement_seed`, so the same request reproduces the same
  arrangement.
- Multiple omitted clips each get an independent draw from the same seeded RNG.
  Overlaps are allowed and summed.

Layer D still ignores the generation `seed`; A/B/C own synthesis and retrieval
seeds. `placement_seed` is separate and only affects random placement fallback.

---

## 6. Limits

- `duration_s <= 30 s` remains the intended stage limit because Layer B
  currently caps at 30 s. This handler itself does not promote v2 as the
  default route yet.
- No maximum clip count is enforced by the mixer; upstream is trusted to emit a
  sane arrangement.
- Overlapping onsets are summed, not resolved or rejected.

---

## 7. Reserved: Weather Change / Transitions

`weather_clips[].change` is a reserved placeholder, intentionally `null` for
now. It is carried through metadata untouched. No transition behavior is
implemented in this attempt.

---

## 8. Output Contract

Unchanged top-level handler payload:

- `wav_bytes`: final 22,050 Hz mono PCM16 WAV.
- `mel_db`: mel spectrogram preview of the mix.
- `metadata.layer_d`: explanation dict.

The explanation is extended with:

- `input_contract.weather` and `input_contract.events`: raw resolved contract
  rows used by the handler.
- `placed_clips.weather` and `placed_clips.events`: reviewer-facing summaries
  for reconstructing the timeline.

Each placed clip summary includes:

- resolved `onsets_s`
- `placement_random`
- `placement_seed` when random placement was used
- `applied_gain_db`
- `gain_override`
- layer default gain
- source duration
- placement count
- weather `continuous` flag
- discrete weather placement rows
- carried `change` placeholder for weather

The explanation must let a reviewer reconstruct exactly where every sound was
placed and why.

---

## 9. Implementation Checklist

1. Done - create `code/` mirroring the MVP layout:
   `handler.py`, `audio_mixer.py`, `audio_format.py`, `audio_metrics.py`.
2. Done - add `params.yaml`.
3. Done - implement handler support for lists of placed weather/event clips.
4. Done - support continuous weather beds and discrete weather clips.
5. Done - support multiple event clips and repeated event onsets.
6. Done - add `placement_seed` plumbing and random fallback.
7. Done - add per-clip `gain_db` override.
8. Done - extend explanation JSON with `input_contract` and `placed_clips`.
9. Done - register the attempt in `acoustic_ai/registry.yaml`.
10. Done - add handler-level unit coverage, including a main multi-clip
    contract test.
11. Done - wire `registry.orchestrate_generation` to pass
    `weather_clips` / `event_clips` when this Layer D attempt is selected.
12. Follow-up - update the LLM parser contract per
    [`prompt_parser_policy.md`](../../../../../.claude/context/ai/prompt_parser_policy.md)
    to emit `onsets_s` / `continuous` per clip.
13. Follow-up - listen to real A/B/C multi-clip mixes and retune defaults if
    needed before promoting this attempt to the Layer D default.
