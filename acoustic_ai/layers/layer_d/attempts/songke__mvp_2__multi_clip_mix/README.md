# Layer D — Multi-Clip Mix (design)

## Status

**Design only — no code yet.** This card is a self-contained handoff spec for
the next Layer D implementer. It firms up the input/output contract for a Layer
D mixer that accepts **multiple placed clips per layer** (the current
`songke__mvp_1__layered_mix` accepts exactly one stem per layer). Nothing here
is wired into the orchestrator or the LLM parser yet; this card defines the
target contract those changes will build toward.

The MVP mixer this supersedes is documented in
[`songke__mvp_1__layered_mix/README.md`](../songke__mvp_1__layered_mix/README.md).
Read that first for the existing fixed-format / gain-staging decisions — they
carry forward unchanged unless this card says otherwise.

---

## 1. Purpose

Layer D mixes the audio produced by Layers A/B/C into one coherent file and
makes the result feel like a real recording. The only new capability over the
MVP is **arrangement**: instead of one weather stem and one event stem, Layer D
now lays down a *timeline* of clips at caller-specified times.

| Layer | What it hands Layer D | Multiplicity |
|---|---|---|
| A — Ambient | One ambient bed | exactly 1 (the "audio bed") |
| B — Weather | Wind / rain (continuous beds) and/or thunder (discrete) | 0…N, optional |
| C — Events | Species calls | 0…N (multiple species, or repeated calls of one species) |

Layer D stays **dumb on purpose**: it does not interpret prompts, environmental
conditions, species, or weather semantics. It places and mixes the bytes it is
given. All "where / how often / which species" decisions are made upstream by
the LLM parser (powered by skills) and passed to Layer D as concrete numbers.

---

## 2. Division of responsibility (read this first)

The single most important rule of this contract:

> **The LLM computes specifics; the mixer consumes specifics.**

- "Frequency" / repetition is **not** a parameter the mixer reasons about. The
  LLM expands a desired cadence into an **explicit list of onset times** and
  passes that list. A clip that recurs 4 times is one clip with 4 onsets.
- **Overlap is allowed and expected.** Real soundscapes layer sound — a bird
  calls while thunder rolls, two species answer each other. The LLM may emit
  onsets that overlap (across clips, or even within one clip), and the mixer
  simply **sums** the overlapping audio; the final peak ceiling (§4) protects
  against clipping. The mixer runs no collision solver and never rejects or
  reshuffles overlapping onsets.
- The mixer **only** owns: format normalization, fitting beds to duration,
  placing discrete clips at given onsets, summing, gain staging, peak
  protection, random-onset fallback when a list is `null`, export, and the
  explanation JSON.

---

## 3. Input contract

Layer D's handler receives one mix request. Audio is passed as in-memory WAV
bytes (as today). The shape:

```jsonc
{
  "duration_s": 30,            // <= 30 for this stage (see §6)
  "placement_seed": 42,        // NEW — seeds random onset fallback only (§5)

  "ambient": {                 // exactly one, required
    "wav": "<bytes>"
  },

  "weather": [                 // 0..N, optional (omit/empty for no weather)
    {
      "wav": "<bytes>",
      "weather_type": "thunder",   // label for the explanation only
      "continuous": false,         // true  -> looped bed (wind, steady rain)
                                   // false -> discrete, placed at onsets
      "onsets_s": [8.0, 21.0],     // required when continuous=false; null = random (§5)
      "gain_db": null,             // optional per-clip override; null = layer default (§4)
      "change": null               // RESERVED PLACEHOLDER — weather transitions (§7)
    }
  ],

  "events": [                  // 0..N, optional
    {
      "wav": "<bytes>",
      "species": "tawny_frogmouth",  // label for the explanation only
      "onsets_s": [5.0, 12.5, 19.0], // explicit call times; null = one random onset (§5)
      "gain_db": null                // optional per-clip override; null = layer default
    }
  ]
}
```

### Field semantics

- **`ambient`** — exactly one bed. Looped (or trimmed) to `duration_s` exactly
  as the MVP does today. No placement; it underlies the whole timeline.
- **`weather[].continuous`** — splits the two weather behaviours:
  - `true` → a **bed** (wind, steady rain). Looped/crossfaded to full
    duration like ambient. `onsets_s` is ignored.
  - `false` → a **discrete** sound (thunder clap). Placed at each time in
    `onsets_s`; the clip is *not* looped to fill duration.
- **`weather[].onsets_s` / `events[].onsets_s`** — list of start times in
  seconds on the final timeline. One element = one occurrence; multiple
  elements = repetition (this is how "frequency" is realised). `null` triggers
  the random fallback in §5. Each placed copy that runs past `duration_s` is
  trimmed at the end (matches MVP event behaviour).
- **`events`** — always discrete; there is no `continuous` flag. Every species
  call is placed at its onsets.
- **`gain_db`** — see §4.

---

## 4. Gain staging

**Provisional for this stage — expected to be re-tuned by ear once multi-clip
artifacts exist.** Start from the MVP's per-layer gains (these came out of
songke's v5 listening pass on real A/B/C stems, so they're a validated baseline,
not a guess):

| Layer / role | Starting gain (provisional) |
|---|---|
| Ambient | 0 dB |
| Weather | −12 dB |
| Event | −18 dB |

These are a starting point, not a locked decision. After the first multi-clip
mixes are generated and listened to, adjust the per-layer defaults (and record
the new values in this attempt's `params.yaml`). The per-clip `gain_db` override
is the fine-tuning lever for individual clips that sit too hot or too quiet
within a layer.

Every clip in a layer gets that layer's default gain unless its `gain_db` field
is set, in which case the per-clip value overrides the layer default for that
clip only. The same final **peak ceiling (0.95)** and runtime format
(22,050 Hz mono float32 → PCM16 export) from the MVP apply after summing.

Event band-pass and the event activity envelope from the MVP also carry forward
and apply per discrete clip.

---

## 5. Random onset fallback (and the seed)

When an `onsets_s` list is `null`, Layer D assigns the onset(s) itself:

- **Events**: place one copy at a random onset within
  `[0, duration_s − clip_length]`.
- **Discrete weather**: same single-random-onset behaviour.
- Random draws use **`placement_seed`** so the same request reproduces the same
  arrangement. Multiple omitted clips each get an independent draw from the same
  seeded RNG; resulting overlaps are fine (they're summed, §2) — Layer D does
  not try to space clips out.

> **Convention change to note in handoff:** the MVP states "Layer D does not
> interpret environmental conditions or generation seeds" and the handler does
> `del seed`. This stays true for *generation* seeds (A/B/C own those). Layer D
> now takes a **separate `placement_seed`** used solely for random-onset
> fallback — it never touches audio synthesis. CLAUDE.md's Layer D
> dev-generation note should be refined accordingly when this lands.

---

## 6. Limits (this stage)

- `duration_s` ≤ **30 s** (unchanged — Layer B currently caps at 30 s).
- No maximum clip count is enforced by the mixer; the LLM is trusted to emit a
  sane arrangement within 30 s.
- Overlapping onsets are summed, not resolved or rejected (see §2).

---

## 7. Reserved: weather change / transitions

`weather[].change` is a **reserved placeholder**, intentionally `null` for now.
Not every soundscape needs a weather transition, and transitions are out of
scope for this stage. The field exists so the schema is forward-compatible: a
future iteration can express "heavy rain → light rain" as a transition spec
(e.g. crossfaded segments with timed intensity changes) **without changing the
top-level contract**. Implementers should carry the field through untouched and
not build transition behaviour yet.

---

## 8. Output contract

Unchanged from the MVP registry payload:

- `wav_bytes` — final 22,050 Hz mono PCM16 WAV.
- `mel_db` — mel spectrogram preview of the mix.
- `metadata.layer_d` — the explanation dict, **extended** with one row per
  placed clip:
  - resolved `onsets_s` (after random fallback / trimming),
  - `placement_random: true|false` and `placement_seed` when random was used,
  - per-clip applied `gain_db`,
  - `continuous` flag (weather),
  - the carried-forward fields (band-pass, activity envelope, peak protection).

The explanation must let a reviewer reconstruct exactly where every sound was
placed and why.

---

## 9. Handoff checklist for the implementer

1. Create `code/` (mirror the MVP layout: `handler.py`, `audio_mixer.py`,
   `audio_format.py`, `audio_metrics.py`) + `params.yaml` (frozen snapshot of
   the gains / format constants above).
2. Generalise `MixRequest` to lists of placed weather/event clips per §3.
3. Add `placement_seed` plumbing and the §5 random fallback.
4. Extend the explanation JSON per §8.
5. Register the attempt in `acoustic_ai/registry.yaml` (`kind: generative`? no
   — Layer D is an algorithmic combiner; follow how `songke__mvp_1` is
   declared) once code exists.
6. Wire the orchestrator (`registry.orchestrate_generation`) to call B/C for
   multiple clips and pass the lists — **separate follow-up task, out of scope
   for this design card.**
7. Update the LLM parser contract per
   [`prompt_parser_policy.md`](../../../../../.claude/context/ai/prompt_parser_policy.md)
   to emit `onsets_s` / `continuous` per clip — also a follow-up.
