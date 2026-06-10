# Layer D Layered Mix

## Status

MVP implementation complete. The attempt can normalize, prepare, mix, and
export upstream Layer A/B/C WAV stems through its registry handler. System-level
generation orchestration is implemented through `POST /generation/render` on
the AI server and `POST /api/generation` on the Express backend.

## Audio Format Decision

Layer D uses the following fixed formats:

- Runtime mix: 22,050 Hz mono float32
- Final WAV export: 22,050 Hz mono PCM16

The decision was made after comparing three real 180-second Layer A, B, and C
stems in both 22.05 kHz mono PCM16 and 44.1 kHz stereo PCM24. The normalized
mono versions preserved duration and loudness metrics, produced no clipping,
and had no audible quality problems. In particular, the stereo weather stem
did not exhibit audible phase cancellation or missing content after downmixing.

Comparison outputs and the listening review are local development artifacts
under `dev-artifacts-self-testing/format_comparison/`.

## MVP Defaults

- Ambient gain: 0 dB
- Weather gain: -12 dB
- Event gain: -18 dB
- Event activity boundary: 1.0-second smoothstep fade
- Event band-pass: 500-8,000 Hz, fourth-order zero-phase Butterworth
- Final peak ceiling: 0.95
- Event placement: `random`, scattered across the timeline with a 2-second edge buffer

These defaults correspond to local listening attempt
`v5_smooth_1s_gain_minus18db_bandpass_500_8000`.
The complete attempt-local parameter snapshot is stored in `params.yaml`;
runtime serving defaults are mirrored in `acoustic_ai/registry.yaml`.

## Handler Contract

`handler.generate()` accepts:

- Required: `ambient_wav_bytes`
- Optional: `weather_wav_bytes`, `event_wav_bytes`, `event_start_s`
- Optional: `seed` (used only to place the event onset; see below)
- Optional: `duration_s` (defaults to 30 seconds)

It returns the standard registry payload: `wav_bytes`, `mel_db`, and `metadata`.
The generation orchestrator calls the selected/default A/B/C attempts in
memory, then passes their WAV bytes directly to this handler.

Parameter ownership is explicit:

- Layer A receives `seed`, `season`, and `diel`.
- Layer B receives `seed`, `weather_type`, `intensity`, and `duration_s`.
- Layer C receives `seed`, `season`, `diel`, and `duration_s`.
- Layer D receives upstream WAV bytes, `duration_s`, and the shared `seed`.

Layer D does not interpret environmental conditions. It uses the shared `seed`
only as a mixing concern — to draw a reproducible event onset (see below); it
does not generate audio from it.

## Event Placement

The event stem (Layer C) is placed at a seeded-random onset rather than always
at `t=0`. Controlled by two `params` (`event_placement`, `event_edge_buffer_s`):

- `event_placement: "random"` (default) scatters the onset across the whole
  timeline using `np.random.default_rng(seed)`: the event may start anywhere
  from `event_edge_buffer_s` up to `duration_s - event_length - event_edge_buffer_s`.
  The window therefore **scales with `duration_s`** — there is no fixed second
  cap, so longer renders scatter events across their full length. The trailing
  buffer guarantees the whole event plays before the end (no end-trim). Same
  `seed` + same inputs → identical onset.
- `event_edge_buffer_s` (default `2.0`) is the lead-in kept at the start and the
  room kept at the end, so events neither open nor close the soundscape.
- `event_placement: "fixed"` honors the explicit `event_start_s` kwarg
  (default `0.0`), preserving the original start-of-timeline behavior.

The orchestrator threads the run's shared `seed` into Layer D, so placement is
reproducible and tied to the same seed as A/B/C.

## Current Orchestration Limits

- Product generation duration is limited to 30 seconds because the current
  Layer B handler accepts at most 30 seconds.
- Layer A may return a shorter ambient bed; Layer D loops it to the requested
  duration.
- Layer C may return a shorter event clip; Layer D places it at a seeded-random
  onset (see "Event Placement") and leaves the remainder event-free.
- Natural-language request to structured generation parameters is not yet
  implemented.
