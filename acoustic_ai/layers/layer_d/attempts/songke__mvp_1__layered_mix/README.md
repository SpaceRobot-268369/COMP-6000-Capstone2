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

These defaults correspond to local listening attempt
`v5_smooth_1s_gain_minus18db_bandpass_500_8000`.
The complete attempt-local parameter snapshot is stored in `params.yaml`;
runtime serving defaults are mirrored in `acoustic_ai/registry.yaml`.

## Handler Contract

`handler.generate()` accepts:

- Required: `ambient_wav_bytes`
- Optional: `weather_wav_bytes`, `event_wav_bytes`, `event_start_s`
- Optional: `duration_s` (defaults to 30 seconds)

It returns the standard registry payload: `wav_bytes`, `mel_db`, and `metadata`.
The generation orchestrator calls the selected/default A/B/C attempts in
memory, then passes their WAV bytes directly to this handler.

Parameter ownership is explicit:

- Layer A receives `seed`, `season`, and `diel`.
- Layer B receives `seed`, `weather_type`, `intensity`, and `duration_s`.
- Layer C receives `seed`, `season`, `diel`, and `duration_s`.
- Layer D receives only upstream WAV bytes, event placement, and `duration_s`.

Layer D does not interpret environmental conditions or generation seeds.

## Current Orchestration Limits

- Product generation duration is limited to 30 seconds because the current
  Layer B handler accepts at most 30 seconds.
- Layer A may return a shorter ambient bed; Layer D loops it to the requested
  duration.
- Layer C may return a shorter event clip; Layer D places it at the start and
  leaves the remainder event-free.
- Natural-language request to structured generation parameters is not yet
  implemented.
