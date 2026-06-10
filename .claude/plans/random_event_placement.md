# Implementation Plan — Randomized Event Placement (Layer D)

**Goal:** Events (Layer C) are placed at a *seeded-random* onset on the final
timeline instead of always at `t=0`. Default behavior: scatter the event at a
random start in a configurable window (e.g. `2s`–`10s`). Same `seed` + same
inputs → same placement (reproducibility contract from CLAUDE.md).

**Design principle:** the *time* is decided in Layer D (a mixing concern), not
by the LLM. The LLM/parser only emits semantic intent (`density` / a
`placement` mode); Layer D draws the actual onset with a seeded RNG.

---

## Current state (why it lands at t=0)

- Mixer already supports timed placement: `EventPlacement.start_s`
  (`audio_mixer.py:48`) → `prepare_event_timeline` places at
  `round(start_s * sample_rate)` (`audio_mixer.py:232`). Out-of-range starts
  are clamped (event trimmed / dropped if past the end).
- Handler defaults `event_start_s: float = 0.0`
  (`handler.py:38`) and wraps the single Layer C WAV in one placement.
- Orchestrator never passes a start and hands Layer D `seed=None`
  (`registry.py:590-597`), so there is no RNG and no onset → every event at 0.
- Parser contract stops at `layer_c: { species, density }`
  (`prompt_parser_policy.md §5`) — no timing field.

---

## Changes

### 1. Layer D handler — draw a seeded onset
File: `acoustic_ai/layers/layer_d/attempts/songke__mvp_1__layered_mix/code/handler.py`

- Stop discarding `seed` (`del seed` at line 44). Use it.
- Add params (read from `state.params`, with defaults):
  - `event_placement: "random" | "fixed"` (default `"random"`)
  - `event_start_window_s: [min, max]` (default `[2.0, 10.0]`)
- When `event_placement == "random"` and an event exists:
  - `rng = np.random.default_rng(seed)` (seed `None` → nondeterministic, which
    is fine; orchestrator will pass a concrete seed).
  - Clamp the window to the event's room on the timeline:
    `hi = min(window_hi, max(0.0, duration_s - event_len_s))`,
    `lo = min(window_lo, hi)`; `start_s = rng.uniform(lo, hi)`.
  - This guarantees the event isn't trimmed off the end for short durations.
- When `"fixed"`, keep honoring the explicit `event_start_s` kwarg (back-compat;
  default `0.0`).
- Echo the resolved `start_s`, window, and mode into the returned metadata
  (mixer already records `requested_start_s` per placement).

### 2. Orchestrator — thread the seed into Layer D
File: `acoustic_ai/server/registry.py` (`orchestrate_generation`, ~line 590)

- Change the Layer D call from `seed=None` to `seed=seed` so placement is
  reproducible and tied to the same seed as A/B/C.
- Add `"layer_d"` routing entry to include `"seed"` and (optionally)
  `event_placement` / `event_start_window_s` in the `parameter_routing`
  metadata block (`registry.py:609-613`) so the response is self-describing.

### 3. (Optional, recommended) Parser contract — semantic intent only
File: `.claude/context/ai/prompt_parser_policy.md` (schema §5) + parser impl

- Extend `layer_c` with an optional `placement` hint, e.g.
  `"placement": "random" | "start" | "scattered"` (keep `density` as-is).
- Map the hint → Layer D params in the orchestrator (e.g. `"start"` →
  `event_placement: "fixed", event_start_s: 0`). Do **not** have the LLM emit a
  numeric time — it isn't reproducible and ignores `duration_s`/fades.
- If we skip this step for v1, Layer D defaults to `"random"` and the feature
  works without any parser change.

### 4. Attempt params + docs
- Add `event_placement` and `event_start_window_s` to the Layer D attempt's
  `params.yaml` (`inference:` section) so they're tunable per attempt.
- Note the behavior in the attempt `README.md`.

---

## Out of scope (v1)

- Multiple events / multiple onsets per render (mixer's `prepare_event_timeline`
  already accepts a tuple — a future change can place N events; for now Layer C
  yields one stem).
- Per-species placement rules. `density` → event *count* is a separate follow-up.

---

## Verification

1. `acoustic_ai/.venv/bin/python` unit check on `prepare_event_timeline`: an
   event placed at `start_s=5.0`, `duration_s=30` lands at frame
   `round(5.0 * 22050)` with `trimmed_at_end == False`.
2. Orchestrated call twice with the **same** seed → identical resolved
   `start_s` in metadata (and identical WAV bytes). Different seed → different
   `start_s` within `[2,10]`.
3. Short-duration edge: `duration_s` smaller than the window upper bound →
   `start_s` clamped so the event still fits (no end-trim).
4. `include_events=false` → no placement, unchanged output.

---

## Risk / notes

- Event length vs. window: clamp `hi` against `duration_s - event_len_s` to
  avoid silently trimming the event tail (mixer trims rather than errors).
- Seed sharing: A/B/C and D all use the same `seed`. That's intentional and
  matches the determinism contract; placement varies with seed, not with a
  separate knob.
