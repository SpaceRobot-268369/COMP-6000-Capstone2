# Layer B Generate Rain — Consolidated Plan

This file consolidates the prior rain-generation planning notes so the attempt directory keeps a single plan document.

## Source: IMPLEMENTATION_PLAN.md

# Layer B Generate Rain — Smoke-1 Implementation Plan

Date: 2026-06-06
Owner: murphy
Status: planned (step-by-step execution)

## 0. Goal and boundaries

Build a true Layer B rain generator for smoke-1 using AudioLDM2 + LoRA, with
the training pool:

`debug/murphy_layer_b_rain_smoke_training_pool_v0_20260606`

This smoke round validates one thing only: can the model learn rain texture
from the current small real-site pool. It is not expected to deliver final
intensity control or broad generalization.

## 1. Data constraints (must be respected)

- Pool size: 72 clips (5 s WAV, 22.05 kHz mono)
- Labels: rain=68, rain+wind=4
- Recordings: 8
- Source: site_257 real recordings only, auto-screened by E-B + human pass
- Risks:
  - overfit from small and correlated clips
  - limited style diversity
  - background noise contamination
  - rain+wind leakage into pure rain target
  - weak intensity controllability

## 2. Core technical strategy

- Model family: AudioLDM2 (`cvssp/audioldm2`) + LoRA (same stable stack as
  Layer A and Layer B generate-wind references)
- Scope: single rain generator attempt first (seed-only runtime control)
- No intensity bank in smoke-1
- Runtime contract:
  - user controls: `seed`
  - server locked: prompt, steps, guidance, duration, postprocess params

## 3. Attempt and artifact paths

- Attempt code:
  - `acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__audioldm2_rain/`
- Candidate checkpoint:
  - `model/candidates/murphy/smoke_1__audioldm2_rain/`
- Data manifests in attempt:
  - `data/rain_manifest.csv`
  - `data/rain_manifest_val.csv`

## 4. Step plan (execution order)

### Step 0 — record plan (this file)

- Create attempt folder and persist this plan.

### Step 1 — scaffold attempt files from wind smoke-2 baseline

- Copy minimal code skeleton from:
  - `murphy__smoke_2__audioldm2_wind`
- Keep files:
  - `code/train_audioldm2.py`
  - `code/audioldm2_dataset.py`
  - `code/sample_audioldm2.py`
  - `code/handler.py`
  - `code/layer_a_visualization.py`
- Add new helper:
  - `code/build_rain_manifest.py`
- Add:
  - `params.yaml`
  - `README.md`
  - `showcase/.gitkeep`
  - `expected/.gitkeep`
  - `dev-artifacts-self-testing/.gitkeep`

### Step 2 — prepare rain manifests from debug pool

- Input files:
  - `debug/.../caption_manifest.csv`
  - `debug/.../training_pool_manifest.csv`
- Output:
  - `data/rain_manifest.csv`
  - `data/rain_manifest_val.csv`
- Rules:
  - split by `recording_id` group (not random)
  - keep caption distinction:
    - rain -> pure-rain caption
    - rain+wind -> rain-with-light-wind caption
  - keep `status=ok`
  - include source recording id columns for audit

### Step 3 — tune params for small-data smoke run

- Training:
  - epochs: 3-4 (early-stop ready)
  - lr: 1e-5
  - batch: 4 (or 2 if needed)
  - lora r/alpha/dropout: 8/32/0.1
  - duration: 5.0 s
  - sample rate: 16 kHz (loader resamples)
- Inference locked defaults:
  - prompt: pure rain
  - negative_prompt includes birds/insects/wind gusts/reverb suppression
  - steps: 200
  - guidance: 3.0 baseline
  - duration: 5.0 s
  - highpass: 80 Hz
  - denoise + fade enabled

### Step 4 — register attempt in `acoustic_ai/registry.yaml`

- Add `layer_b` attempt:
  - id: `murphy__smoke_1__audioldm2_rain`
  - uses_seed: true
  - checkpoint and params aligned with attempt files

### Step 5 — expose attempt in backend generate merge list

- Include new attempt id in Layer B generate attempts merge allowlist.
- Keep existing behavior for other attempts unchanged.

### Step 6 — frontend generate page visibility

- Ensure attempt appears in dev generation UI.
- Seed-only controls for this attempt.

### Step 7 — smoke verification (no hidden auto-fixes)

- Sanity checks:
  - manifest loads
  - train script arg parse
  - sample/handler dry-run path checks
  - lints for changed files
- Report blockers immediately and pause for user decision.

## 5. Pause rule

During execution, after each step is completed:

1. report what changed and where,
2. report checks run and results,
3. wait for user confirmation before next step.

If any issue appears, report and pause; do not improvise destructive or
unapproved workaround actions.

## Source: NEXT_STEP_PLAN_20260606.md

# Layer B Rain Smoke-1 — Next Step Plan (2026-06-06)

## Background

Based on the local diagnostic folder:

- `spectrum_diagnostic_20260606/REPORT.md`
- `spectrum_diagnostic_20260606/band_energy_metrics.csv`
- `spectrum_diagnostic_20260606/per_file_2_8khz_metrics.csv`

the main observations are:

- No clear deficit in 2-8 kHz for generated samples vs real rain (mean PSD).
- Main gap is in 8-11 kHz, where generated audio is far lower than real rain.
- This aligns with the 16 kHz generation ceiling (Nyquist 8 kHz) of current AudioLDM2 path.

## Key Judgement

The current "muffled" perception is mainly a missing >8 kHz brightness issue,
not a 2-8 kHz weakness. Therefore:

- Extra denoise is not the main fix path.
- Post-generation bandwidth extension is the most direct low-cost intervention.
- Overfitting concern should be handled in a separate retraining track (early stop + augmentation),
  not mixed into the immediate timbre-brightness fix.

## Recommended Next Steps (No execution in this file)

### Completed A2 update: curated intensity contract

The rain smoke attempt now has a curated two-bin seed contract for frontend and
server follow-up work:

- `seed_mode=curated`
- `uses_intensity=true`
- `intensities=[light, heavy]`
- `default_intensity=light`
- `good_seeds_by_intensity` and flat `good_seeds` are mirrored in
  `params.yaml` and `acoustic_ai/registry.yaml`

This is a smoke-stage curation layer, not model conditioning. Because output is
RMS-normalised, the `light`/`heavy` labels describe texture, density, and
spectral balance rather than playback volume. The split is based on objective
features (`low_mid_ratio_db`, `crest_factor`, `spectral_centroid_hz`) plus human
listening; `good_seeds_audit.csv` is the evidence source.

The whitelist and bins are valid only for the current checkpoint and locked BWE
parameters. If model weights, prompt/postprocess behavior, or BWE settings
change, the whitelist and intensity labels must be re-audited.

### Completed A3-A6 update: curated seed runtime and showcase closure

The runtime contract now treats curated seed selection as a server-side decision:

- If a request includes a seed in `good_seeds`, the AI service uses it for
  reproducibility.
- If the seed is missing, invalid, or outside the whitelist, the AI service
  randomly chooses a reviewed seed from `good_seeds`.
- The resolved seed is returned in response metadata so users can replay a good
  result.
- Express/frontend do not make the final seed decision; they may omit seed for
  random rain generation.

Seed robustness is deliberately scoped. This smoke attempt only exposes audited
seeds because arbitrary seeds were not consistently acceptable. The evidence
source remains `good_seeds_audit.csv`, and the whitelist is bound to the current
checkpoint plus locked BWE parameters.

The formal showcase has been refreshed with 10 representative good seeds:

- light: `42`, `43`, `44`, `51`, `999983`
- heavy: `46`, `48`, `2000000001`, `123456789`, `69317`

A6 retest output:

- `spectrum_diagnostic_20260606/bwe_prototype/a6_good_seed_showcase_retest_20260607/REPORT.md`
- Mean 8-11 kHz gap vs real reference: -2.48 dB
- Mean 8-11 minus 2-8 kHz drop: -13.70 dB
- Mean 2-8 kHz gap vs real reference: -0.75 dB
- All objective gates passed.

### Step 1 (Priority): Offline BWE prototype on existing showcase outputs

Goal:

- Synthesize plausible >8 kHz rain texture and export to 22.05/24 kHz outputs.

Method:

- Use a noise-shaped high-band reconstruction strategy (SBR-like / exciter-style).
- Match high-band spectral slope to real-rain reference statistics from diagnostic CSV.
- Run only on existing showcase seeds (42-51) for A/B listening and spectrum checks.

Deliverables:

- New diagnostic comparison plots (before vs after BWE vs real reference).
- Per-file and aggregated band metrics update.
- Listening notes for subjective pass/fail.

### Step 2: Reduce or disable current spectral denoise in the test branch

Reason:

- Rain itself is noise-like; aggressive spectral subtraction can remove natural rain texture.

Action direction:

- Start from denoise disabled (or much weaker), then re-check hiss vs openness trade-off.

### Step 3: Retraining track for overfitting risk (separate track)

Goal:

- Improve generalization under the 8-recording pool constraint.

Direction:

- Select checkpoint by validation (not simply last epoch).
- Add small-data augmentation (gain/time shift/light stretch/mix-style variants).
- Increase regularization conservatively (e.g., LoRA dropout up one notch).

## Decision Rule

Proceed in order:

1. Validate whether BWE solves perceived muffled quality.
2. Only if needed, tune denoise and prompt/inference details.
3. Keep retraining changes isolated from postprocess experiments for clean attribution.

## Source: A2_INTENSITY_CONFIG_V2_PLAN.md

# A2 Intensity Config V2 Plan

Status: planned only. This document records the next-step contract for locking
the curated rain good-seed whitelist and two-bin intensity structure into the
attempt configuration. No code or config changes are implied by this note.

## Goal

Move the rain attempt from a flat curated good-seed list toward a config-backed
two-intensity contract:

- `light`: lighter, more textured, higher-centroid or more eventful rain.
- `heavy`: denser, steadier, lower-mid weighted rain.

This is a curated binning layer, not model conditioning. Because the current
pipeline RMS-normalises output, intensity should be described as texture,
density, and spectral balance rather than loudness.

## A2.1 Label The 28 Good Seeds

Input: the accepted 28-row `good_seeds_audit.csv`.

For every accepted seed, compute binning features:

- `low_mid_ratio_db`: energy in `0-2 kHz` minus energy in `2-8 kHz`.
  Higher values suggest a heavier low-frequency balance and should bias toward
  `heavy`.
- `crest_factor`: peak/RMS style transient measure. Higher peakiness can bias
  toward `light`.
- Short-time envelope variance: more fluctuation or drop texture can bias
  toward `light`; steadier continuous energy can bias toward `heavy`.
- `spectral_centroid_hz`: higher centroid suggests brighter/sandier texture and
  should bias toward `light`.

Use `low_mid_ratio_db` as the primary initial axis, with `crest_factor`,
envelope variance, and `spectral_centroid_hz` as supporting evidence.

Then perform a 28-item listening pass. Human listening is the final decision for
borderline items.

Balance target: each intensity bin should contain at least 8 accepted seeds. If
one bin is short, generate a targeted supplement batch, do objective prefiltering
for the missing bin, then only listen to prefiltered candidates.

## A2.2 Extend `good_seeds_audit.csv`

Add these columns:

- `intensity`: `light` or `heavy`.
- `low_mid_ratio_db`.
- `crest_factor`.
- `spectral_centroid_hz`.
- `feature_source`.

All 28 rows must have:

- `review_status=accepted`.
- a non-empty `intensity`.
- computed feature values.
- a `feature_source` that records the script, diagnostic run, or manual source
  used to derive the feature values.

## A2.3 Lock The Contract In Attempt `params.yaml`

In the rain attempt `params.yaml`, add the curated seed contract under
`inference`:

```yaml
inference:
  seed_mode: curated
  uses_intensity: true
  intensities: [light, heavy]
  default_intensity: light
  good_seeds_by_intensity:
    light: []
    heavy: []
  good_seeds: []
  bwe:
    output_sr: 24000
```

Rules:

- `good_seeds_by_intensity.light` and `.heavy` are the source of truth for
  per-intensity random choice.
- `good_seeds` is the deduplicated union, kept for fallback and audit.
- `bwe.output_sr: 24000` must remain explicit, not only implied by code
  defaults.
- Do not add per-intensity RMS or EQ parameters in this version. The current v2
  plan is pure curation only.

## A2.4 Mirror In `acoustic_ai/registry.yaml`

For the same attempt entry in `acoustic_ai/registry.yaml`, mirror these fields
exactly:

- `seed_mode`.
- `uses_intensity`.
- `intensities`.
- `default_intensity`.
- `good_seeds_by_intensity`.
- `good_seeds`.

The registry mirror must match the attempt `params.yaml` values byte-for-byte at
the semantic YAML level. Configuration should be auditable without relying on
code defaults.

## A2.5 Consistency Checks

Add or run a read-only validation step that checks:

- `light union heavy == good_seeds`.
- `light` and `heavy` have no overlap.
- no duplicate seed exists in either bin or in the flat `good_seeds`.
- every listed seed exists in `good_seeds_audit.csv`.
- every listed seed has `review_status=accepted`.
- every listed seed has an `intensity` matching its bin.
- each bin has at least 8 seeds.
- `default_intensity` is one of `intensities`.
- attempt `params.yaml` and `acoustic_ai/registry.yaml` match for the mirrored
  fields.
- required explicit keys exist and are not `None`: `seed_mode`,
  `uses_intensity`, `intensities`, `good_seeds_by_intensity`, and
  `bwe.output_sr`.

## A2.6 Frontend Contract

Confirm that `GET /layers` for this attempt exposes enough metadata for the
frontend to render an intensity selector:

- `seed_mode=curated`.
- `uses_intensity=true`.
- `intensities=[light, heavy]`.
- `default_intensity=light`.

The frontend does not need to receive the full seed buckets. A3 should keep
bucket selection server-side.

## A2.7 Documentation

Update the attempt README, model card notes, or next-step plan with:

- Intensity is a curated two-bin split, not model conditioning.
- Because RMS is normalised, intensity means texture, density, and spectral
  balance, not louder volume.
- The binning evidence lives in `good_seeds_audit.csv`.
- The whitelist and bins are bound to the current checkpoint and BWE parameters.
  Changing model weights or BWE settings requires re-audit.

## A2.8 Exit Criteria

A2 is complete when:

- all 28 accepted seeds have `light` or `heavy` labels;
- each bin has at least 8 seeds;
- no seed is in both bins;
- `params.yaml` and `acoustic_ai/registry.yaml` carry matching curated seed
  contract fields;
- validation passes;
- `GET /layers` exposes the intensity contract needed by the frontend;
- docs honestly state the smoke-stage limitation and audit basis.

After A2 passes, A3 can implement server-side bucket selection and random seed
choice by requested intensity.
