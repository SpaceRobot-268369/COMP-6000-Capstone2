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
