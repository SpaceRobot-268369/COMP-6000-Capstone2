# Layer C Boobook Smoke Sample Audit

Branch: `model/burger/layer-c-smoke-eval`
Candidate: `model/candidates/lucas/layer-c-audiogen-boobook-smoke/`
Base model: `facebook/audiogen-medium`
LoRA: Southern Boobook / `Ninox boobook`
Audit date: 2026-05-21

## Purpose

Evaluate whether the existing Layer C AudioGen LoRA can generate recognizable
Southern Boobook nocturnal call events from seeds 42-51.

Smoke-test pass criterion from `pipeline_design.md`: at least 4 of 10 seeds
produce a clip where the two-note "boo-book" structure is identifiable in the
first 3 seconds, with no obvious EnCodec warble.

## Generation Settings

Fill this from the generated sample metadata:

| Setting | Value |
|---|---|
| Prompt | `Southern Boobook owl two-note call at night, Bowra dry woodland` |
| Seeds | `42-51` |
| Duration | `5.0 s` |
| Guidance scale | `3.0` |
| Temperature | `1.0` |
| Top-k | `250` |
| Top-p | `0.0` |
| Output target RMS | `0.02` |
| Sample rate | `16000 Hz` |

## Audit Rubric

Mark each sample. This pass is a spectrogram / signal-structure audit, not a
replacement for human listening. Final promotion should still use headphones.

| Verdict | Meaning |
|---|---|
| Pass | Clear two-note boobook-like call in first 3 s; usable as a foreground event |
| Borderline | Bird-like/nocturnal event but weak structure, artifacts, or timing issue |
| Fail | No recognizable boobook structure, heavy warble, machine noise, or unusable audio |

## Seed Results

| Seed | Verdict | Notes |
|---:|---|---|
| 42 | Fail | Sustained high-band texture across most of the clip; no clear two-note event structure. |
| 43 | Borderline | Strong foreground energy in first ~2.5 s with horizontal bands, but reads as a sustained phrase rather than a clean two-note call. |
| 44 | Fail | Mostly near-silent after the first ~0.4 s; no usable foreground event. |
| 45 | Borderline | Short foreground band in the first second; possible bird-like event, but too brief / underspecified for smoke pass. |
| 46 | Borderline | Brief low-frequency event near the start plus a small later artifact; not enough two-note structure. |
| 47 | Pass | Clear separated foreground blocks within the first 3 s; strongest two-note / repeated-call candidate in this set. |
| 48 | Fail | Broadband full-duration texture; likely noisy bed rather than isolated boobook event. |
| 49 | Borderline | Short low-mid event in the first second; plausible onset but too truncated for a confident pass. |
| 50 | Pass | Multiple separated foreground blocks with consistent banding; plausible repeated-call event, though needs listening check for warble. |
| 51 | Fail | Full-duration noisy texture with no isolated event shape. |

## Summary

Status: spectrogram audit complete; human listening recommended.

Seed 42-51 sample bundles were generated successfully on CPU. Each bundle
contains `generated_event.wav`, `generated_event_spectrogram.png`, and
`generated_event_metadata.json`.

Spectrogram-based result: **2 pass / 4 borderline / 4 fail**. This does **not**
meet the smoke-test pass criterion of at least 4 clear passes out of 10. The
best candidates are seeds **47** and **50**. Seeds **43, 45, 46, and 49** are
worth a quick headphone check, but should not be counted as passes unless the
two-note boobook structure is clearly audible.

## Follow-Up

- Do a short headphone pass on seeds 43, 45, 46, 47, 49, and 50.
- If listening confirms fewer than 4 passes, keep this candidate as
  inconclusive / below smoke bar.
- Investigate whether the tiny 50-segment dataset, prompt wording, LoRA target
  modules, or 5-epoch / 1e-5 training setup is limiting event specificity before
  training the second species.
