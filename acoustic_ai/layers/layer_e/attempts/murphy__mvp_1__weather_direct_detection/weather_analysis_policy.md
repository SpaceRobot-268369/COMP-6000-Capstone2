# E-B Weather Analysis Policy

Layer E-B uses direct detection on an uploaded mixture. It must not attempt to
separate sources before detection.

## Detection Priority

1. Pre-trained model evidence is primary:
   - CLAP audio-text similarity for open weather prompts.
   - PANNs or YAMNet AudioSet weather tags as an independent cross-check.
2. Acoustic features are secondary:
   - used to explain, calibrate, or lower confidence
   - never enough on their own to declare a weather element present
3. Environmental metadata and Layer B pool labels are not required inputs.

## Prompt / Label Sets

Initial CLAP prompt groups:

| Element | Positive prompts |
|---|---|
| rain | `rain`, `light rain`, `steady rain`, `heavy rain`, `rain on leaves`, `rainfall ambience` |
| wind | `wind`, `light wind`, `strong wind`, `wind in trees`, `gusty wind` |
| thunder | `thunder`, `distant thunder`, `thunder rumble`, `thunderstorm` |
| none | `quiet dry woodland ambience`, `quiet outdoor ambience`, `no weather sound` |
| contamination | `birdsong`, `insects`, `cicadas`, `human voice`, `machinery`, `microphone handling noise` |

PANNs/YAMNet AudioSet tags should map to the same three elements where
available: rain, wind, thunder, thunderstorm.

## Fusion Rules

The first MVP should use transparent rules before trying a trained head.

Suggested element confidence:

```text
confidence(element) =
  0.65 * clap_score(element)
  + 0.25 * audioset_score(element)
  + 0.10 * feature_support(element)
  - contamination_penalty
```

If PANNs/YAMNet is not available in the first smoke implementation, redistribute
the AudioSet weight to CLAP and mark `debug.audioset_available = false`.

## Presence Thresholds

Initial MVP thresholds:

| Element | Present if | Notes |
|---|---|---|
| rain | confidence >= 0.55 | Require margin over `none` and contamination. |
| wind | confidence >= 0.55 | Lower confidence if clipping/overload is high. |
| thunder | confidence >= 0.60 | Require transient or low-frequency support. |

If no element passes its threshold, return `overall_label: none`.

## Intensity Rules

Intensity is calibrated from confidence, RMS, duration coverage, and feature
support.

| Intensity | Suggested rule |
|---|---|
| none | element absent |
| light | present but weak confidence, low RMS, or sparse coverage |
| medium | stable confidence and moderate RMS / coverage |
| heavy | high confidence plus high RMS, strong coverage, or clear thunder burst |

Thunder is burst-like, so coverage should not be required in the same way as
rain or wind.

## Warnings

Add warnings when ambiguity is likely:

- `possible_bio_overlap`: bio/insect prompt or AudioSet score is close to a
  weather score.
- `possible_human_or_machine_overlap`: human/machine prompt is close to a
  weather score.
- `possible_wind_overload`: thunder score is high but clipping, peak, or wind
  score is also high.
- `possible_clipping`: clipping ratio exceeds the MVP threshold.
- `weather_mixed_with_ambient`: weather is present but not dominant.
- `low_confidence`: one or more present elements are near threshold.

## Audit Policy

Use small targeted audits to tune thresholds:

- balanced weather clips: rain, wind, thunder/storm, and mixed weather
- negative controls: quiet ambience, bird/insect dominant, human/machine noise
- site-derived clips and non-site reference clips may both be used for analysis
  evaluation, because E-B is not a site retrieval feature

Audit output should compare human labels against model output, not promote
assets into a runtime pool.

