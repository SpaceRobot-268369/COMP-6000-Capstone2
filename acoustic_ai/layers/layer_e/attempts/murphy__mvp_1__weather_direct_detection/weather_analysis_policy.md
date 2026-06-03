# E-B Weather Analysis Policy

Layer E-B uses direct detection on an uploaded mixture. It must not attempt to
separate sources before detection.

## Detection Priority

1. Pre-trained model evidence is primary:
   - CLAP audio-text similarity for open weather prompts.
   - PANNs CNN14 AudioSet weather tags as broad independent evidence.
   - AST AudioSet tags as a conservative guard for ambiguous weather mixtures.
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

PANNs and AST AudioSet tags should map to the same three elements where
available: rain, wind, thunder, thunderstorm. BEATs is not part of the MVP main
path; if revisited later, it should be treated as an optional guard rather than
a raw averaged score.

## Fusion Rules

The MVP uses transparent gate rules before trying a trained head. Model scores
are evidence channels, not calibrated probabilities to average blindly:

- CLAP proposes candidate weather elements.
- PANNs provides broad AudioSet weather support.
- AST guards against ambiguous or over-eager CLAP/PANNs decisions.
- Acoustic features explain risk such as clipping, overload, or weak coverage.

Do not force a single winning weather class. Decide each element independently
and derive the composite label from the present elements.

## Presence Thresholds

Gate v1.1 MVP thresholds:

| Element | Present if | Notes |
|---|---|---|
| rain | strict CLAP rain path, or weak rain-under-wind path | Weak path requires close rain evidence under wind plus PANNs or AST rain support. |
| wind | CLAP wind is strong enough and not dominated by `none`/contamination | Lower confidence if clipping/overload is high. |
| thunder | CLAP thunder plus supporting evidence and no obvious wind-overload conflict | Never declare thunder from PANNs/AST alone. |

Current weak rain-under-wind gate:

```text
clap_wind >= 0.45
clap_rain >= 0.49
clap_rain >= clap_wind - 0.04
clap_rain >= clap_thunder - 0.15
top_clap_weather in {rain, wind}
PANNs or AST has rain support
```

When this weak path fires, mark rain present with low confidence and add
`possible_rain_under_wind`.

If no element passes its threshold, return `overall_label: none`.

Composite labels are derived from independent element decisions. Do not force a
single winning weather class. For example:

- rain present + wind present -> `rain+wind`
- rain present + thunder present -> `rain+thunder`
- rain present + wind present + thunder present -> `rain+thunder+wind`

Mixed labels should include confidence and coverage for each element, because
E-B is detecting elements in a mixture, not separating stems.

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
- `possible_rain_under_wind`: wind is present and rain evidence is close to the
  rain threshold but not strong enough to mark rain present.
- `rain_confirmed_without_beats_guard`: rain was confirmed by the active MVP
  evidence channels while BEATs was not used as a guard.
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

Current frozen gate reference:

- Server B 12-sample calibration after gate v1.1: `10/12` exact.
- Holdout sanity check: `6/8` exact.
- The two holdout rain+wind misses were human-reviewed as wind with extremely
  subtle rain, so the conservative `wind` output is acceptable for MVP.
