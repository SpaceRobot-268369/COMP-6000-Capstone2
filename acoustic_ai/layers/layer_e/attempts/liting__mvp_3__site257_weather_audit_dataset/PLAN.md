# Implementation Plan - E-B Site257 Weather Audit Dataset

| | |
|---|---|
| **Attempt ID** | `liting__mvp_3__site257_weather_audit_dataset` |
| **Layer / role** | `layer_e` - Analysis · E-B weather detector data expansion |
| **Stage** | `mvp_3` |
| **Primary output** | E-B-owned Site257 audit candidate manifest |
| **Scope** | Site257-only candidate mining for multi-attribute weather audit |
| **Author / date** | liting · 2026-06-03 |
| **Branch** | `feat/liting/e-b-site257-audit-dataset` |

## 1. Goal

Build a larger, E-B-owned Site257 weather audit dataset.

The current 63 audited seed clips are useful, but they are not enough for final
MVP generalisation. This attempt expands from those seed clips into the broader
Site257 clip universe.

## 2. Data Principle

Use Site257-derived audio only for client-facing E-B claims.

Existing audited seed clips are used as a calibration seed and candidate mining
guide. They are not treated as the full feature scope.

## 3. Multi-Attribute Labels

Each clip can contain multiple simultaneous acoustic conditions. The audit
schema must not force a single class per clip.

Required labels:

| Field | Values |
|---|---|
| `wind_intensity` | `none`, `light`, `moderate`, `strong`, `uncertain` |
| `rain_intensity` | `none`, `light`, `moderate`, `heavy`, `uncertain` |
| `thunder_status` | `none`, `present`, `insufficient_site_data`, `uncertain` |
| `mixed_weather` | `true`, `false`, `uncertain` |
| `bird_activity` | `none`, `low`, `medium`, `high`, `uncertain` |
| `insect_activity` | `none`, `low`, `medium`, `high`, `uncertain` |
| `background_noise` | `low`, `medium`, `high`, `uncertain` |
| `audit_status` | `pending`, `audited`, `reject`, `uncertain` |

## 4. Candidate Mining Strategy

Candidate mining is not ground truth. It is a prioritisation step so manual
listening can be focused.

Signals used in the first pass:

- existing audited weather seed labels,
- Site257 environmental wind speed metadata,
- season/diel diversity,
- source recording diversity,
- random holdout sampling.

Future passes can add:

- PANNs scores over materialised audio,
- CLAP prompt scores,
- DSP wind/rain features,
- similarity to audited seed clips.

## 5. Server B Path

This attempt should be run on Server B once DVC audio access is available:

1. Pull Site257 DVC clips.
2. Run candidate mining.
3. Materialise candidate audio where needed.
4. Run PANNs/CLAP/DSP scoring over candidates.
5. Export a listening queue.
6. Manually audit sampled candidates.
7. Produce `e_b_site257_weather_training_manifest.csv`.

## 6. Short-Term Use

For the urgent MVP path, keep using the existing 63 audited seed clips to run
the E-B baseline. Use this attempt to prepare the larger dataset without
blocking the current demo.

