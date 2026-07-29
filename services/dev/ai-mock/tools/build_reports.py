#!/usr/bin/env python
"""Bake the canned analysis reports (one per season x diel cell).

Called by build_fixtures.py. Writes fixtures/analysis/<cell>.json, each holding
the fused `analysis_aggregator.v1` report, the three per-head reports, and the
model-lineage snapshots — i.e. everything /analysis/run and the per-head
/analyze route need to return.

The numbers are fabricated demo data, but they are internally consistent (the
fused decision matches the heads it claims to fuse) and they steer clear of the
three shapes that make the UI render blanks:
  * never `undetermined` season/diel (frontend/src/lib/analysisScene.js:13)
  * `disagreements` / `limitations` never empty (two cards hide when they are)
  * `overall_confidence` >= 0.5 (a near-zero gauge looks broken)
"""

from __future__ import annotations

import json
from pathlib import Path

SEASONS = ("spring", "summer", "autumn", "winter")
DIELS = ("dawn", "morning", "afternoon", "night")

HOUR = {"dawn": 5.6, "morning": 8.4, "afternoon": 14.7, "night": 22.3}
MONTH = {"spring": 10.2, "summer": 1.4, "autumn": 4.1, "winter": 7.3}

# Species voiced per diel bin, with the phenology the events head "reports".
FAUNA = {
    "dawn": [
        ("horsfields_bronze_cuckoo", "Horsfield's Bronze-cuckoo", "Chalcites basalis", "dawn", 0.78, "spring", 0.44),
        ("gray_shrikethrush", "Gray Shrikethrush", "Colluricincla harmonica", "dawn", 0.62, "weak", 0.21),
    ],
    "morning": [
        ("yellow_throated_miner", "Yellow-throated Miner", "Manorina flavigula", "morning", 0.71, "weak", 0.18),
        ("australian_raven", "Australian Raven", "Corvus coronoides", "morning", 0.58, "weak", 0.16),
    ],
    "afternoon": [
        ("willie_wagtail", "Willie Wagtail", "Rhipidura leucophrys", "afternoon", 0.55, "weak", 0.15),
        ("zebra_finch", "Zebra Finch", "Taeniopygia guttata", "afternoon", 0.61, "weak", 0.19),
    ],
    "night": [
        ("spotted_nightjar", "Spotted Nightjar", "Eurostopodus argus", "night", 0.88, "summer", 0.41),
        ("barn_owl", "Barn Owl", "Tyto alba", "night", 0.83, "weak", 0.17),
    ],
}

# Cells that get audible weather in the canned report; everything else is a
# dry, still night/day with only a background breeze.
WEATHER_BY_CELL = {
    "winter_dawn": ("rain", "light", 0.31, 0.14),
    "autumn_morning": ("rain+wind", "moderate", 0.58, 0.44),
    "spring_afternoon": ("wind", "moderate", 0.0, 0.52),
    "summer_afternoon": ("wind", "light", 0.0, 0.27),
    "autumn_night": ("rain", "light", 0.22, 0.11),
}

HABITAT = "woodland/open woodland"

LINEAGE = {
    "ambient": {
        "id": "lucas__mvp_2__clap_knn_probe_enlarged",
        "label": "CLAP k-NN ambient probe (enlarged)",
        "layer": "layer_e",
        "head": "ambient",
        "stage": "mvp_2",
        "author": "lucas",
        "status": "candidate",
    },
    "weather": {
        "id": "murphy__mvp_1__weather_direct_detection",
        "label": "Direct weather detection",
        "layer": "layer_e",
        "head": "weather",
        "stage": "mvp_1",
        "author": "murphy",
        "status": "mvp",
    },
    "events": {
        "id": "songke__prod_1__e_c_species_event_detector",
        "label": "Species event detector",
        "layer": "layer_e",
        "head": "events",
        "stage": "prod_1",
        "author": "songke",
        "status": "production",
    },
    "aggregator": {
        "id": "songke__smoke_3__analysis_aggregator",
        "label": "Analysis aggregator v1",
        "layer": "layer_e",
        "head": "aggregator",
        "stage": "smoke_3",
        "author": "songke",
        "status": "smoke_test",
    },
}


def _dist(keys, winner, posterior):
    """Spread the remaining probability over the losers, largest first."""
    rest = [k for k in keys if k != winner]
    left = round(1.0 - posterior, 4)
    weights = [0.5, 0.3, 0.2][: len(rest)]
    total = sum(weights)
    out = {winner: posterior}
    for k, w in zip(rest, weights):
        out[k] = round(left * w / total, 4)
    return {k: out[k] for k in keys}


def _weather_for(cell):
    wtype, label, rain_i, wind_i = WEATHER_BY_CELL.get(cell, ("none", "none", 0.0, 0.09))
    rain_label = "none" if rain_i <= 0.05 else ("light" if rain_i < 0.4 else "moderate")
    wind_label = "none" if wind_i <= 0.05 else ("light" if wind_i < 0.35 else "moderate")
    derived = wtype if wtype != "none" else ("wind" if wind_i > 0.05 else "none")
    return {
        "type": derived,
        "rain": {"label": rain_label, "intensity": rain_i, "coverage": round(min(0.95, rain_i * 1.6), 2), "confidence": 0.61 if rain_i else 0.88},
        "wind": {"label": wind_label, "intensity": wind_i, "coverage": round(min(0.98, 0.25 + wind_i), 2), "confidence": 0.79},
        "thunder": {"label": "none", "intensity": 0.0, "coverage": 0.0, "confidence": 0.92},
        "confidence": 0.74 if rain_i else 0.81,
    }


def _events_for(cell, diel):
    out = []
    for i, (slug, common, sci, diel_sig, diel_conf, season_sig, season_conf) in enumerate(FAUNA[diel]):
        onset = 3.4 + i * 8.7
        out.append(
            {
                "label": slug,
                "common_name": common,
                "scientific_name": sci,
                "confidence": round(0.91 - i * 0.13, 3),
                "onset_s": round(onset, 2),
                "offset_s": round(onset + 4.6, 2),
                "diel_signal": diel_sig,
                "diel_confidence": diel_conf,
                "season_signal": season_sig,
                "season_confidence": season_conf,
                "habitat_signal": HABITAT,
            }
        )
    return out


def _ambient_report(cell, season, diel, events):
    """E-A head. Deliberately a weaker prior than the events head."""
    # The ambient head sometimes disagrees on diel — that's what makes the
    # aggregator's `disagreements` block non-empty and honest.
    confused = DIELS[(DIELS.index(diel) + 2) % len(DIELS)] if diel in ("night", "afternoon") else diel
    return {
        "estimated_conditions": {
            "season": season,
            "diel_bin": confused,
            "hour": HOUR[confused],
            "month": MONTH[season],
        },
        "similar_clips": [
            {
                "segment_id": f"{cell}_seg_{i:05d}",
                "source_clip": f"{cell}_clip{i:03d}",
                "similarity": round(0.74 - i * 0.06, 3),
            }
            for i in range(1, 6)
        ],
        "confidence": 0.38,
        "season_confidence": 0.52,
        "head_agreement": 0.6,
        "ood_flag": False,
        "k": 5,
        "tau": 0.1,
    }


def _weather_report(weather):
    def summary(block, variability):
        return {
            "summary": {
                "intensity": block["intensity"],
                "variability": variability,
                "coverage": block["coverage"],
                "label": block["label"],
                "confidence": block["confidence"],
            }
        }

    thunder = summary(weather["thunder"], 0.0)
    thunder["events"] = []
    thunder["mean_interval_s"] = None
    return {
        "observations": {
            "weather": {
                "wind": summary(weather["wind"], 0.36),
                "rain": summary(weather["rain"], 0.58),
                "thunder": thunder,
                "confidence": weather["confidence"],
                "derived_label": weather["type"],
                "warnings": ["Weather head runs on the raw mixture; faint rain can hide under wind."],
            }
        }
    }


def _events_report(cell, events):
    threshold = 0.55
    detected = []
    for ev in events:
        start = ev["onset_s"]
        while start < ev["offset_s"]:
            detected.append(
                {
                    "start_s": round(start, 2),
                    "end_s": round(min(start + 1.0, ev["offset_s"]), 2),
                    "confidence": round(ev["confidence"] - 0.04, 3),
                    "top_label": ev["label"],
                }
            )
            start += 1.0
    return {
        "num_events": len(events),
        "num_detected_windows": len(detected),
        "num_windows": 26,
        "known_species": sorted({e["label"] for e in events}),
        "threshold": threshold,
        "events": [
            {
                "label": ev["label"],
                "onset_s": ev["onset_s"],
                "offset_s": ev["offset_s"],
                "confidence_mean": round(ev["confidence"] - 0.05, 3),
                "confidence_max": ev["confidence"],
                "window_count": max(1, int(ev["offset_s"] - ev["onset_s"])),
                "species_matches": [
                    {"label": ev["common_name"], "score": ev["confidence"]},
                    {"label": "Gray Shrikethrush", "score": round(ev["confidence"] * 0.42, 3)},
                    {"label": "Willie Wagtail", "score": round(ev["confidence"] * 0.27, 3)},
                ],
                "phenology": {
                    "common_name": ev["common_name"],
                    "scientific_name": ev["scientific_name"],
                    "diel_signal": ev["diel_signal"],
                    "diel_confidence": ev["diel_confidence"],
                    "season_signal": ev["season_signal"],
                    "season_confidence": ev["season_confidence"],
                    "habitat_signal": ev["habitat_signal"],
                    "inference_notes": f"Calls concentrated in the {ev['diel_signal']} bin at this site.",
                },
            }
            for ev in events
        ],
        "diagnostics": {"detected_windows": detected},
        "analysis_report": {
            "observations": [
                {"type": "species", "value": ev["common_name"], "confidence": ev["confidence"]} for ev in events
            ],
            "inferred_context": [
                {"type": "diel", "value": events[0]["diel_signal"], "confidence": events[0]["diel_confidence"]},
                {"type": "habitat", "value": HABITAT, "confidence": 0.66},
            ],
            "disagreements": [],
        },
    }


def build_cell(cell: str) -> dict:
    season, diel = cell.split("_")
    weather = _weather_for(cell)
    events = _events_for(cell, diel)
    ambient = _ambient_report(cell, season, diel, events)
    ambient_diel = ambient["estimated_conditions"]["diel_bin"]

    diel_post = 0.86
    season_post = 0.57
    diel_dist = _dist(DIELS, diel, diel_post)
    season_dist = _dist(SEASONS, season, season_post)
    lead = events[0]

    disagreements = []
    if ambient_diel != diel:
        disagreements.append(
            {
                "field": "diel",
                "ambient": ambient_diel,
                "events": diel,
                "resolution": "events_preferred",
                "reason": f"{lead['common_name']} is a stronger time-of-day cue than ambient texture.",
            }
        )
    disagreements.append(
        {
            "field": "season",
            "ambient": season,
            "events": "inconclusive",
            "resolution": "low_confidence_range_reported",
            "reason": "Season evidence was present but too weak for a confident estimate.",
        }
    )

    weather_decision = {
        "label": weather["type"],
        "confidence": weather["confidence"],
        "rain": weather["rain"],
        "wind": weather["wind"],
        "thunder": {**weather["thunder"], "events": [], "mean_interval_s": None},
        "warnings": [],
    }

    decision = {
        "schema_version": "analysis_decision.v1",
        "time_of_day": {
            "value": diel,
            "confidence": diel_post,
            "distribution": diel_dist,
            "evidence": f"E-C: {lead['common_name']} supports {diel}",
        },
        "season": {
            "value": season,
            "confidence": season_post,
            "distribution": season_dist,
            "evidence": f"E-A: ambient bed most consistent with {season}",
        },
        "weather": weather_decision,
        "detected_calls": events,
        "disagreements": disagreements,
        "overall_confidence": 0.72,
        "limitations": [],
    }

    weather_phrase = {
        "none": "still, dry air",
        "wind": f"{weather['wind']['label']} wind",
        "rain": f"{weather['rain']['label']} rain",
        "rain+wind": f"{weather['rain']['label']} rain on a {weather['wind']['label']} wind",
    }[weather["type"]]
    call_names = ", ".join(e["common_name"] for e in events)

    narration = {
        "schema_version": "analysis_narration.v1",
        "source": "deterministic_fallback",
        "summary": (
            f"The recording reads as {diel}, most consistent with {season}, with {weather_phrase}. "
            f"Detected call evidence includes {call_names}."
        ),
        "bullets": [
            f"Time of day: {diel} ({int(diel_post * 100)}%)",
            f"Season: {season} ({int(season_post * 100)}%)",
            f"Weather: {weather['type']} ({int(weather['confidence'] * 100)}%)",
            f"Detected calls: {call_names}",
        ],
        "caveats": ["Demo fixture: these values are pre-authored, not model output."],
    }

    limitations = [
        "Season is hard to infer from a single short clip at this site.",
        "Ambient context is a weak prior, not a ground-truth label.",
        "Demo build: this report is a pre-authored fixture, not live inference.",
    ]

    report = {
        "schema_version": "analysis_aggregator.v1",
        "mode": "analysis",
        "mock": True,
        "observations": {
            "ambient": {
                "similar_clips": ambient["similar_clips"],
                "estimated_conditions": ambient["estimated_conditions"],
                "confidence": ambient["confidence"],
                "season_confidence": ambient["season_confidence"],
                "ood_flag": ambient["ood_flag"],
            },
            "weather": _weather_report(weather)["observations"]["weather"],
            "events": [
                {
                    "label": e["label"],
                    "common_name": e["common_name"],
                    "scientific_name": e["scientific_name"],
                    "confidence": e["confidence"],
                    "onset_s": e["onset_s"],
                    "offset_s": e["offset_s"],
                    "phenology": {
                        "diel_signal": e["diel_signal"],
                        "diel_confidence": e["diel_confidence"],
                        "season_signal": e["season_signal"],
                        "season_confidence": e["season_confidence"],
                        "habitat_signal": e["habitat_signal"],
                    },
                }
                for e in events
            ],
        },
        "inferred_context": {
            "diel": {
                "estimate": diel,
                "posterior": diel_post,
                "distribution": diel_dist,
                "primary_evidence": f"E-C: {lead['common_name']} has a {lead['diel_signal']} activity signal",
                "evidence": [
                    {
                        "source_head": "events",
                        "value": diel,
                        "weight": 0.74,
                        "reason": f"{lead['common_name']} detected with high confidence",
                    },
                    {
                        "source_head": "ambient",
                        "value": ambient_diel,
                        "weight": 0.14,
                        "reason": "Ambient head estimate carried low confidence",
                    },
                ],
            },
            "season": {
                "estimate": season,
                "posterior": season_post,
                "distribution": season_dist,
                "primary_evidence": "Ambient texture and env cues lean this way; no strongly seasonal species",
                "evidence": [
                    {
                        "source_head": "ambient",
                        "value": season,
                        "weight": 0.42,
                        "reason": "Nearest training clips share this season",
                    }
                ],
            },
        },
        "decision": {**decision, "limitations": limitations},
        "narration": narration,
        "llm_input": {
            "schema_version": "analysis_llm_input.v1",
            "task": (
                "Render this ecoacoustic analysis decision JSON as immersive, third-person perspective "
                "narration with an analytical tone. Narrate only the provided observations, inferred "
                "context, disagreements, limitations, timestamps, and confidence values; do not invent "
                "species, season, time of day, weather, certainty, or causes beyond the JSON."
            ),
            "decision": decision,
        },
        "model_lineage": LINEAGE,
        "disagreements": disagreements,
        "overall_confidence": 0.72,
        "limitations": limitations,
    }

    return {
        "cell": cell,
        "report": report,
        "attempts": LINEAGE,
        "head_reports": {
            "ambient": ambient,
            "weather": _weather_report(weather),
            "events": _events_report(cell, events),
        },
    }


def build_all(fixtures: Path) -> None:
    out_dir = fixtures / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        for diel in DIELS:
            cell = f"{season}_{diel}"
            (out_dir / f"{cell}.json").write_text(json.dumps(build_cell(cell), indent=2) + "\n")
    print(f"  analysis reports: {len(SEASONS) * len(DIELS)} cells")


if __name__ == "__main__":
    build_all(Path(__file__).resolve().parents[1] / "fixtures")
