"""Canned analysis: upload → cell → pre-authored report, plus the narrator.

Preset recordings are recognised by MD5 of their bytes (the HomePage cards
upload the very WAVs vendored under fixtures/layers, renamed to preset.wav, so
filename tells us nothing and content tells us everything). Anything else gets
a stable pseudo-random cell derived from the same hash, so an arbitrary upload
still produces a complete, self-consistent report instead of an error.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from .settings import CELLS, DEFAULT_CELL, FIXTURES


@lru_cache(maxsize=1)
def _preset_map() -> dict:
    path = FIXTURES / "analysis" / "presets.json"
    return json.loads(path.read_text()) if path.exists() else {}


@lru_cache(maxsize=32)
def _bundle(cell: str) -> dict:
    path = FIXTURES / "analysis" / f"{cell}.json"
    if not path.exists():
        path = FIXTURES / "analysis" / f"{DEFAULT_CELL}.json"
    return json.loads(path.read_text())


def cell_for_upload(payload: bytes) -> tuple[str, bool]:
    """(cell, was_recognised_preset)"""
    digest = hashlib.md5(payload).hexdigest()
    hit = _preset_map().get(digest)
    if hit:
        return hit["cell"], True
    return CELLS[int(digest[:8], 16) % len(CELLS)], False


def bundle_for_upload(payload: bytes) -> dict:
    cell, recognised = cell_for_upload(payload)
    bundle = json.loads(json.dumps(_bundle(cell)))  # deep copy
    bundle["report"]["mock_source"] = "preset_recording" if recognised else "unrecognised_upload"
    if not recognised:
        bundle["report"]["limitations"].insert(
            0,
            "This upload is not one of the demo's preset recordings; the demo build "
            "returns a representative report rather than a measured one.",
        )
    return bundle


def bundle_for_cell(cell: str) -> dict:
    return json.loads(json.dumps(_bundle(cell if cell in CELLS else DEFAULT_CELL)))


# --------------------------------------------------------------- narration
def _decision_bits(report: dict) -> dict:
    """Pull the narratable fields out of either a full aggregator report or the
    partial one the frontend synthesises from a generation contract
    (frontend/src/lib/generationReport.js) — which has `decision` and nothing else."""
    decision = report.get("decision") or {}
    weather = decision.get("weather") or {}
    calls = decision.get("detected_calls") or []
    inferred = report.get("inferred_context") or {}

    diel = (decision.get("time_of_day") or {}).get("value") or (inferred.get("diel") or {}).get("estimate") or "night"
    season = (decision.get("season") or {}).get("value") or (inferred.get("season") or {}).get("estimate") or "autumn"

    rain = weather.get("rain") or {}
    wind = weather.get("wind") or {}
    thunder = weather.get("thunder") or {}
    names = [c.get("common_name") or c.get("label") or "an unidentified call" for c in calls]

    return {
        "diel": diel,
        "season": season,
        "weather_label": weather.get("label") or "none",
        "rain_label": rain.get("label") or "none",
        "wind_label": wind.get("label") or "none",
        "thunder_label": thunder.get("label") or "none",
        "names": names,
        "diel_conf": (decision.get("time_of_day") or {}).get("confidence"),
        "season_conf": (decision.get("season") or {}).get("confidence"),
        "overall": report.get("overall_confidence"),
    }


def _weather_phrase(bits: dict) -> str:
    rain, wind = bits["rain_label"], bits["wind_label"]
    wet = rain not in ("", "none")
    windy = wind not in ("", "none")
    if wet and windy:
        return f"{rain} rain carried on a {wind} wind"
    if wet:
        return f"{rain} rain"
    if windy:
        return f"a {wind} wind moving through the canopy"
    return "still, dry air"


def _join(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def narrate(report: dict, register: str) -> dict:
    bits = _decision_bits(report)
    weather = _weather_phrase(bits)
    calls = _join(bits["names"])

    if register == "immersive":
        lines = [
            f"It is {bits['diel']} in the dry woodland, and the air belongs to {bits['season']}.",
            f"The recording holds {weather}.",
        ]
        if calls:
            verb = "mark" if len(bits["names"]) > 1 else "marks"
            lines.append(f"Out in the canopy, {calls} {verb} the time more surely than the light does.")
        else:
            lines.append("Nothing calls across it — only the bed of the place itself, holding steady.")
        lines.append("This is what the recording remembers.")
        text = " ".join(lines)
    else:
        parts = [
            f"**Time of day:** {bits['diel']}"
            + (f" (confidence {bits['diel_conf']:.2f})" if isinstance(bits["diel_conf"], (int, float)) else ""),
            f"**Season:** {bits['season']}"
            + (f" (confidence {bits['season_conf']:.2f})" if isinstance(bits["season_conf"], (int, float)) else ""),
            f"**Weather:** {bits['weather_label']} — {weather}.",
            f"**Detected calls:** {calls}." if calls else "**Detected calls:** none above threshold.",
        ]
        if isinstance(bits["overall"], (int, float)):
            parts.append(f"**Overall confidence:** {bits['overall']:.2f}.")
        for note in (report.get("limitations") or [])[:2]:
            parts.append(f"**Caveat:** {note}")
        text = "\n\n".join(parts)

    return {
        "register": register,
        "text": text,
        "source": "mock_deterministic_writer",
        "faithful": True,
        "violations": [],
        "mock": True,
    }
