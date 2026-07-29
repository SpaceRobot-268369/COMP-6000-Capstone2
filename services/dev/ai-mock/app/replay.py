"""Pre-baked artifact replay — the whole of the mock's "generation".

Nothing is synthesised at request time. A request is scored against the baked
preset set, the winner's files are read off disk and base64'd into the same
envelope the real FastAPI server returns:

    {ok, audio_b64, image_b64, metadata, sample_rate, duration_s[, stems]}

Fixtures come from tools/build_fixtures.py. See README.md.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path

from . import registry
from .settings import CELLS, DEFAULT_CELL, FIXTURES

LAYER_A_ATTEMPT = "lucas__prod_1__per_cell_loras"
STEM_LAYERS = ("layer_a", "layer_b", "layer_c")


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii") if path.exists() else ""


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# ------------------------------------------------------------------ presets
@lru_cache(maxsize=1)
def presets() -> list[dict]:
    root = FIXTURES / "generation"
    out = []
    for case in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        meta = _json(case / "metadata.json")
        if not meta:
            continue
        orch = meta.get("orchestration", {})
        out.append(
            {
                "id": case.name,
                "dir": case,
                "metadata": meta,
                "cell": meta.get("cell", DEFAULT_CELL),
                "season": orch.get("season", ""),
                "diel": orch.get("diel", ""),
                "weather_type": orch.get("weather_type", "none"),
                "intensity": orch.get("intensity", "none"),
                "species": [e["species_common_name"] for e in orch.get("events", [])],
            }
        )
    return out


def pick_preset(
    *,
    season: str | None,
    diel: str | None,
    weather_type: str | None,
    intensity: str | None,
    species: str | None,
    include_weather: bool = True,
    include_events: bool = True,
) -> dict:
    """Score the baked presets against the request; best match wins.

    Off-preset requests still resolve — to the nearest bake — rather than
    erroring, so no page in the demo can dead-end on an unusual prompt.
    """
    want_weather = bool(include_weather and weather_type and weather_type != "none")
    candidates = presets()
    if not candidates:
        raise RuntimeError("no generation fixtures found — run tools/build_fixtures.py")

    def score(p: dict) -> tuple:
        s = 0
        if species and include_events and species in p["species"]:
            s += 8
        elif include_events and species and p["species"]:
            s += 1
        if not include_events and not p["species"]:
            s += 2
        if diel and p["diel"] == diel:
            s += 4
        if season and p["season"] == season:
            s += 3
        has_weather = p["weather_type"] != "none"
        if want_weather and has_weather:
            s += 2
            if p["weather_type"] == weather_type:
                s += 2
            if intensity and p["intensity"] == intensity:
                s += 1
        if not want_weather and not has_weather:
            s += 3
        return (s, p["id"])

    return max(candidates, key=score)


# ------------------------------------------------------------- envelopes
def _envelope(case_dir: Path, metadata: dict) -> dict:
    audio = metadata.get("audio", {})
    return {
        "ok": True,
        "audio_b64": _b64(case_dir / "audio.wav"),
        "image_b64": _b64(case_dir / "spectrogram.png"),
        "metadata": metadata,
        "sample_rate": audio.get("sample_rate", 22050),
        "duration_s": audio.get("duration_s", 30.0),
    }


def render(request: dict) -> dict:
    """POST /generation/render — replay a baked Layer D mix."""
    preset = pick_preset(
        season=request.get("season"),
        diel=request.get("diel"),
        weather_type=request.get("weather_type"),
        intensity=request.get("intensity"),
        species=request.get("species_common_name"),
        include_weather=request.get("include_weather", True),
        include_events=request.get("include_events", True),
    )

    metadata = json.loads(json.dumps(preset["metadata"]))  # deep copy
    orch = metadata.setdefault("orchestration", {})
    # Echo back what was asked for, so the UI's "Models used" / seed readouts
    # reflect the request even though the audio is a fixed bake.
    seed = request.get("seed")
    if seed is not None:
        metadata["seed"] = seed
        orch["seed"] = seed
    for key in ("layer_a_attempt", "layer_b_attempt", "layer_c_attempt", "layer_d_attempt"):
        value = request.get(key)
        if value:
            orch.setdefault("attempts", {})[key.replace("_attempt", "")] = value
    orch["requested"] = {
        k: request.get(k)
        for k in ("seed", "duration_s", "season", "diel", "weather_type", "intensity",
                  "include_weather", "include_events", "species_common_name")
    }
    metadata["mock_preset_id"] = preset["id"]

    response = _envelope(preset["dir"], metadata)

    if request.get("include_stems"):
        stems = {}
        for layer in STEM_LAYERS:
            stem_dir = preset["dir"] / "stems" / layer
            if (stem_dir / "audio.wav").exists():
                stems[layer] = _envelope(stem_dir, _json(stem_dir / "metadata.json"))
        if stems:
            response["stems"] = stems
    return response


# --------------------------------------------------- single-layer replay
def _layer_a_case(cell: str) -> Path | None:
    root = FIXTURES / "layers" / "layer_a" / "attempts" / LAYER_A_ATTEMPT / "expected" / cell
    if not root.is_dir():
        return None
    return next((c for c in sorted(root.iterdir()) if (c / "audio.wav").exists()), None)


def _layer_b_case(weather_type: str | None, intensity: str | None) -> Path | None:
    """Nearest vendored weather stem for the requested controls."""
    root = FIXTURES / "layers" / "layer_b" / "attempts"
    wanted = (weather_type or "wind", intensity or "medium")
    table = {
        ("rain", "light"): ("murphy__mvp_1__rain_intensity_seed_pool", "site_pool_rain_light"),
        ("rain", "medium"): ("murphy__mvp_1__weather_stem_selector", "rain_medium_site_example"),
        ("rain", "heavy"): ("murphy__mvp_1__rain_intensity_seed_pool", "site_pool_rain_heavy"),
        ("wind", "light"): ("murphy__mvp_1__weather_stem_selector", "wind_medium_site_example"),
        ("wind", "medium"): ("murphy__mvp_1__weather_stem_selector", "wind_medium_site_example"),
        ("wind", "heavy"): ("murphy__mvp_1__weather_stem_selector", "wind_medium_site_example"),
        ("rain+wind", "light"): ("murphy__mvp_1__weather_stem_selector", "rain_wind_medium_site_example"),
        ("rain+wind", "medium"): ("murphy__mvp_1__weather_stem_selector", "rain_wind_medium_site_example"),
        ("rain+wind", "heavy"): ("murphy__mvp_1__weather_stem_selector", "rain_wind_medium_site_example"),
    }
    attempt, case = table.get(wanted, table[("wind", "medium")])
    path = root / attempt / "expected" / case
    return path if (path / "audio.wav").exists() else None


@lru_cache(maxsize=1)
def _event_catalog() -> dict:
    path = FIXTURES / "events" / "catalog.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _slug_for_species(common_name: str | None) -> str | None:
    if not common_name:
        return None
    for slug, entry in _event_catalog().items():
        if entry.get("species_common_name", "").lower() == common_name.lower():
            return slug
    return None


def generate(layer_id: str, attempt_id: str, request: dict) -> dict:
    """POST /layers/{layer}/attempts/{attempt}/generate — single-stem replay."""
    snapshot = registry.attempt_snapshot(layer_id, attempt_id)
    seed = request.get("retrieval_seed", request.get("seed"))

    if layer_id == "layer_c":
        return _generate_event(attempt_id, request, snapshot, seed)
    if layer_id == "layer_b":
        case = _layer_b_case(request.get("weather_type"), request.get("intensity") or request.get("wind_intensity"))
        prompt = f"{request.get('intensity') or 'medium'} {request.get('weather_type') or 'wind'} weather stem"
    elif layer_id == "layer_d":
        return render({**request, "include_stems": True})
    else:
        cell = _cell_from(request)
        case = _layer_a_case(cell) or _layer_a_case(DEFAULT_CELL)
        prompt = f"{cell.split('_')[1]} {cell.split('_')[0]} ambient soundscape, Bowra dry woodland, Australia"

    if case is None:
        raise FileNotFoundError(f"no vendored fixture for {layer_id}/{attempt_id}")

    source = _json(case / "metadata.json")
    metadata = _stem_metadata(source, snapshot, seed, prompt, request)
    if layer_id == "layer_a":
        metadata["cell"] = _cell_from(request)
    return _envelope(case, metadata)


def _generate_event(attempt_id: str, request: dict, snapshot: dict, seed) -> dict:
    """Layer C replays one reference call for the requested species."""
    common = request.get("species_common_name")
    slug = _slug_for_species(common)
    events_dir = FIXTURES / "events"
    if slug and (events_dir / f"{slug}.wav").exists():
        entry = _event_catalog().get(slug, {})
        metadata = _stem_metadata({}, snapshot, seed, f"{entry.get('species_common_name', slug)} call", request)
        metadata.update(
            {
                "species_common_name": entry.get("species_common_name", slug),
                "species_scientific_name": entry.get("species_scientific_name", ""),
                "asset_id": entry.get("asset_id", slug),
                "audio": {
                    "sample_rate": 22050,
                    "duration_s": entry.get("duration_s", 3.6),
                    "rms": 0.061,
                    "peak": 0.42,
                },
            }
        )
        return {
            "ok": True,
            "audio_b64": _b64(events_dir / f"{slug}.wav"),
            "image_b64": _b64(events_dir / f"{slug}.png"),
            "metadata": metadata,
            "sample_rate": 22050,
            "duration_s": entry.get("duration_s", 3.6),
        }

    # Unknown/unset species: fall back to a vendored layer_c expected case.
    root = FIXTURES / "layers" / "layer_c" / "attempts" / attempt_id / "expected"
    if not root.is_dir():
        root = FIXTURES / "layers" / "layer_c" / "attempts" / "burger__mvp_2__retrieval_v2_library" / "expected"
    case = next((c for c in sorted(root.iterdir()) if (c / "audio.wav").exists()), None) if root.is_dir() else None
    if case is None:
        raise FileNotFoundError("no vendored layer_c fixture")
    metadata = _stem_metadata(_json(case / "metadata.json"), snapshot, seed, common or "site species call", request)
    return _envelope(case, metadata)


def _cell_from(request: dict) -> str:
    season, diel = request.get("season"), request.get("diel")
    cell = f"{season}_{diel}"
    return cell if cell in CELLS else DEFAULT_CELL


def _stem_metadata(source: dict, snapshot: dict, seed, prompt: str, request: dict) -> dict:
    audio = source.get("audio") or {
        "sample_rate": 22050,
        "duration_s": 13.1,
        "rms": 0.055,
        "peak": 0.38,
    }
    metadata = {
        "mock": True,
        "mock_note": "Replayed demo fixture — no model was run.",
        "generator": "ai_mock_replay",
        "prompt": prompt,
        "prompt_locked": True,
        "seed": seed,
        "audio": audio,
        "attempt": snapshot,
        "source_metadata": source,
    }
    if request.get("retrieval_seed") is not None:
        metadata["retrieval_seed"] = request["retrieval_seed"]
    return metadata
