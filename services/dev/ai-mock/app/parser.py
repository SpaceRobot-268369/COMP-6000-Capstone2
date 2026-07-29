"""Fake prompt parser for POST /generation/parse.

Stands in for the LLM-OSS parser described in
.claude/context/ai/prompt_parser_policy.md. Same three-stage behaviour
(default-fill → validity gate → layer contracts) and the same output contract,
but the "understanding" is keyword matching.

The word lists deliberately mirror frontend/src/demo/resolvePrompt.js, which
drives the 3D scene from the same prompt on the client side. If the two drift,
the immersive scene renders rain over an audio mix that has none.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from .settings import FIXTURES

SEASON_WORDS = {
    "spring": ["spring", "blossom", "bloom"],
    "summer": ["summer", "hot", "humid", "cicada", "cicadas"],
    "autumn": ["autumn", "fall", "falling leaves", "leaves"],
    "winter": ["winter", "snow", "snowy", "frost", "frozen", "cold"],
}

DIEL_WORDS = {
    "dawn": ["dawn", "sunrise", "daybreak", "first light", "early morning"],
    "morning": ["morning", "mid-morning"],
    "afternoon": ["afternoon", "midday", "noon", "daytime"],
    "night": ["night", "evening", "dusk", "nightfall", "midnight", "moonlit", "moon", "twilight"],
}

RAIN_WORDS = ["rain", "rainy", "raining", "drizzle", "shower", "showers", "downpour", "wet", "storm", "stormy", "thunderstorm"]
WIND_WORDS = ["wind", "windy", "breeze", "breezy", "gust", "gusts", "gusty", "blustery", "blowing"]
HEAVY_WORDS = ["downpour", "heavy", "torrential", "pouring", "gale", "gusty", "gusts", "blustery", "howling", "strong", "fierce"]
LIGHT_WORDS = ["drizzle", "faint", "gentle", "soft", "light", "breeze", "breezy"]

# Things this site cannot voice. Hitting one of these is what turns a request
# into "corrected" (drop the impossible part) or "rejected" (nothing is left).
IMPOSSIBLE = {
    "ocean": ["ocean", "sea", "waves", "beach", "surf", "shore", "seagull", "seagulls"],
    "urban": ["city", "traffic", "downtown", "car", "cars", "horn", "horns", "siren", "sirens",
              "subway", "train", "engine", "motorway", "highway", "street"],
    "rainforest": ["rainforest", "jungle", "tropical", "monkey", "monkeys"],
    "snowfall": ["snow", "snowy", "blizzard", "snowfall"],
}

IMPOSSIBLE_LABEL = {
    "ocean": "ocean waves",
    "urban": "city and traffic noise",
    "rainforest": "rainforest fauna",
    "snowfall": "falling snow",
}

# Generic "some birds" phrasing → a couple of common site species, so a prompt
# that never names a species still gets events.
GENERIC_BIRD_WORDS = ["birdsong", "birds", "bird", "chorus", "dawn chorus"]
GENERIC_BIRDS = ["Yellow-throated Miner", "Willie Wagtail"]

DENSITY_FREQUENT = ["restless", "over and over", "constant", "constantly", "repeatedly", "frequent",
                    "chorus", "many", "lots", "busy"]
DENSITY_SPARSE = ["lone", "a single", "occasional", "occasionally", "every so often", "distant", "sparse", "now and then"]


@lru_cache(maxsize=1)
def _species_catalog() -> dict:
    path = FIXTURES / "events" / "catalog.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _has(text: str, words) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def _count(text: str, words) -> int:
    return sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", text))


def _pick(text: str, table: dict, fallback: str) -> tuple[str, bool]:
    """Returns (choice, was_explicit) so we can report filled defaults."""
    best, best_score = fallback, 0
    for key, words in table.items():
        score = _count(text, words)
        if score > best_score:
            best, best_score = key, score
    return best, best_score > 0


def _normalise(text: str) -> str:
    """Fold the apostrophe variants so "Horsfield's" matches "Horsfield's"."""
    return text.replace("’", "'").replace("ʼ", "'").lower()


def _species(text: str) -> list[str]:
    """Match species by full common name; fall back to a distinctive tail word.

    The fallback only fires when the tail word identifies exactly one species —
    "nightjar" alone is ambiguous here (Spotted and White-throated are both in
    the bank), and guessing would put a bird in the scene the user never asked
    for.
    """
    haystack = _normalise(text)
    catalog = _species_catalog()

    named = [
        entry["species_common_name"]
        for entry in catalog.values()
        if entry.get("species_common_name") and _normalise(entry["species_common_name"]) in haystack
    ]
    if named:
        return sorted(dict.fromkeys(named), key=lambda s: (-len(s), s))[:3]

    by_tail: dict[str, list[str]] = {}
    for slug, entry in catalog.items():
        common = entry.get("species_common_name")
        if common:
            by_tail.setdefault(slug.split("_")[-1], []).append(common)

    found = [
        names[0]
        for tail, names in by_tail.items()
        if len(names) == 1 and len(tail) > 4 and re.search(rf"\b{re.escape(tail)}s?\b", haystack)
    ]
    return sorted(dict.fromkeys(found))[:3]


def parse(prompt: str) -> dict:
    text = f" {prompt.lower().strip()} "
    filled: list[str] = []

    season, season_explicit = _pick(text, SEASON_WORDS, "autumn")
    diel, diel_explicit = _pick(text, DIEL_WORDS, "dawn")
    if not season_explicit:
        filled.append(f"season:{season}")
    if not diel_explicit:
        filled.append(f"diel:{diel}")

    rain = _has(text, RAIN_WORDS)
    wind = _has(text, WIND_WORDS)
    if rain and wind:
        weather_type = "rain+wind"
    elif rain:
        weather_type = "rain"
    elif wind:
        weather_type = "wind"
    else:
        weather_type = None

    if weather_type:
        if _has(text, HEAVY_WORDS):
            intensity = "heavy"
        elif _has(text, LIGHT_WORDS):
            intensity = "light"
        else:
            intensity = "medium"
        layer_b = {"weather_type": weather_type, "intensity": intensity, "duration_s": 30}
    else:
        layer_b = None
        filled.append("weather:none")

    species = _species(text)
    if not species and _has(text, GENERIC_BIRD_WORDS):
        species = list(GENERIC_BIRDS)
    if _has(text, DENSITY_FREQUENT):
        density = "frequent"
    elif _has(text, DENSITY_SPARSE):
        density = "sparse"
    else:
        density = "sparse"
    layer_c = {"species": species, "density": density}
    if not species:
        filled.append("events:empty")

    # --- validity gate -------------------------------------------------------
    blocked = [key for key, words in IMPOSSIBLE.items() if _has(text, words)]
    # "snow" is also a winter season cue, so only treat it as impossible when
    # the prompt is actually asking to hear snowfall.
    if "snowfall" in blocked and not _has(text, ["snowfall", "blizzard"]):
        blocked.remove("snowfall")

    if blocked:
        dropped = ", ".join(IMPOSSIBLE_LABEL[k] for k in blocked)
        keeps = bool(species) or bool(layer_b)
        if not keeps:
            return {
                "status": "rejected",
                "note": (
                    f"This site is a remote inland dry woodland in western Queensland, so it cannot voice "
                    f"{dropped}, and nothing else in the request is left to build a scene from. "
                    f"Try something like \"summer night, a lone Spotted Nightjar churring\" instead."
                ),
                "filled_defaults": filled,
                "layer_a": None,
                "layer_b": None,
                "layer_c": None,
                "mock": True,
            }
        return {
            "status": "corrected",
            "note": (
                f"This site is a remote inland dry woodland, so I dropped {dropped} and kept the rest of "
                f"the scene: {season} {diel}"
                + (f", {layer_b['intensity']} {layer_b['weather_type']}" if layer_b else "")
                + (f", {', '.join(species)}" if species else "")
                + "."
            ),
            "filled_defaults": filled + [f"dropped:{k}" for k in blocked],
            "layer_a": {"season": season, "diel": diel},
            "layer_b": layer_b,
            "layer_c": layer_c,
            "mock": True,
        }

    return {
        "status": "ok",
        "note": "",
        "filled_defaults": filled,
        "layer_a": {"season": season, "diel": diel},
        "layer_b": layer_b,
        "layer_c": layer_c,
        "mock": True,
    }
