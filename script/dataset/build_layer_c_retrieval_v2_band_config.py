#!/usr/bin/env python3
"""Create initial species band defaults for Layer C retrieval v2.

These bands are starting points for review-package generation, not final
ornithological truth. The review pass should tighten or override bands per
sample when the mel/spectrogram shows the target call more clearly.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_ROOT = REPO_ROOT / "resources" / "site_257_bowra-dry-a" / "layer_c_retrieval_event_library_v2"
DEFAULT_QUOTA = LIB_ROOT / "species_quota_v2.csv"
DEFAULT_OUTPUT = LIB_ROOT / "species_band_config_v2.csv"


EXACT_BANDS = {
    "chestnut_rumped_thornbill": (3500, 9500, "prior_human_pass"),
    "crested_bellbird": (700, 4000, "prior_human_pass"),
    "white_browed_woodswallow": (1200, 8000, "prior_human_pass"),
    "red_capped_robin": (1800, 8500, "prior_human_pass"),
    "superb_fairywren": (2800, 10000, "prior_human_pass"),
    "horsfields_bronze_cuckoo": (2100, 4100, "prior_final_pass"),
    "splendid_fairywren": (3000, 10000, "prior_final_pass"),
    "southern_boobook": (480, 800, "prior_human_pass_not_in_other_tags_top63"),
}


def slugify(value: str) -> str:
    value = value.lower().replace("'", "")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def heuristic_band(common_name: str, scientific_name: str) -> tuple[int, int, str]:
    name = common_name.lower()
    sci = scientific_name.lower()
    if any(token in name for token in ("raven", "owl", "frogmouth")):
        return 300, 3500, "heuristic_low_voiced_large_or_nocturnal_bird"
    if "nightjar" in name:
        return 600, 5500, "heuristic_nocturnal_churring_or_call_band"
    if any(token in name for token in ("thornbill", "fairywren", "finch", "pardalote", "firetail", "fantail", "weebill")):
        return 2500, 10000, "heuristic_small_passerine_high_band"
    if any(token in name for token in ("honeyeater", "myzomela", "spinebill", "miner")):
        return 1500, 9000, "heuristic_honeyeater_high_mid_band"
    if any(token in name for token in ("parrot", "cockatoo", "cockatiel", "ringneck", "budgerigar")):
        return 800, 7000, "heuristic_parrot_call_band"
    if any(token in name for token in ("cuckoo", "triller", "songlark", "whistler", "shrikethrush", "cuckooshrike", "flycatcher")):
        return 1200, 7000, "heuristic_song_or_whistle_band"
    if any(token in name for token in ("sandpiper", "greenshank", "whimbrel", "coot", "lapwing", "bronzewing", "dove", "osprey", "falcon")):
        return 700, 6500, "heuristic_nonpasserine_call_band"
    if "artamus" in sci:
        return 1200, 8000, "heuristic_woodswallow_band"
    return 1000, 8000, "heuristic_general_bird_call_band"


def main() -> int:
    rows: list[dict[str, Any]] = []
    for quota in read_csv(DEFAULT_QUOTA):
        common = quota["species_common_name"]
        scientific = quota["species_scientific_name"]
        slug = slugify(common)
        if slug in EXACT_BANDS:
            low, high, source = EXACT_BANDS[slug]
        else:
            low, high, source = heuristic_band(common, scientific)
        rows.append(
            {
                "rank": quota["rank"],
                "species_common_name": common,
                "species_scientific_name": scientific,
                "species_slug": slug,
                "default_low_hz": low,
                "default_high_hz": high,
                "band_source": source,
                "review_status": "initial",
                "notes": "Initial v2 default; tighten per sample during spectrogram review.",
            }
        )

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {DEFAULT_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
