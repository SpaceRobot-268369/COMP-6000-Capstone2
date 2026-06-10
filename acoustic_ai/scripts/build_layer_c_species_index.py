#!/usr/bin/env python3
"""Build the Layer C species index from each attempt's source of truth.

Layer C attempts can only voice the species they were built for. This script
derives a single species manifest per attempt so the frontend `/generation`
species rail (and anything else) stays in lock-step with the data — no
hand-transcribed species lists.

Sources of truth (read-only):
  * burger__mvp_2__retrieval_v2_library  -> data/media_asset_bank/
        layer_c_retrieval_v2_event_index.csv  (unique species_common_name rows)
  * burger__mvp_3__sa3_generative_live   -> registry.yaml `species_pools`

Outputs (both git-tracked metadata, identical content):
  1. CANONICAL, beside the CSV it derives from (per conventions.md: attempt-local
     derived data lives under <attempt>/data/):
        acoustic_ai/layers/layer_c/attempts/
          burger__mvp_2__retrieval_v2_library/data/media_asset_bank/species_index.json
  2. FRONTEND copy (the frontend Docker build context is `frontend/` only, so it
     cannot import the canonical file across packages):
        frontend/src/demo/layerCSpeciesIndex.json

Run:  ./acoustic_ai/.venv/bin/python acoustic_ai/scripts/build_layer_c_species_index.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

# --- paths -----------------------------------------------------------------

_AI_ROOT = Path(__file__).resolve().parent.parent          # acoustic_ai/
_PROJECT_ROOT = _AI_ROOT.parent
_REGISTRY = _AI_ROOT / "registry.yaml"

_RETRIEVAL_ATTEMPT = "burger__mvp_2__retrieval_v2_library"
_SA3_ATTEMPT = "burger__mvp_3__sa3_generative_live"

_RETRIEVAL_DATA = (
    _AI_ROOT / "layers" / "layer_c" / "attempts" / _RETRIEVAL_ATTEMPT
    / "data" / "media_asset_bank"
)
_RETRIEVAL_CSV = _RETRIEVAL_DATA / "layer_c_retrieval_v2_event_index.csv"

_CANONICAL_OUT = _RETRIEVAL_DATA / "species_index.json"
_FRONTEND_OUT = _PROJECT_ROOT / "frontend" / "src" / "demo" / "layerCSpeciesIndex.json"


def _retrieval_species() -> list[dict]:
    """Unique (event_type, common name) pairs from the v2 event index CSV."""
    seen: dict[str, str] = {}
    with _RETRIEVAL_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("event_type") or "").strip()
            label = (row.get("species_common_name") or "").strip()
            if key and label and key not in seen:
                seen[key] = label
    return [{"key": k, "label": seen[k]} for k in sorted(seen)]


# SA3's species_pools live in registry.yaml on the feat/burger/layer-c-generative
# branch; this branch may not carry that attempt yet. Fall back to its two
# documented pools so the index is complete regardless of which branch is built.
_SA3_FALLBACK = [
    {"key": "horsfields_bronze_cuckoo", "label": "Horsfield's Bronze-cuckoo"},
    {"key": "spotted_nightjar", "label": "Spotted Nightjar"},
]


def _sa3_species() -> list[dict]:
    """Species pools declared for the SA3 live attempt in registry.yaml.

    Falls back to `_SA3_FALLBACK` when the attempt isn't present in the
    registry on the current branch.
    """
    doc = yaml.safe_load(_REGISTRY.read_text(encoding="utf-8"))
    att = doc.get("layers", {}).get("layer_c", {}).get("attempts", {}).get(_SA3_ATTEMPT)
    if not att:
        return list(_SA3_FALLBACK)
    pools = att.get("params", {}).get("species_pools", {}) or {}
    out = [
        {"key": key, "label": (pool.get("species_common_name") or key).strip()}
        for key, pool in pools.items()
    ]
    return sorted(out, key=lambda s: s["key"]) or list(_SA3_FALLBACK)


def build_index() -> dict:
    return {
        "_generated_by": "acoustic_ai/scripts/build_layer_c_species_index.py",
        "_note": "Derived artifact — do not edit by hand; rerun the script.",
        "attempts": {
            _RETRIEVAL_ATTEMPT: {"kind": "retrieval", "species": _retrieval_species()},
            _SA3_ATTEMPT: {"kind": "generative", "species": _sa3_species()},
        },
    }


def main() -> None:
    index = build_index()
    payload = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    for out in (_CANONICAL_OUT, _FRONTEND_OUT):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        counts = {a: len(v["species"]) for a, v in index["attempts"].items()}
        print(f"wrote {out.relative_to(_PROJECT_ROOT)}  {counts}")


if __name__ == "__main__":
    main()
