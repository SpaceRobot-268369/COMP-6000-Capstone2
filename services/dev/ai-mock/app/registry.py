"""GET /layers for the demo mock.

Mirrors the shape `acoustic_ai/server/registry.py::list_layers` returns, read
from the same `registry.yaml`, with one deliberate difference: every attempt is
reported `available: true`. Nothing is really loadable here — there are no
weights in the demo image — but the frontend disables its Generate buttons and
greys out the /dev/settings slots on `available: false`, so a truthful "no
weights" answer would make every page look broken instead of fake.
"""

from __future__ import annotations

import json
from functools import lru_cache

import yaml

from .settings import CELLS, FIXTURES, REGISTRY_PATH


@lru_cache(maxsize=1)
def _doc() -> dict:
    if not REGISTRY_PATH.exists():
        raise RuntimeError(f"registry.yaml not found at {REGISTRY_PATH}")
    return yaml.safe_load(REGISTRY_PATH.read_text()) or {}


@lru_cache(maxsize=1)
def _event_catalog() -> dict:
    path = FIXTURES / "events" / "catalog.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _species_options(layer_id: str, attempt: dict) -> list[dict]:
    """Species dropdown entries for retrieval-kind attempts.

    The real server derives these from the retrieval bank's index.json; the
    mock derives them from the vendored event-clip catalog, which is that same
    bank narrowed to one reference call per species.
    """
    if layer_id != "layer_c":
        return []
    pools = (attempt.get("params") or {}).get("species_pools")
    if isinstance(pools, dict) and pools:
        slugs = list(pools.keys())
    elif attempt.get("kind") == "retrieval":
        slugs = list(_event_catalog().keys())
    else:
        return []
    catalog = _event_catalog()
    out = []
    for slug in slugs:
        entry = catalog.get(slug, {})
        label = entry.get("species_common_name") or slug.replace("_", " ").title()
        out.append(
            {
                "value": label,
                "label": label,
                "slug": slug,
                "scientific_name": entry.get("species_scientific_name", ""),
            }
        )
    return sorted(out, key=lambda x: x["label"])


def _attempt_payload(layer_id: str, attempt_id: str, attempt: dict) -> dict:
    params = attempt.get("params") or {}
    cells = sorted((params.get("cells") or {}).keys())
    if attempt.get("uses_cells") and not cells:
        # layer_c's retrieval attempt declares 16 empty cells; fall back to the
        # canonical grid so the selector is never rendered with zero options.
        cells = list(CELLS)
    return {
        "id": attempt_id,
        "label": attempt.get("label", attempt_id),
        "stage": attempt.get("stage", ""),
        "author": attempt.get("author", ""),
        "head": attempt.get("head"),
        "kind": attempt.get("kind") or "generative",
        "status": attempt.get("status", ""),
        "description": attempt.get("description", ""),
        "checkpoint": attempt.get("checkpoint"),
        "asset_bank": attempt.get("asset_bank"),
        "available": True,
        "unavailable_reason": None,
        "missing_files": [],
        "uses_seed": bool(attempt.get("uses_seed", False)),
        "uses_cells": bool(attempt.get("uses_cells", False)),
        "uses_weather_controls": bool(attempt.get("uses_weather_controls", False)),
        "cells": cells,
        "default_cell": params.get("default_cell") or (cells[0] if cells else None),
        "species_options": _species_options(layer_id, attempt),
        "params": params,
        "mock": True,
    }


def list_layers() -> list[dict]:
    out = []
    for layer_id, block in (_doc().get("layers") or {}).items():
        attempts = [
            _attempt_payload(layer_id, aid, att or {})
            for aid, att in (block.get("attempts") or {}).items()
        ]
        out.append(
            {
                "id": layer_id,
                "label": block.get("label", layer_id),
                "default": block.get("default"),
                "attempts": attempts,
            }
        )
    return out


def get_layer(layer_id: str) -> dict | None:
    return next((layer for layer in list_layers() if layer["id"] == layer_id), None)


def get_attempt(layer_id: str, attempt_id: str) -> dict | None:
    layer = get_layer(layer_id)
    if not layer:
        return None
    return next((a for a in layer["attempts"] if a["id"] == attempt_id), None)


def attempt_snapshot(layer_id: str, attempt_id: str) -> dict:
    attempt = get_attempt(layer_id, attempt_id) or {}
    return {
        "layer": layer_id,
        "id": attempt_id,
        "label": attempt.get("label", attempt_id),
        "stage": attempt.get("stage", ""),
        "head": attempt.get("head"),
        "author": attempt.get("author", ""),
        "status": attempt.get("status", ""),
        "checkpoint": attempt.get("checkpoint"),
        "asset_bank": attempt.get("asset_bank"),
    }
