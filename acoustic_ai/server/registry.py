"""Registry loader: reads ``acoustic_ai/registry.yaml`` and resolves attempts
to their handler modules.

The FastAPI server uses this to:
  - serve ``GET /layers`` (drives the frontend dropdown)
  - dispatch ``POST /layers/{layer}/attempts/{attempt_id}/generate``

Naming rules: ``.claude/context/dev/attempt_naming.md``
Handler interface (each attempt's ``handler.py`` must expose these):

    load(checkpoint_dir: Path | None, params: dict) -> object
        # One-time load. Returns whatever generate() needs.
        # `checkpoint_dir` is None when registry's `checkpoint:` is null.

    generate(state, seed: int | None, **runtime_params) -> dict
        # Returns {
        #     "wav_bytes": bytes,
        #     "mel_db":    np.ndarray | None,
        #     "metadata":  dict,
        # }
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# --- paths -----------------------------------------------------------------

_AI_ROOT = Path(__file__).resolve().parent.parent     # acoustic_ai/
_PROJECT_ROOT = _AI_ROOT.parent
_REGISTRY_PATH = _AI_ROOT / "registry.yaml"

# --- types -----------------------------------------------------------------


@dataclass(frozen=True)
class AttemptSpec:
    layer:   str           # "layer_a"
    id:      str           # "lucas__smoke_1__audioldm2_spring_night"
    label:   str
    author:  str
    stage:   str           # "smoke_1" | "mvp_1" | "prod_1" | ...
    status:  str
    checkpoint: Path | None
    extra_checkpoints: dict[str, Path]
    params:  dict[str, Any]
    notes:   list[str]

    @property
    def handler_module(self) -> str:
        return f"layers.{self.layer}.attempts.{self.id}.code.handler"


# --- loading ---------------------------------------------------------------


@lru_cache(maxsize=1)
def _registry_doc() -> dict:
    with _REGISTRY_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_checkpoint(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return (_PROJECT_ROOT / str(value)).resolve()


# Weight-file extensions we treat as "real" model binaries (i.e. not pointers).
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt")


def _ckpt_availability(ckpt_dir: Path | None, cell_names: list[str] | None = None) -> dict:
    """Inspect a checkpoint directory and report whether the actual weight
    blobs are materialised on disk (as opposed to only DVC pointer files).

    Returns:
        {
            "available": bool,
            "reason":    str | None,    # human-readable when not available
            "missing":   list[str],     # base filenames that need `dvc pull`
        }
    """
    if ckpt_dir is None:
        # Attempts without a checkpoint (placeholders) are considered
        # available — the handler doesn't need any weights.
        return {"available": True, "reason": None, "missing": []}

    if not ckpt_dir.exists():
        return {
            "available": False,
            "reason": f"checkpoint directory missing: {ckpt_dir}",
            "missing": [],
        }

    # Bank layout: weights live in per-cell subdirs (ckpt_dir/<cell>/adapter…).
    # Available iff every declared cell has its weight on disk; otherwise list
    # the missing pointers so the UI can name the exact `dvc pull` targets.
    if cell_names:
        missing = []
        for cell in cell_names:
            w = ckpt_dir / cell / "adapter_model.safetensors"
            if not w.is_file():
                missing.append(f"{cell}/adapter_model.safetensors")
        if not missing:
            return {"available": True, "reason": None, "missing": []}
        return {
            "available": False,
            "reason": (f"{len(missing)}/{len(cell_names)} cell adapters not on disk. "
                       f"Run `dvc pull` under {ckpt_dir}."),
            "missing": missing,
        }

    has_weight = any(
        p.is_file() and p.suffix in _WEIGHT_SUFFIXES
        for p in ckpt_dir.iterdir()
    )
    if has_weight:
        return {"available": True, "reason": None, "missing": []}

    # No real weights — list any `.dvc` pointer files so the UI can tell the
    # user exactly which paths to `dvc pull`.
    pointers = sorted(
        p.name[:-4]  # strip ".dvc"
        for p in ckpt_dir.iterdir()
        if p.is_file() and p.name.endswith(".dvc")
        and any(p.name.endswith(s + ".dvc") for s in _WEIGHT_SUFFIXES)
    )
    if pointers:
        return {
            "available": False,
            "reason": (
                "Weights are DVC-tracked but not on disk locally. "
                f"Run `dvc pull` for: {', '.join(pointers)}"
            ),
            "missing": pointers,
        }
    return {
        "available": False,
        "reason": f"no weight files found in {ckpt_dir}",
        "missing": [],
    }


def list_layers() -> list[dict]:
    """Return the dropdown payload — one entry per layer, with attempt list."""
    out: list[dict] = []
    for layer_id, layer_block in _registry_doc()["layers"].items():
        attempts = []
        for att_id, att in layer_block["attempts"].items():
            ckpt = _resolve_checkpoint(att.get("checkpoint"))
            params = att.get("params") or {}
            cells = sorted((params.get("cells") or {}).keys())
            avail = _ckpt_availability(ckpt, cell_names=cells)
            attempts.append({
                "id":      att_id,
                "label":   att.get("label", att_id),
                "stage":   att.get("stage", ""),
                "author":  att.get("author", ""),
                "status":  att.get("status", ""),
                "checkpoint":       str(ckpt) if ckpt else None,
                "available":        avail["available"],
                "unavailable_reason": avail["reason"],
                "missing_files":    avail["missing"],
                "uses_seed":        bool(att.get("uses_seed", False)),
                "uses_cells":       bool(att.get("uses_cells", False)),
                "cells":            cells,
                "default_cell":     params.get("default_cell"),
            })
        out.append({
            "id":      layer_id,
            "label":   layer_block.get("label", layer_id),
            "default": layer_block.get("default"),
            "attempts": attempts,
        })
    return out


def get_attempt(layer_id: str, attempt_id: str) -> AttemptSpec:
    doc = _registry_doc()
    layers = doc.get("layers", {})
    if layer_id not in layers:
        raise KeyError(f"unknown layer: {layer_id!r}")
    attempts = layers[layer_id].get("attempts", {})
    if attempt_id not in attempts:
        raise KeyError(f"unknown attempt: {layer_id}/{attempt_id}")

    att = attempts[attempt_id]
    extras = {
        k: _resolve_checkpoint(v)
        for k, v in (att.get("extra_checkpoints") or {}).items()
    }
    return AttemptSpec(
        layer=layer_id,
        id=attempt_id,
        label=att.get("label", attempt_id),
        author=att.get("author", ""),
        stage=att.get("stage", ""),
        status=att.get("status", ""),
        checkpoint=_resolve_checkpoint(att.get("checkpoint")),
        extra_checkpoints=extras,
        params=dict(att.get("params") or {}),
        notes=list(att.get("notes") or []),
    )


# --- handler dispatch ------------------------------------------------------

# --- samples ---------------------------------------------------------------

_CANONICAL_SEED_DEFAULT = 42


def canonical_seed(spec: AttemptSpec) -> int:
    """Per-attempt canonical seed (registry override, else project default)."""
    samples_cfg = spec.params.get("samples") if isinstance(spec.params, dict) else None
    if isinstance(samples_cfg, dict) and "canonical_seed" in samples_cfg:
        return int(samples_cfg["canonical_seed"])
    return _CANONICAL_SEED_DEFAULT


def _samples_root(spec: AttemptSpec) -> Path:
    """Attempt root — the `expected/` and `showcase/` tiers live directly here."""
    return _AI_ROOT / "layers" / spec.layer / "attempts" / spec.id


def list_samples(layer_id: str, attempt_id: str) -> dict:
    """Enumerate what's actually present on disk under <attempt>/expected and
    <attempt>/showcase. Returns artefact descriptors the server can hand to the
    frontend; PNG/JSON contents are inlined (small) and WAV presence is a
    flag (caller fetches separately if wanted).
    """
    spec = get_attempt(layer_id, attempt_id)
    root = _samples_root(spec)
    seed = canonical_seed(spec)

    def _scan(tier: str) -> list[dict]:
        d = root / tier
        if not d.is_dir():
            return []
        entries: dict[str, dict] = {}
        for f in sorted(d.iterdir()):
            if f.name in {".gitkeep", ".gitignore"}:
                continue
            # Strip suffixes to a triplet stem.
            stem = f.name
            for suffix in (".wav.dvc", ".png.dvc", ".metadata.json.dvc",
                           ".wav", ".png", ".metadata.json"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            triplet = entries.setdefault(stem, {
                "stem": stem,
                "has_png": False, "has_wav": False, "has_json": False,
                "png_b64": None, "metadata": None, "wav_url": None,
            })
            name = f.name
            if name.endswith(".png"):
                triplet["has_png"] = True
                try:
                    import base64 as _b64
                    triplet["png_b64"] = _b64.b64encode(f.read_bytes()).decode("ascii")
                except OSError:
                    pass
            elif name.endswith(".metadata.json"):
                triplet["has_json"] = True
                try:
                    import json as _json
                    triplet["metadata"] = _json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            elif name.endswith(".wav"):
                triplet["has_wav"] = True
                triplet["wav_url"] = (
                    f"/layers/{layer_id}/attempts/{attempt_id}/samples/{tier}/{stem}.wav"
                )
            # .dvc pointers count as "exists" but we can't inline contents.
            elif name.endswith(".png.dvc"):
                triplet["has_png"] = True
            elif name.endswith(".wav.dvc"):
                triplet["has_wav"] = True
            elif name.endswith(".metadata.json.dvc"):
                triplet["has_json"] = True
        return list(entries.values())

    return {
        "attempt":        attempt_id,
        "layer":          layer_id,
        "canonical_seed": seed,
        "expected":       _scan("expected"),
        "showcase":       _scan("showcase"),
    }


def sample_wav_path(layer_id: str, attempt_id: str, tier: str, stem: str) -> Path:
    """Resolve an <attempt>/<tier>/<stem>.wav path, restricted to legal tiers."""
    if tier not in {"expected", "showcase"}:
        raise ValueError(f"illegal tier: {tier}")
    spec = get_attempt(layer_id, attempt_id)
    safe_stem = stem.replace("/", "_").replace("..", "_")
    return _samples_root(spec) / tier / f"{safe_stem}.wav"


# --- handler dispatch ------------------------------------------------------

_state_cache: dict[str, object] = {}
_state_lock = threading.Lock()


def _get_handler_module(spec: AttemptSpec):
    return importlib.import_module(spec.handler_module)


def _get_state(spec: AttemptSpec):
    cache_key = f"{spec.layer}/{spec.id}"
    if cache_key in _state_cache:
        return _state_cache[cache_key]
    with _state_lock:
        if cache_key in _state_cache:
            return _state_cache[cache_key]
        mod = _get_handler_module(spec)
        state = mod.load(spec.checkpoint, dict(spec.params), extra=spec.extra_checkpoints)
        _state_cache[cache_key] = state
        return state


def generate(layer_id: str, attempt_id: str, seed: int | None, **runtime_params) -> dict:
    """Dispatch to the attempt's handler.generate().

    Returns whatever the handler returns (typically dict with wav_bytes,
    mel_db, metadata).
    """
    spec = get_attempt(layer_id, attempt_id)
    mod = _get_handler_module(spec)
    state = _get_state(spec)
    result = mod.generate(state, seed=seed, **runtime_params)
    # Always attach the spec snapshot for traceability.
    md = result.setdefault("metadata", {})
    md.setdefault("attempt", {
        "layer":  spec.layer,
        "id":     spec.id,
        "label":  spec.label,
        "stage":  spec.stage,
        "author": spec.author,
        "status": spec.status,
        "checkpoint": str(spec.checkpoint) if spec.checkpoint else None,
    })
    return result
