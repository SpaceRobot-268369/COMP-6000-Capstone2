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
import sys
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

if str(_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_ROOT))

# --- types -----------------------------------------------------------------


@dataclass(frozen=True)
class AttemptSpec:
    layer:   str           # "layer_a"
    id:      str           # "lucas__smoke_1__audioldm2_spring_night"
    label:   str
    author:  str
    stage:   str           # "smoke_1" | "mvp_1" | "prod_1" | ...
    head:    str | None    # Layer E only: "ambient" | "weather" | "events"
    status:  str
    kind:    str | None
    checkpoint: Path | None
    asset_bank: Path | None
    extra_checkpoints: dict[str, Path]
    params:  dict[str, Any]
    notes:   list[str]

    @property
    def handler_module(self) -> str:
        return f"layers.{self.layer}.attempts.{self.id}.code.handler"

    @property
    def artifact_root(self) -> Path | None:
        if self.kind == "retrieval" and self.asset_bank is not None:
            return self.asset_bank
        return self.checkpoint


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


def _bank_availability(bank_dir: Path | None) -> dict:
    if bank_dir is None:
        return {"available": True, "reason": None, "missing": []}
    if not bank_dir.exists():
        return {
            "available": False,
            "reason": f"asset bank directory missing: {bank_dir}",
            "missing": [],
        }
    index = bank_dir / "index.json"
    media_dir = bank_dir / "media_asset_bank"
    missing = []
    if not index.is_file():
        missing.append("index.json")
    if not media_dir.exists():
        missing.append("media_asset_bank/")
    elif not any(media_dir.rglob("*.wav")):
        pointers = sorted(
            str(p.relative_to(bank_dir))
            for p in media_dir.rglob("*.dvc")
            if p.is_file()
        )
        if pointers:
            return {
                "available": False,
                "reason": (
                    "Asset bank is DVC-tracked but audio is not on disk locally. "
                    f"Run `dvc pull` for: {', '.join(pointers[:3])}"
                    + (" ..." if len(pointers) > 3 else "")
                ),
                "missing": pointers,
            }
        missing.append("media_asset_bank audio")
    if missing:
        return {
            "available": False,
            "reason": f"asset bank incomplete: {', '.join(missing)}",
            "missing": missing,
        }
    return {"available": True, "reason": None, "missing": []}


def list_layers() -> list[dict]:
    """Return the dropdown payload — one entry per layer, with attempt list."""
    out: list[dict] = []
    for layer_id, layer_block in _registry_doc()["layers"].items():
        attempts = []
        for att_id, att in layer_block["attempts"].items():
            kind = att.get("kind")
            ckpt = _resolve_checkpoint(att.get("checkpoint"))
            asset_bank = _resolve_checkpoint(att.get("asset_bank"))
            params = att.get("params") or {}
            cells = sorted((params.get("cells") or {}).keys())
            avail = (
                _bank_availability(asset_bank)
                if kind == "retrieval" and asset_bank is not None
                else _ckpt_availability(ckpt, cell_names=cells)
            )
            attempts.append({
                "id":      att_id,
                "label":   att.get("label", att_id),
                "stage":   att.get("stage", ""),
                "author":  att.get("author", ""),
                "head":    att.get("head"),
                "status":  att.get("status", ""),
                "description": att.get("description", ""),
                "checkpoint":       str(ckpt) if ckpt else None,
                "kind":             kind,
                "asset_bank":       str(asset_bank) if asset_bank else None,
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
        head=att.get("head"),
        status=att.get("status", ""),
        checkpoint=_resolve_checkpoint(att.get("checkpoint")),
        kind=att.get("kind"),
        asset_bank=_resolve_checkpoint(att.get("asset_bank")),
        extra_checkpoints=extras,
        params=dict(att.get("params") or {}),
        notes=list(att.get("notes") or []),
    )


def default_attempt_id(layer_id: str) -> str:
    layers = _registry_doc().get("layers", {})
    if layer_id not in layers:
        raise KeyError(f"unknown layer: {layer_id!r}")
    attempt_id = layers[layer_id].get("default")
    if not attempt_id:
        raise KeyError(f"layer has no default attempt: {layer_id!r}")
    return str(attempt_id)


def default_layer_e_head_attempt(head: str) -> str:
    """Return the default attempt for one Layer E analysis head.

    The layer-level default is the ambient head. For weather/events/aggregator
    we select the first registered attempt whose `head:` field matches.
    """
    layers = _registry_doc().get("layers", {})
    layer = layers.get("layer_e")
    if not layer:
        raise KeyError("unknown layer: 'layer_e'")
    attempts = layer.get("attempts", {})
    layer_default = layer.get("default")
    if layer_default and attempts.get(layer_default, {}).get("head") == head:
        return str(layer_default)
    for attempt_id, attempt in attempts.items():
        if attempt.get("head") == head:
            return str(attempt_id)
    raise KeyError(f"Layer E has no registered {head!r} analysis head")


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


def _is_case_dir(d: Path) -> bool:
    """A case dir holds the fixed triplet {audio.wav, spectrogram.png, metadata.json}
    (plus matching .dvc pointers). Presence of audio.wav or its DVC pointer is
    the canonical signal."""
    return (d / "audio.wav").is_file() or (d / "audio.wav.dvc").is_file()


def _read_case_dir(layer_id: str, attempt_id: str, tier: str,
                   case: Path, *, cell: str | None) -> dict:
    """Build one sample entry from a case sub-directory (canonical layout
    per conventions.md §2.6). Reads audio.wav / .wav.dvc, spectrogram.png /
    .png.dvc and metadata.json / .json.dvc.

    `wav_url` mirrors the on-disk path so the frontend doesn't have to know
    the layout — it just resolves the returned URL.
    """
    import base64 as _b64
    import json as _json

    rel_parts = [tier]
    if cell:
        rel_parts.append(cell)
    rel_parts.append(case.name)
    rel_wav = "/".join(rel_parts + ["audio.wav"])

    entry = {
        "stem":    case.name,
        "cell":    cell,
        "has_wav": False, "has_png": False, "has_json": False,
        "png_b64": None, "metadata": None, "wav_url": None,
    }

    audio_wav = case / "audio.wav"
    audio_dvc = case / "audio.wav.dvc"
    if audio_wav.is_file() or audio_dvc.is_file():
        entry["has_wav"] = True
        entry["wav_url"] = (
            f"/layers/{layer_id}/attempts/{attempt_id}/samples/{rel_wav}"
        )

    png = case / "spectrogram.png"
    if png.is_file():
        entry["has_png"] = True
        try:
            entry["png_b64"] = _b64.b64encode(png.read_bytes()).decode("ascii")
        except OSError:
            pass
    elif (case / "spectrogram.png.dvc").is_file():
        entry["has_png"] = True

    md = case / "metadata.json"
    if md.is_file():
        entry["has_json"] = True
        try:
            entry["metadata"] = _json.loads(md.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    elif (case / "metadata.json.dvc").is_file():
        entry["has_json"] = True

    return entry


def _read_flat_entries(layer_id: str, attempt_id: str, tier: str, d: Path) -> list[dict]:
    """Legacy fallback: read flat files `expected/<stem>.{wav,png,metadata.json}`
    (older attempts that pre-date the case-dir convention)."""
    import base64 as _b64
    import json as _json

    entries: dict[str, dict] = {}
    for f in sorted(d.iterdir()):
        if f.is_dir() or f.name in {".gitkeep", ".gitignore"}:
            continue
        stem = f.name
        for suffix in (".wav.dvc", ".png.dvc", ".metadata.json.dvc",
                       ".wav", ".png", ".metadata.json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        e = entries.setdefault(stem, {
            "stem":    stem,
            "cell":    None,
            "has_wav": False, "has_png": False, "has_json": False,
            "png_b64": None, "metadata": None, "wav_url": None,
        })
        n = f.name
        if n.endswith(".png"):
            e["has_png"] = True
            try:
                e["png_b64"] = _b64.b64encode(f.read_bytes()).decode("ascii")
            except OSError:
                pass
        elif n.endswith(".metadata.json"):
            e["has_json"] = True
            try:
                e["metadata"] = _json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        elif n.endswith(".wav"):
            e["has_wav"] = True
            e["wav_url"] = (
                f"/layers/{layer_id}/attempts/{attempt_id}/samples/{tier}/{stem}.wav"
            )
        elif n.endswith(".png.dvc"):
            e["has_png"] = True
        elif n.endswith(".wav.dvc"):
            e["has_wav"] = True
            if not e["wav_url"]:
                e["wav_url"] = (
                    f"/layers/{layer_id}/attempts/{attempt_id}/samples/{tier}/{stem}.wav"
                )
        elif n.endswith(".metadata.json.dvc"):
            e["has_json"] = True
    return list(entries.values())


def list_samples(layer_id: str, attempt_id: str) -> dict:
    """Enumerate samples present on disk under <attempt>/expected and
    <attempt>/showcase. Three layouts are supported:

      tier/<case>/{audio.wav, spectrogram.png, metadata.json}              ← canonical
      tier/<cell>/<case>/{audio.wav, spectrogram.png, metadata.json}       ← bank (uses_cells)
      tier/<stem>.{wav,png,metadata.json}                                  ← legacy flat

    PNG/JSON contents are inlined (small). `wav_url` is built to match the
    actual on-disk path so the frontend can play it without knowing the
    layout.
    """
    spec = get_attempt(layer_id, attempt_id)
    root = _samples_root(spec)
    seed = canonical_seed(spec)

    def _scan(tier: str) -> list[dict]:
        d = root / tier
        if not d.is_dir():
            return []
        entries: list[dict] = []
        has_dirs = False
        for child in sorted(d.iterdir()):
            if child.name in {".gitkeep", ".gitignore"}:
                continue
            if not child.is_dir():
                continue
            has_dirs = True
            if _is_case_dir(child):
                entries.append(
                    _read_case_dir(layer_id, attempt_id, tier, child, cell=None)
                )
            else:
                # Cell-grouped: walk one level deeper.
                for case in sorted(child.iterdir()):
                    if case.is_dir() and _is_case_dir(case):
                        entries.append(
                            _read_case_dir(layer_id, attempt_id, tier, case,
                                           cell=child.name)
                        )
        # If no case dirs were found, fall back to legacy flat layout.
        if not has_dirs and not entries:
            entries.extend(_read_flat_entries(layer_id, attempt_id, tier, d))
        return entries

    return {
        "attempt":        attempt_id,
        "layer":          layer_id,
        "canonical_seed": seed,
        "expected":       _scan("expected"),
        "showcase":       _scan("showcase"),
    }


def sample_wav_path(layer_id: str, attempt_id: str, tier: str, rel_path: str) -> Path:
    """Resolve a `samples/{tier}/{rel_path}` URL to a filesystem path. The
    rel_path is the remainder of the URL after the tier — anywhere from
    `<stem>.wav` (legacy flat) to `<cell>/<case>/audio.wav` (bank)."""
    if tier not in {"expected", "showcase"}:
        raise ValueError(f"illegal tier: {tier}")
    spec = get_attempt(layer_id, attempt_id)
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p not in {"", "."}]
    if any(p == ".." for p in parts):
        raise ValueError(f"illegal path: {rel_path!r}")
    return (_samples_root(spec) / tier).joinpath(*parts)


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
        state = mod.load(spec.artifact_root, dict(spec.params), extra=spec.extra_checkpoints)
        _state_cache[cache_key] = state
        return state


def warm(layer_id: str, attempt_id: str) -> None:
    """Eager-load and cache an attempt's handler state (no generation).

    This is the same lazy ``_get_state`` the first request would otherwise
    trigger inside the request path — calling it up front (see
    ``server.prewarm``) keeps the heavy cold load (e.g. the Layer A 16-adapter
    AudioLDM2 bank) out of the request, so the first user generate no longer
    blows past the backend ``AI_REQUEST_TIMEOUT_MS``.
    """
    spec = get_attempt(layer_id, attempt_id)
    _get_state(spec)


def prewarm_defaults(layers: set[str] | None = None) -> list[dict]:
    """Eager-load each layer's ``default`` attempt. Resilient: a failure for
    one attempt (un-installed dep, un-pulled checkpoint, …) is captured and
    skipped, never raised — one broken head must not stop the server booting.

    ``layers`` restricts which layer defaults are warmed (None => all).
    Returns one result row per warmed layer default, for the caller to log.
    """
    results: list[dict] = []
    for layer_id, layer_block in _registry_doc()["layers"].items():
        if layers is not None and layer_id not in layers:
            continue
        default = layer_block.get("default")
        if not default:
            continue
        row = {"layer": layer_id, "attempt": default, "ok": True, "error": None}
        try:
            warm(layer_id, default)
        except Exception as exc:  # noqa: BLE001 — boot must survive any handler failure
            row["ok"] = False
            row["error"] = str(exc)
        results.append(row)
    return results


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
    md.setdefault("attempt", _attempt_snapshot(spec))
    return result


def orchestrate_generation(
    *,
    seed: int | None,
    duration_s: float,
    season: str | None = None,
    diel: str | None = None,
    weather_type: str = "wind",
    intensity: str = "light",
    include_weather: bool = True,
    include_events: bool = True,
    layer_a_attempt: str | None = None,
    layer_b_attempt: str | None = None,
    layer_c_attempt: str | None = None,
    layer_d_attempt: str | None = None,
    include_stems: bool = False,
) -> dict:
    """Run generation layers A/B/C and hand their in-memory WAVs to Layer D."""

    duration_s = float(duration_s)
    if not 0.0 < duration_s <= 30.0:
        raise ValueError("duration_s must be greater than 0 and at most 30 seconds")
    attempts = {
        "layer_a": layer_a_attempt or default_attempt_id("layer_a"),
        "layer_b": layer_b_attempt or default_attempt_id("layer_b"),
        "layer_c": layer_c_attempt or default_attempt_id("layer_c"),
        "layer_d": layer_d_attempt or default_attempt_id("layer_d"),
    }

    layer_a = generate(
        "layer_a",
        attempts["layer_a"],
        seed=seed,
        season=season,
        diel=diel,
    )
    layer_b = (
        generate(
            "layer_b",
            attempts["layer_b"],
            seed=seed,
            weather_type=weather_type,
            intensity=intensity,
            duration_s=duration_s,
        )
        if include_weather
        else None
    )
    layer_c = (
        generate(
            "layer_c",
            attempts["layer_c"],
            seed=seed,
            season=season,
            diel=diel,
            duration_s=duration_s,
        )
        if include_events
        else None
    )
    final = generate(
        "layer_d",
        attempts["layer_d"],
        seed=None,
        ambient_wav_bytes=layer_a.get("wav_bytes"),
        weather_wav_bytes=layer_b.get("wav_bytes") if layer_b else None,
        event_wav_bytes=layer_c.get("wav_bytes") if layer_c else None,
        duration_s=duration_s,
    )
    final.setdefault("metadata", {})["orchestration"] = {
        "seed": seed,
        "duration_s": duration_s,
        "season": season,
        "diel": diel,
        "weather_type": weather_type if include_weather else None,
        "intensity": intensity if include_weather else None,
        "include_weather": include_weather,
        "include_events": include_events,
        "attempts": attempts,
        "parameter_routing": {
            "layer_a": ["seed", "season", "diel"],
            "layer_b": ["seed", "weather_type", "intensity", "duration_s"],
            "layer_c": ["seed", "season", "diel", "duration_s"],
            "layer_d": ["ambient_wav_bytes", "weather_wav_bytes", "event_wav_bytes", "duration_s"],
        },
        "upstream": {
            "layer_a": layer_a.get("metadata", {}),
            "layer_b": layer_b.get("metadata", {}) if layer_b else None,
            "layer_c": layer_c.get("metadata", {}) if layer_c else None,
        },
    }
    if include_stems:
        final["_stems"] = {
            "layer_a": layer_a,
            "layer_b": layer_b,
            "layer_c": layer_c,
        }
    return final


def orchestrate_analysis(
    audio_path: str,
    *,
    ambient_attempt: str | None = None,
    weather_attempt: str | None = None,
    events_attempt: str | None = None,
    aggregator_attempt: str | None = None,
    include_head_reports: bool = True,
) -> dict:
    """Run E-A/E-B/E-C analysis heads and fuse them through the aggregator."""

    attempts = {
        "ambient": ambient_attempt or default_layer_e_head_attempt("ambient"),
        "weather": weather_attempt or default_layer_e_head_attempt("weather"),
        "events": events_attempt or default_layer_e_head_attempt("events"),
        "aggregator": aggregator_attempt or default_layer_e_head_attempt("aggregator"),
    }

    ambient = analyze("layer_e", attempts["ambient"], audio_path)
    weather = analyze("layer_e", attempts["weather"], audio_path)
    events = analyze("layer_e", attempts["events"], audio_path)
    aggregator = aggregate_analysis_reports(
        attempts["aggregator"],
        ambient_report=ambient.get("report", {}),
        weather_report=weather.get("report", {}),
        events_report=events.get("report", {}),
    )
    model_lineage = {
        "ambient": ambient["attempt"],
        "weather": weather["attempt"],
        "events": events["attempt"],
        "aggregator": aggregator["attempt"],
    }
    aggregator["report"]["model_lineage"] = model_lineage

    result = {
        "report": aggregator["report"],
        "attempts": model_lineage,
    }
    if include_head_reports:
        result["head_reports"] = {
            "ambient": ambient.get("report", {}),
            "weather": weather.get("report", {}),
            "events": events.get("report", {}),
        }
    return result


def _attempt_snapshot(spec: AttemptSpec) -> dict:
    """Compact spec block attached to every dispatch result for traceability."""
    return {
        "layer":  spec.layer,
        "id":     spec.id,
        "label":  spec.label,
        "stage":  spec.stage,
        "head":   spec.head,
        "author": spec.author,
        "status": spec.status,
        "checkpoint": str(spec.checkpoint) if spec.checkpoint else None,
    }


def aggregate_analysis_reports(
    attempt_id: str,
    *,
    ambient_report: dict,
    weather_report: dict,
    events_report: dict,
) -> dict:
    """Dispatch report fusion to a Layer E aggregator attempt."""
    spec = get_attempt("layer_e", attempt_id)
    if spec.head != "aggregator":
        raise ValueError(f"{attempt_id} is not a Layer E aggregator attempt")
    mod = _get_handler_module(spec)
    if not hasattr(mod, "aggregate"):
        raise NotImplementedError(
            f"layer_e/{attempt_id} handler has no aggregate(); this attempt "
            f"does not support report fusion."
        )
    state = _get_state(spec)
    report = mod.aggregate(
        state,
        ambient_report=ambient_report,
        weather_report=weather_report,
        events_report=events_report,
    )
    return {"report": report, "attempt": _attempt_snapshot(spec)}


def analyze(layer_id: str, attempt_id: str, audio_path: str) -> dict:
    """Dispatch an upload-based analysis call to the attempt's handler.

    Analysis attempts (Layer E) are upload-based rather than seed-based: the
    handler exposes ``analyze(state, audio_path) -> dict`` returning a per-head
    report. Returns ``{"report": <handler dict>, "attempt": <spec snapshot>}``.
    """
    spec = get_attempt(layer_id, attempt_id)
    mod = _get_handler_module(spec)
    if not hasattr(mod, "analyze"):
        raise NotImplementedError(
            f"{layer_id}/{attempt_id} handler has no analyze(); this attempt "
            f"does not support upload analysis."
        )
    state = _get_state(spec)
    report = mod.analyze(state, audio_path)
    return {"report": report, "attempt": _attempt_snapshot(spec)}
