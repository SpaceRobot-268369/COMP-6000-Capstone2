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
import json
import secrets
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
_LAYER_D_MULTI_CLIP_ATTEMPT = "songke__mvp_2__multi_clip_mix"

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


def _resolve_asset_bank(value: Any) -> Path | None:
    return _resolve_checkpoint(value)


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

    # Adapter-bank layout: weights live under ckpt_dir/adapters/<profile>/.
    # Layer B wind intensity uses this for medium/heavy profiles, with light
    # derived from medium at runtime.
    adapter_root = ckpt_dir / "adapters"
    if adapter_root.is_dir():
        adapter_dirs = [p for p in sorted(adapter_root.iterdir()) if p.is_dir()]
        missing = [
            f"adapters/{p.name}/adapter_model.safetensors"
            for p in adapter_dirs
            if not (p / "adapter_model.safetensors").is_file()
        ]
        if adapter_dirs and not missing:
            return {"available": True, "reason": None, "missing": []}
        if missing:
            return {
                "available": False,
                "reason": (
                    f"{len(missing)}/{len(adapter_dirs)} adapter profiles not on disk. "
                    f"Run `dvc pull` under {ckpt_dir}."
                ),
                "missing": missing,
            }

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


def _asset_bank_availability(bank_dir: Path | None) -> dict:
    """Inspect a retrieval asset bank and report whether it is materialised."""
    if bank_dir is None:
        return {
            "available": False,
            "reason": "retrieval attempt missing registry-level asset_bank",
            "missing": [],
        }
    if not bank_dir.exists():
        return {
            "available": False,
            "reason": f"asset bank directory missing: {bank_dir}",
            "missing": [],
        }

    index_path = bank_dir / "index.json"
    media_dir = bank_dir / "media_asset_bank"
    missing = []
    if not index_path.is_file():
        missing.append("index.json")
    if not media_dir.is_dir():
        missing.append("media_asset_bank/")
    if missing:
        pull_hint = ""
        if (bank_dir / "media_asset_bank.dvc").is_file():
            pull_hint = f" Run `dvc pull {bank_dir / 'media_asset_bank.dvc'}`."
        return {
            "available": False,
            "reason": f"retrieval asset bank not materialized: {', '.join(missing)}.{pull_hint}",
            "missing": missing,
        }

    try:
        doc = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - user-facing registry health
        return {
            "available": False,
            "reason": f"failed to read asset bank index: {exc}",
            "missing": ["index.json"],
        }

    if not any(media_dir.rglob("*.wav")):
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
        return {
            "available": False,
            "reason": "retrieval asset bank not materialized: media_asset_bank audio.",
            "missing": ["media_asset_bank audio"],
        }

    missing_audio = [
        str(asset.get("audio_path", ""))
        for asset in doc.get("assets", [])
        if asset.get("audio_path") and not (bank_dir / str(asset["audio_path"])).is_file()
    ]
    if missing_audio:
        shown = ", ".join(missing_audio[:3])
        suffix = "..." if len(missing_audio) > 3 else ""
        return {
            "available": False,
            "reason": f"{len(missing_audio)} asset bank audio files missing: {shown}{suffix}",
            "missing": missing_audio,
        }

    return {"available": True, "reason": None, "missing": []}


def _asset_bank_species_options(bank_dir: Path | None) -> list[dict[str, str]]:
    """Return unique retrieval species from an asset bank index.

    Cached expected samples are intentionally tiny and should not define the UI
    species selector for a larger retrieval library.
    """
    if bank_dir is None:
        return []
    index_path = bank_dir / "index.json"
    if not index_path.is_file():
        return []
    try:
        doc = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - registry payload should degrade gently
        return []

    by_name: dict[str, dict[str, str]] = {}
    for asset in doc.get("assets", []):
        attrs = asset.get("attributes") or {}
        name = attrs.get("species_common_name")
        if not name:
            continue
        by_name[str(name)] = {
            "value": str(name),
            "label": str(name),
            "slug": str(attrs.get("event_type") or attrs.get("species_slug") or ""),
            "scientific_name": str(attrs.get("species_scientific_name") or ""),
        }
    return [by_name[name] for name in sorted(by_name)]


def list_layers() -> list[dict]:
    """Return the dropdown payload — one entry per layer, with attempt list."""
    out: list[dict] = []
    for layer_id, layer_block in _registry_doc()["layers"].items():
        attempts = []
        for att_id, att in layer_block["attempts"].items():
            kind = att.get("kind")
            ckpt = _resolve_checkpoint(att.get("checkpoint"))
            asset_bank = _resolve_asset_bank(att.get("asset_bank"))
            params = att.get("params") or {}
            cells = sorted((params.get("cells") or {}).keys())
            avail = (
                _asset_bank_availability(asset_bank)
                if kind == "retrieval"
                else _ckpt_availability(ckpt, cell_names=cells)
            )
            attempts.append({
                "id":      att_id,
                "label":   att.get("label", att_id),
                "stage":   att.get("stage", ""),
                "author":  att.get("author", ""),
                "head":    att.get("head"),
                "kind":    kind,
                "status":  att.get("status", ""),
                "description": att.get("description", ""),
                "checkpoint":       str(ckpt) if ckpt else None,
                "asset_bank":       str(asset_bank) if asset_bank else None,
                "available":        avail["available"],
                "unavailable_reason": avail["reason"],
                "missing_files":    avail["missing"],
                "uses_seed":        bool(att.get("uses_seed", False)),
                "uses_cells":       bool(att.get("uses_cells", False)),
                "uses_weather_controls": bool(att.get("uses_weather_controls", False)),
                "cells":            cells,
                "default_cell":     params.get("default_cell"),
                "species_options": (
                    _asset_bank_species_options(asset_bank)
                    if kind == "retrieval"
                    else []
                ),
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
        asset_bank=_resolve_asset_bank(att.get("asset_bank")),
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

    `wav_url` is returned only when the real WAV exists locally. A DVC pointer
    sets `wav_dvc` so clients can show a pull hint without rendering a broken
    audio element.
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
        "has_wav": False, "wav_dvc": False,
        "has_png": False, "has_json": False,
        "png_b64": None, "metadata": None, "wav_url": None,
    }

    audio_wav = case / "audio.wav"
    audio_dvc = case / "audio.wav.dvc"
    if audio_wav.is_file():
        entry["has_wav"] = True
        entry["wav_url"] = (
            f"/layers/{layer_id}/attempts/{attempt_id}/samples/{rel_wav}"
        )
    elif audio_dvc.is_file():
        entry["wav_dvc"] = True

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
            "has_wav": False, "wav_dvc": False,
            "has_png": False, "has_json": False,
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
            e["wav_dvc"] = True
        elif n.endswith(".metadata.json.dvc"):
            e["has_json"] = True
    return list(entries.values())


_ASSET_BANK_PREFIX = "__asset_bank__"


def _safe_asset_path(root: Path, rel_path: str) -> Path | None:
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p not in {"", "."}]
    if any(p == ".." for p in parts):
        return None
    full = (root.joinpath(*parts)).resolve()
    try:
        full.relative_to(root.resolve())
    except ValueError:
        return None
    return full


def _asset_bank_mel_path(audio_path: str) -> str:
    audio = Path(audio_path)
    if audio.name == "crop_bandpass.wav":
        return str(audio.with_name("mel_bandpass.png"))
    if audio.suffix == ".wav":
        return str(audio.with_suffix(".png"))
    return str(audio.parent / "mel_bandpass.png")


def _read_asset_bank_expected(spec: AttemptSpec) -> list[dict]:
    """Expose one retrieval reference per species from an attempt asset bank.

    Retrieval attempts keep their full DVC-backed references under
    ``model/candidates/.../media_asset_bank`` rather than under the lightweight
    ``acoustic_ai/layers/.../expected`` preview directory.
    """
    import base64 as _b64

    bank = spec.asset_bank
    if bank is None:
        return []
    index = bank / "index.json"
    if not index.is_file():
        return []
    try:
        doc = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    by_species: dict[str, tuple[dict, dict, str, str]] = {}
    for asset in doc.get("assets", []):
        attrs = asset.get("attributes") or {}
        slug = attrs.get("event_type") or attrs.get("species_slug")
        common = attrs.get("species_common_name")
        audio_path = asset.get("audio_path")
        if not slug or not common or not audio_path or slug in by_species:
            continue
        by_species[str(slug)] = (asset, attrs, str(slug), str(common))

    entries: list[dict] = []
    for _, (asset, attrs, slug, common) in sorted(
        by_species.items(),
        key=lambda item: item[1][3],
    ):
        audio_path = str(asset["audio_path"])
        audio_full = _safe_asset_path(bank, audio_path)
        png_rel = _asset_bank_mel_path(audio_path)
        png_full = _safe_asset_path(bank, png_rel)
        metadata_rel = str(Path(audio_path).parent / "metadata.json")
        metadata_full = _safe_asset_path(bank, metadata_rel)
        stem = str(asset.get("id") or f"{slug}_reference")
        rel_wav = "/".join([_ASSET_BANK_PREFIX] + [
            p for p in audio_path.replace("\\", "/").split("/") if p
        ])

        metadata = {
            **attrs,
            "species_common_name": common,
            "species_scientific_name": attrs.get("species_scientific_name") or "",
            "species_slug": slug,
            "expected_source": "retrieval_asset_bank",
            "expected_item": asset.get("id") or stem,
            "display_title": f"{common} · retrieval reference",
            "display_audio": Path(audio_path).name,
            "display_spectrogram": Path(png_rel).name,
            "asset_bank_audio_path": audio_path,
        }
        if metadata_full and metadata_full.is_file():
            try:
                metadata.update(json.loads(metadata_full.read_text(encoding="utf-8")))
                metadata["species_common_name"] = common
                metadata["species_scientific_name"] = (
                    attrs.get("species_scientific_name") or ""
                )
                metadata["species_slug"] = slug
            except (OSError, ValueError):
                pass

        entry = {
            "stem": stem,
            "cell": None,
            "has_wav": bool(audio_full and audio_full.is_file()),
            "wav_dvc": not bool(audio_full and audio_full.is_file()),
            "has_png": bool((png_full and png_full.is_file()) or png_rel),
            "has_json": True,
            "png_b64": None,
            "metadata": metadata,
            "wav_url": None,
        }
        if entry["has_wav"]:
            entry["wav_url"] = (
                f"/layers/{spec.layer}/attempts/{spec.id}/samples/expected/{rel_wav}"
            )
        if png_full and png_full.is_file():
            try:
                entry["png_b64"] = _b64.b64encode(png_full.read_bytes()).decode("ascii")
            except OSError:
                pass
        entries.append(entry)
    return entries


def list_samples(layer_id: str, attempt_id: str) -> dict:
    """Enumerate samples present on disk under <attempt>/expected and
    <attempt>/showcase. Three layouts are supported:

      tier/<case>/{audio.wav, spectrogram.png, metadata.json}              ← canonical
      tier/<cell>/<case>/{audio.wav, spectrogram.png, metadata.json}       ← bank (uses_cells)
      tier/<stem>.{wav,png,metadata.json}                                  ← legacy flat

    PNG/JSON contents are inlined (small). `wav_url` is returned only when the
    actual WAV is materialised locally; otherwise `wav_dvc` marks a pullable
    DVC pointer.
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

    expected = _scan("expected")
    asset_bank_expected = _read_asset_bank_expected(spec)
    if asset_bank_expected:
        expected = asset_bank_expected

    return {
        "attempt":        attempt_id,
        "layer":          layer_id,
        "canonical_seed": seed,
        "expected":       expected,
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
    if tier == "expected" and parts and parts[0] == _ASSET_BANK_PREFIX:
        if spec.asset_bank is None:
            raise ValueError(f"attempt has no asset bank: {attempt_id}")
        path = _safe_asset_path(spec.asset_bank, "/".join(parts[1:]))
        if path is None:
            raise ValueError(f"illegal path: {rel_path!r}")
        return path
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

    # No seed supplied -> draw a fresh random one so each run differs. The
    # resolved value is shared across A/B/C/D and echoed in metadata so the
    # user can pin it to reproduce a run.
    if seed is None:
        seed = secrets.randbelow(2_147_483_648)  # [0, 2147483647]
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
    layer_d_params = _layer_d_generation_params(
        layer_d_attempt=attempts["layer_d"],
        layer_a=layer_a,
        layer_b=layer_b,
        layer_c=layer_c,
        duration_s=duration_s,
        placement_seed=seed,
    )
    final = generate(
        "layer_d",
        attempts["layer_d"],
        seed=None,
        **layer_d_params,
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
            "layer_d": list(layer_d_params.keys()),
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


def _layer_d_generation_params(
    *,
    layer_d_attempt: str,
    layer_a: dict,
    layer_b: dict | None,
    layer_c: dict | None,
    duration_s: float,
    placement_seed: int | None,
) -> dict[str, Any]:
    if layer_d_attempt != _LAYER_D_MULTI_CLIP_ATTEMPT:
        return {
            "ambient_wav_bytes": layer_a.get("wav_bytes"),
            "weather_wav_bytes": layer_b.get("wav_bytes") if layer_b else None,
            "event_wav_bytes": layer_c.get("wav_bytes") if layer_c else None,
            "duration_s": duration_s,
        }
    return {
        "ambient_wav_bytes": layer_a.get("wav_bytes"),
        "weather_clips": _weather_clips_for_layer_d(layer_b),
        "event_clips": _event_clips_for_layer_d(layer_c),
        "duration_s": duration_s,
        "placement_seed": placement_seed,
    }


def _weather_clips_for_layer_d(layer_b: dict | None) -> list[dict[str, Any]]:
    if not layer_b:
        return []
    metadata = layer_b.get("metadata") or {}
    existing = metadata.get("weather_clips") or layer_b.get("weather_clips")
    if isinstance(existing, list):
        return [_normalise_clip_wav_key(clip) for clip in existing if isinstance(clip, dict)]
    wav = layer_b.get("wav_bytes")
    if wav is None:
        return []
    layer_b_md = metadata.get("layer_b") or {}
    requested = layer_b_md.get("requested") or {}
    selected = layer_b_md.get("selected") or {}
    weather_type = requested.get("weather_type") or selected.get("primary_weather") or "weather"
    role = str(selected.get("layer_d_role") or "").lower()
    continuous = role != "discrete"
    return [
        {
            "wav": wav,
            "weather_type": weather_type,
            "continuous": continuous,
            "onsets_s": None if not continuous else None,
            "gain_db": None,
            "change": None,
        }
    ]


def _event_clips_for_layer_d(layer_c: dict | None) -> list[dict[str, Any]]:
    if not layer_c:
        return []
    metadata = layer_c.get("metadata") or {}
    existing = metadata.get("event_clips") or layer_c.get("event_clips")
    if isinstance(existing, list):
        return [_normalise_clip_wav_key(clip) for clip in existing if isinstance(clip, dict)]
    wav = layer_c.get("wav_bytes")
    if wav is None:
        return []
    return [
        {
            "wav": wav,
            "species": metadata.get("species") or "layer_c_events",
            "onsets_s": [0.0],
            "gain_db": None,
        }
    ]


def _normalise_clip_wav_key(clip: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(clip)
    if "wav" not in normalised and "wav_bytes" in normalised:
        normalised["wav"] = normalised["wav_bytes"]
    return normalised


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
        "asset_bank": str(spec.asset_bank) if spec.asset_bank else None,
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
