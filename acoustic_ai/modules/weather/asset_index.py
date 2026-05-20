"""Module B weather asset retrieval.

The weather layer is retrieval-based: curated wind/rain clips live under
``acoustic_ai/data/weather/weather_assets/`` and ``asset_index.csv`` records
their layer, intensity, source, and license metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

WEATHER_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "weather"
DEFAULT_ASSET_INDEX = WEATHER_DATA_DIR / "asset_index.csv"
DEFAULT_ASSET_ROOT = WEATHER_DATA_DIR / "weather_assets"

VALID_LAYERS = {"wind", "rain"}
VALID_INTENSITIES = {
    "wind": {"light", "moderate", "strong"},
    "rain": {"light", "moderate", "heavy"},
}
REQUIRED_COLUMNS = {
    "asset_id",
    "clip_path",
    "layer",
    "intensity",
    "source",
    "license",
}


@dataclass(frozen=True)
class WeatherAsset:
    """A selected weather asset plus audit metadata from the index."""

    asset_id: str
    path: Path
    layer: str
    intensity: str
    source: str = ""
    license: str = ""
    attribution: str = ""
    notes: str = ""


def load_asset_index(index_path: Path | str = DEFAULT_ASSET_INDEX) -> pd.DataFrame:
    """Load and validate the weather asset index.

    Empty indexes are valid while the asset library is being curated. Missing
    columns are not valid because they make later DVC/license audit ambiguous.
    """

    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Weather asset index not found: {path}")

    df = pd.read_csv(path).fillna("")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Weather asset index missing columns: {missing_cols}")

    if df.empty:
        return df

    df["layer"] = df["layer"].astype(str).str.strip().str.lower()
    df["intensity"] = df["intensity"].astype(str).str.strip().str.lower()

    invalid_layers = sorted(set(df["layer"]) - VALID_LAYERS)
    if invalid_layers:
        raise ValueError(f"Invalid weather layers in asset index: {invalid_layers}")

    invalid_rows = []
    for row in df.itertuples(index=False):
        if row.intensity not in VALID_INTENSITIES[row.layer]:
            invalid_rows.append(f"{row.asset_id}:{row.layer}/{row.intensity}")
    if invalid_rows:
        raise ValueError(f"Invalid weather intensities: {invalid_rows}")

    return df


def select_asset(
    layer: str,
    intensity: str,
    *,
    seed: Optional[int] = None,
    index_path: Path | str = DEFAULT_ASSET_INDEX,
    asset_root: Path | str = DEFAULT_ASSET_ROOT,
) -> Optional[WeatherAsset]:
    """Select one matching asset deterministically.

    Returns ``None`` when the requested bucket is valid but no curated asset is
    available yet. The mixer treats that as silence and marks metadata as
    ``missing_asset`` rather than failing the whole generation request.
    """

    layer = layer.strip().lower()
    intensity = intensity.strip().lower()
    _validate_request(layer, intensity)

    df = load_asset_index(index_path)
    matches = df[(df["layer"] == layer) & (df["intensity"] == intensity)]
    if matches.empty:
        return None

    matches = matches.sort_values("asset_id").reset_index(drop=True)
    selected = matches.iloc[_stable_index(layer, intensity, seed, len(matches))]
    path = _resolve_clip_path(selected["clip_path"], Path(asset_root))

    return WeatherAsset(
        asset_id=str(selected["asset_id"]),
        path=path,
        layer=layer,
        intensity=intensity,
        source=str(selected.get("source", "")),
        license=str(selected.get("license", "")),
        attribution=str(selected.get("attribution", "")),
        notes=str(selected.get("notes", "")),
    )


def available_assets(
    *,
    index_path: Path | str = DEFAULT_ASSET_INDEX,
    asset_root: Path | str = DEFAULT_ASSET_ROOT,
) -> list[WeatherAsset]:
    """Return all indexed weather assets."""

    df = load_asset_index(index_path)
    assets: list[WeatherAsset] = []
    for row in df.itertuples(index=False):
        assets.append(
            WeatherAsset(
                asset_id=str(row.asset_id),
                path=_resolve_clip_path(str(row.clip_path), Path(asset_root)),
                layer=str(row.layer),
                intensity=str(row.intensity),
                source=str(getattr(row, "source", "")),
                license=str(getattr(row, "license", "")),
                attribution=str(getattr(row, "attribution", "")),
                notes=str(getattr(row, "notes", "")),
            )
        )
    return assets


def _validate_request(layer: str, intensity: str) -> None:
    if layer not in VALID_LAYERS:
        raise ValueError(f"Unknown weather layer: {layer}")
    if intensity not in VALID_INTENSITIES[layer]:
        valid = ", ".join(sorted(VALID_INTENSITIES[layer]))
        raise ValueError(f"Invalid {layer} intensity '{intensity}'. Expected one of: {valid}")


def _stable_index(layer: str, intensity: str, seed: Optional[int], n: int) -> int:
    key = f"{layer}:{intensity}:{seed if seed is not None else 0}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % n


def _resolve_clip_path(clip_path: str, asset_root: Path) -> Path:
    path = Path(clip_path)
    if path.is_absolute():
        return path
    return asset_root / path
