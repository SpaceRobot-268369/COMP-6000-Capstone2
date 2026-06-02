"""Module E-B — weather intensity detector.

Smoke-test detector for audible wind/rain in a raw ecoacoustic mixture.

The MVP design in ``pipeline_design.md`` points E-B toward PANNs/CLAP as the
primary detector. For the first smoke test we keep the dependency footprint
small: extract explainable spectral features, optionally calibrate them against
Layer B's labelled weather asset index, and return the report shape the
frontend already expects.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_DURATION_S = 30.0

COMPONENTS = ("rain", "wind", "thunder")
CANONICAL = {
    "": "none",
    "no": "none",
    "false": "none",
    "none": "none",
    "medium": "moderate",
    "moderate": "moderate",
    "heavy": "heavy",
    "strong": "strong",
    "light": "light",
    "unclear": "unclear",
}


@dataclass(frozen=True)
class WeatherAsset:
    asset_id: str
    audio_path: Path
    labels: dict[str, str]
    source: str
    metadata: dict[str, str] | None = None


def analyse_weather(
    audio_path: str | Path,
    calibration_assets: Iterable[WeatherAsset] | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
) -> dict:
    """Analyse one audio file and return an E-B weather report."""
    path = Path(audio_path)
    features = extract_weather_features(path, sample_rate=sample_rate, duration_s=duration_s)
    assets = list(calibration_assets or [])

    if assets:
        component_results = _predict_with_calibration(features, assets)
        if any("site257_clap_promoted" in asset.source for asset in assets):
            method = "site257_clap_promoted_calibrated_spectral_nearest_centroid"
        else:
            method = "calibrated_spectral_nearest_centroid"
    else:
        component_results = _predict_with_heuristics(features)
        method = "spectral_heuristic"

    rain = component_results["rain"]
    wind = component_results["wind"]
    thunder = component_results.get("thunder", {"intensity": "none", "confidence": 0.0})
    confidence = float(np.mean([rain["confidence"], wind["confidence"]]))

    return {
        "component": "E-B",
        "method": method,
        "wind_intensity": wind["intensity"],
        "rain_intensity": rain["intensity"],
        "thunder_intensity": thunder["intensity"],
        "confidence": round(confidence, 3),
        "component_confidence": {
            "wind": round(wind["confidence"], 3),
            "rain": round(rain["confidence"], 3),
            "thunder": round(thunder["confidence"], 3),
        },
        "features": features,
        "limitations": [
            "Smoke-test baseline: spectral features plus labelled Layer B/site-weather assets.",
            "When using site257_clap_promoted assets, labels come from Murphy's Server A CLAP-first candidate policy.",
            "The live detector is still spectral calibration, not a fresh CLAP prompt pass on each uploaded file.",
            "Does not separate weather stems; it scores the raw mixture directly.",
        ],
    }


def extract_weather_features(
    audio_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
) -> dict[str, float]:
    """Compute compact, explainable features for rain/wind heuristics."""
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration_s)
    if y.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {audio_path}")

    y = np.asarray(y, dtype=np.float32)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total_energy = np.maximum(stft.sum(axis=0), 1e-12)

    def band_ratio(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.mean(stft[mask].sum(axis=0) / total_energy))

    centroid = librosa.feature.spectral_centroid(S=np.sqrt(stft), sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=np.sqrt(stft))[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]

    peak = float(np.max(np.abs(y)))
    rms_mean = float(np.mean(rms))
    rms_dbfs = _amp_to_db(rms_mean)

    return {
        "duration_s": round(float(y.size / sr), 3),
        "sample_rate": float(sr),
        "rms_dbfs": round(rms_dbfs, 3),
        "peak_dbfs": round(_amp_to_db(peak), 3),
        "low_band_ratio_20_250": round(band_ratio(20, 250), 6),
        "low_mid_band_ratio_250_1000": round(band_ratio(250, 1000), 6),
        "rain_band_ratio_2000_8000": round(band_ratio(2000, 8000), 6),
        "high_band_ratio_4000_10000": round(band_ratio(4000, 10000), 6),
        "spectral_centroid_hz": round(float(np.mean(centroid)), 3),
        "spectral_flatness": round(float(np.mean(flatness)), 6),
        "onset_strength_mean": round(float(np.mean(onset)), 6),
        "onset_strength_p95": round(float(np.percentile(onset, 95)), 6),
        "zero_crossing_rate": round(float(np.mean(zcr)), 6),
        "rms_modulation": round(float(np.std(rms) / (np.mean(rms) + 1e-9)), 6),
    }


def load_weather_assets_from_index(index_path: str | Path) -> list[WeatherAsset]:
    """Load labelled Layer B assets from a Murphy/new-schema asset index."""
    index = Path(index_path)
    if not index.exists():
        raise FileNotFoundError(f"Weather asset index not found: {index}")

    with index.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assets: list[WeatherAsset] = []
    for row in rows:
        rel = row.get("clip_path", "").strip()
        if not rel:
            continue
        audio_path = _resolve_index_audio_path(index, row)
        labels = {
            "rain": _canonical_intensity(row.get("rain_intensity")),
            "wind": _canonical_wind(row.get("wind_intensity")),
            "thunder": _canonical_intensity(row.get("thunder_intensity")),
        }
        assets.append(
            WeatherAsset(
                asset_id=row.get("asset_id") or audio_path.stem,
                audio_path=audio_path,
                labels=labels,
                source=str(index),
                metadata=row,
            )
        )
    return assets


def load_site_promoted_weather_assets(manifest_path: str | Path) -> list[WeatherAsset]:
    """Load Murphy's Server A CLAP-promoted site 257 weather assets.

    The promoted manifest is produced by
    ``script/dataset/promote_site_weather_candidates.py`` on Server A. It is not
    the same shape as the shared Layer B asset index: rows point at 22.05 kHz
    mono Layer-D-ready WAV files and carry CLAP/weather pool labels.
    """
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Site weather promoted manifest not found: {manifest}")

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assets: list[WeatherAsset] = []
    for row in rows:
        if (row.get("layer_d_ready") or "").strip().lower() not in {"true", "yes", "1"}:
            continue
        audio_path = _resolve_site_promoted_audio_path(manifest, row)
        labels = _labels_from_site_promoted_row(row)
        assets.append(
            WeatherAsset(
                asset_id=row.get("clip_id") or audio_path.stem,
                audio_path=audio_path,
                labels=labels,
                source=str(manifest),
                metadata=row,
            )
        )
    return assets


def _resolve_site_promoted_audio_path(manifest_path: Path, row: dict[str, str]) -> Path:
    rel = (row.get("layer_d_asset_path") or row.get("wav_path") or "").strip()
    if rel:
        candidate = Path(rel)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        # Server manifests keep paths rooted at the server run directory, e.g.
        # runs/mvp_pool_.../assets_wav_22050_mono/...
        parts = candidate.parts
        if "assets_wav_22050_mono" in parts:
            idx = parts.index("assets_wav_22050_mono")
            local = manifest_path.parent.joinpath(*parts[idx:])
            if local.exists():
                return local

        local = manifest_path.parent / candidate
        if local.exists():
            return local

    clip_id = (row.get("clip_id") or "").strip()
    matches = list(manifest_path.parent.glob(f"assets_wav_22050_mono/**/*{clip_id}*.wav"))
    if matches:
        return matches[0]

    return manifest_path.parent / "assets_wav_22050_mono" / Path(rel).name


def _labels_from_site_promoted_row(row: dict[str, str]) -> dict[str, str]:
    labels = {"rain": "none", "wind": "none", "thunder": "none"}
    pool_label = (row.get("pool_label") or row.get("clap_weather_label") or "").strip().lower()
    hint = (row.get("pool_intensity_hint") or "").strip().lower()
    intensity = _canonical_wind(hint) if hint else "moderate"

    if "rain" in pool_label:
        labels["rain"] = "moderate" if hint == "mixed" else _canonical_intensity(hint or "moderate")
    if "wind" in pool_label:
        labels["wind"] = "moderate" if hint == "mixed" else _canonical_wind(hint or "moderate")

    # Murphy's current policy explicitly keeps site thunder out of the default
    # MVP promoted pool; thunder remains library fallback unless separately
    # validated.
    return labels


def _resolve_index_audio_path(index_path: Path, row: dict[str, str]) -> Path:
    """Resolve Layer B asset paths across old and attempt-local layouts."""
    rel = row.get("clip_path", "").strip()
    audio_path = Path(rel)
    if not audio_path.is_absolute():
        audio_path = PROJECT_ROOT / audio_path
    if audio_path.exists():
        return audio_path

    # Murphy's current index keeps historical repo-relative paths such as
    # acoustic_ai/data/weather/rain/foo.wav, while the DVC outputs materialise
    # beside this index as data/weather/{rain,wind,thunder}/foo.wav.
    filename = Path(rel).name
    parent = index_path.parent
    for component in _candidate_components(row):
        candidate = parent / component / filename
        if candidate.exists():
            return candidate

    # Return the first attempt-local candidate even before `dvc pull`; this
    # lets smoke-test error messages point at the path that should materialise.
    candidates = _candidate_components(row)
    if candidates:
        return parent / candidates[0] / filename
    return audio_path


def _candidate_components(row: dict[str, str]) -> list[str]:
    components = []
    primary = (row.get("primary_weather") or "").strip().lower()
    if primary in {"rain", "wind", "thunder"}:
        components.append(primary)
    for component in COMPONENTS:
        if _truthy_or_unclear(row.get(f"has_{component}", "")) and component not in components:
            components.append(component)
    return components


def discover_legacy_weather_assets(
    root: str | Path = PROJECT_ROOT / "acoustic_ai" / "data" / "weather" / "weather_assets",
) -> list[WeatherAsset]:
    """Discover the pre-restructure Layer B assets kept in older branches."""
    base = Path(root)
    assets: list[WeatherAsset] = []
    if not base.exists():
        return assets

    for wav in sorted(base.rglob("*.wav")):
        parts = wav.relative_to(base).parts
        if len(parts) < 3:
            continue
        component, intensity = parts[0], parts[1]
        if component not in {"rain", "wind"}:
            continue
        labels = {"rain": "none", "wind": "none", "thunder": "none"}
        labels[component] = _canonical_wind(intensity) if component == "wind" else _canonical_intensity(intensity)
        assets.append(
            WeatherAsset(
                asset_id=wav.stem,
                audio_path=wav,
                labels=labels,
                source=str(base),
                metadata=None,
            )
        )
    return assets


def _predict_with_calibration(features: dict[str, float], assets: list[WeatherAsset]) -> dict:
    feature_cache = []
    for asset in assets:
        if not asset.audio_path.exists():
            continue
        try:
            feature_cache.append((asset, extract_weather_features(asset.audio_path)))
        except Exception:
            continue

    if not feature_cache:
        return _predict_with_heuristics(features)

    results = {}
    for component in COMPONENTS:
        labelled = [
            (asset, feat) for asset, feat in feature_cache
            if asset.labels.get(component, "unclear") != "unclear"
        ]
        results[component] = _nearest_centroid(component, features, labelled)
    return results


def _nearest_centroid(
    component: str,
    features: dict[str, float],
    labelled: list[tuple[WeatherAsset, dict[str, float]]],
) -> dict:
    if not labelled:
        return {"intensity": "none", "confidence": 0.0}

    keys = _feature_keys_for_component(component)
    matrix = np.array([[float(feat[k]) for k in keys] for _, feat in labelled], dtype=float)
    query = np.array([float(features[k]) for k in keys], dtype=float)
    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0) + 1e-6
    matrix = (matrix - mu) / sigma
    query = (query - mu) / sigma

    by_label: dict[str, list[np.ndarray]] = {}
    for (asset, _feat), row in zip(labelled, matrix):
        by_label.setdefault(asset.labels.get(component, "none"), []).append(row)

    distances = {}
    for label, rows in by_label.items():
        centroid = np.vstack(rows).mean(axis=0)
        distances[label] = float(np.linalg.norm(query - centroid))

    label, best = min(distances.items(), key=lambda item: item[1])
    ordered = sorted(distances.values())
    margin = ordered[1] - ordered[0] if len(ordered) > 1 else 1.0
    confidence = 1.0 / (1.0 + math.exp(-(margin + 0.35)))
    if best > 5.0:
        confidence *= 0.7
    return {"intensity": label, "confidence": float(np.clip(confidence, 0.05, 0.95))}


def _predict_with_heuristics(features: dict[str, float]) -> dict:
    rain_score = (
        1.4 * features["rain_band_ratio_2000_8000"]
        + 0.8 * features["high_band_ratio_4000_10000"]
        + 1.2 * features["spectral_flatness"]
        + 0.06 * features["onset_strength_mean"]
    )
    wind_score = (
        1.6 * features["low_band_ratio_20_250"]
        + 0.8 * features["low_mid_band_ratio_250_1000"]
        + 0.35 * features["rms_modulation"]
        - 0.25 * features["high_band_ratio_4000_10000"]
    )
    thunder_score = (
        1.8 * features["low_band_ratio_20_250"]
        + 0.1 * features["onset_strength_p95"]
        + 0.2 * features["rms_modulation"]
    )
    return {
        "rain": _score_to_result(rain_score, ["none", "light", "moderate", "heavy"], [0.28, 0.48, 0.72]),
        "wind": _score_to_result(wind_score, ["none", "light", "moderate", "strong"], [0.22, 0.40, 0.62]),
        "thunder": _score_to_result(thunder_score, ["none", "moderate", "heavy"], [0.45, 0.75]),
    }


def _score_to_result(score: float, labels: list[str], thresholds: list[float]) -> dict:
    idx = 0
    for threshold in thresholds:
        if score >= threshold:
            idx += 1
    nearest = min([abs(score - t) for t in thresholds] or [1.0])
    confidence = float(np.clip(0.55 + nearest, 0.2, 0.9))
    return {"intensity": labels[idx], "confidence": confidence}


def _feature_keys_for_component(component: str) -> list[str]:
    if component == "rain":
        return [
            "rain_band_ratio_2000_8000",
            "high_band_ratio_4000_10000",
            "spectral_flatness",
            "onset_strength_mean",
            "zero_crossing_rate",
            "rms_dbfs",
        ]
    if component == "wind":
        return [
            "low_band_ratio_20_250",
            "low_mid_band_ratio_250_1000",
            "high_band_ratio_4000_10000",
            "spectral_centroid_hz",
            "rms_modulation",
            "rms_dbfs",
        ]
    return [
        "low_band_ratio_20_250",
        "onset_strength_p95",
        "rms_modulation",
        "rms_dbfs",
    ]


def _canonical_intensity(value: str | None) -> str:
    return CANONICAL.get((value or "").strip().lower(), "unclear")


def _canonical_wind(value: str | None) -> str:
    label = _canonical_intensity(value)
    return "strong" if label == "heavy" else label


def _truthy_or_unclear(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "yes", "1", "unclear"}


def _amp_to_db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))
