"""Module E-B MVP-1 weather detector.

PANNs CNN14 is the target primary detector from ``pipeline_design.md``. The
local demo environment may not yet have torch/PANNs installed, so this module
keeps the smoke baseline as a stable fallback while exposing explicit PANNs
availability and evidence fields in the report.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

from layers.layer_e.attempts.liting__smoke_1__e_b_weather_analysis.code import weather_detector as smoke


WeatherAsset = smoke.WeatherAsset
extract_weather_features = smoke.extract_weather_features
load_site_promoted_weather_assets = smoke.load_site_promoted_weather_assets
load_weather_assets_from_index = smoke.load_weather_assets_from_index
discover_legacy_weather_assets = smoke.discover_legacy_weather_assets

DEFAULT_SAMPLE_RATE = 32_000
FALLBACK_SAMPLE_RATE = smoke.DEFAULT_SAMPLE_RATE
DEFAULT_DURATION_S = smoke.DEFAULT_DURATION_S

PANN_LABEL_GROUPS = {
    "rain": ("Rain", "Raindrop", "Rain on surface"),
    "wind": ("Wind", "Rustling leaves", "Wind noise"),
    "thunder": ("Thunderstorm", "Thunder"),
}
_PANNS_CACHE: dict[str, object] | None = None


@dataclass(frozen=True)
class PannsEvidence:
    available: bool
    reason: str
    scores: dict[str, float]
    labels: dict[str, dict[str, float]]


def analyse_weather(
    audio_path: str | Path,
    calibration_assets: Iterable[WeatherAsset] | None = None,
    sample_rate: int = FALLBACK_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
) -> dict:
    """Analyse one audio file and return an E-B MVP weather report."""
    path = Path(audio_path)
    spectral = smoke.analyse_weather(
        path,
        calibration_assets=calibration_assets,
        sample_rate=sample_rate,
        duration_s=duration_s,
    )
    panns = score_with_panns(path, duration_s=duration_s)

    if panns.available:
        weather = _fuse_panns_with_spectral(spectral, panns)
        method = "panns_cnn14_audioset__site257_spectral_support"
    else:
        weather = dict(spectral)
        method = f"panns_unavailable__{spectral['method']}"

    weather["method"] = method
    weather["stage"] = "mvp_1"
    weather["primary_model"] = "PANNs CNN14 AudioSet tagger"
    weather["panns_available"] = panns.available
    weather["panns_status"] = panns.reason
    weather["panns_evidence"] = {
        "component_scores": {k: round(v, 4) for k, v in panns.scores.items()},
        "matched_labels": {
            component: {label: round(score, 4) for label, score in labels.items()}
            for component, labels in panns.labels.items()
        },
    }
    weather["supporting_detector"] = {
        "method": spectral["method"],
        "wind_intensity": spectral["wind_intensity"],
        "rain_intensity": spectral["rain_intensity"],
        "thunder_intensity": spectral["thunder_intensity"],
        "confidence": spectral["confidence"],
    }
    weather["limitations"] = [
        "MVP-1 path: PANNs CNN14 primary when dependencies/weights are available.",
        "Current local environment may fall back to the site257 calibrated spectral detector.",
        "PANNs is a general AudioSet model; site-specific calibration is still required.",
        "No source separation is performed; E-B scores the raw uploaded mixture.",
    ]
    return weather


def score_with_panns(audio_path: str | Path, duration_s: float = DEFAULT_DURATION_S) -> PannsEvidence:
    """Return PANNs weather evidence if the optional dependency is available."""
    try:
        panns_runtime = _get_panns_runtime()
    except Exception as exc:
        return _unavailable(str(exc))

    try:
        audio, _sr = librosa.load(audio_path, sr=DEFAULT_SAMPLE_RATE, mono=True, duration=duration_s)
        if audio.size == 0:
            return _unavailable("audio file is empty or unreadable")

        tagger = panns_runtime["tagger"]
        clipwise_output, _embedding = tagger.inference(audio[None, :])
        probs = np.asarray(clipwise_output)[0]
    except Exception as exc:
        return _unavailable(f"PANNs inference failed: {exc}")

    labels = panns_runtime["labels"]
    matched: dict[str, dict[str, float]] = {}
    component_scores: dict[str, float] = {}
    for component, names in PANN_LABEL_GROUPS.items():
        scores = {}
        for name in names:
            idx = _find_label_index(labels, name)
            if idx is not None and idx < probs.size:
                scores[name] = float(probs[idx])
        matched[component] = scores
        component_scores[component] = max(scores.values()) if scores else 0.0

    return PannsEvidence(
        available=True,
        reason="PANNs inference completed",
        scores=component_scores,
        labels=matched,
    )


def _get_panns_runtime() -> dict[str, object]:
    global _PANNS_CACHE
    if _PANNS_CACHE is not None:
        return _PANNS_CACHE

    panns_home = Path(os.environ.get("PANNS_HOME", "/private/tmp/panns_home"))
    panns_home.mkdir(parents=True, exist_ok=True)
    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(panns_home)
    try:
        try:
            panns_mod = importlib.import_module("panns_inference")
        except Exception as exc:
            raise RuntimeError(f"panns_inference is not available: {exc}") from exc

        labels = _load_panns_labels(panns_mod)
        checkpoint_path = panns_home / "panns_data" / "Cnn14_mAP=0.431.pth"
        # Zenodo's Cnn14_mAP=0.431 checkpoint is 327,428,481 bytes. Keep
        # the guard below the official size so partial HTML/error downloads
        # are rejected without misclassifying the valid checkpoint.
        if not checkpoint_path.exists() or checkpoint_path.stat().st_size < 300_000_000:
            raise RuntimeError(
                "PANNs checkpoint is not materialised or is incomplete; "
                f"expected {checkpoint_path}"
            )
        tagger_cls = getattr(panns_mod, "AudioTagging")
        tagger = tagger_cls(checkpoint_path=str(checkpoint_path), device="cpu")
    finally:
        _restore_home(old_home)

    _PANNS_CACHE = {"labels": labels, "tagger": tagger}
    return _PANNS_CACHE


def _fuse_panns_with_spectral(spectral: dict, panns: PannsEvidence) -> dict:
    result = dict(spectral)
    result["rain_intensity"] = _panns_score_to_intensity(
        panns.scores.get("rain", 0.0),
        ["none", "light", "moderate", "heavy"],
        fallback=spectral["rain_intensity"],
    )
    result["wind_intensity"] = _panns_score_to_intensity(
        panns.scores.get("wind", 0.0),
        ["none", "light", "moderate", "strong"],
        fallback=spectral["wind_intensity"],
    )
    result["thunder_intensity"] = _panns_score_to_intensity(
        panns.scores.get("thunder", 0.0),
        ["none", "moderate", "heavy"],
        fallback=spectral["thunder_intensity"],
    )

    component_confidence = dict(spectral.get("component_confidence", {}))
    for component in ("rain", "wind", "thunder"):
        panns_conf = float(np.clip(0.35 + panns.scores.get(component, 0.0), 0.05, 0.95))
        component_confidence[component] = round(max(component_confidence.get(component, 0.0), panns_conf), 3)
    result["component_confidence"] = component_confidence
    result["confidence"] = round(float(np.mean([component_confidence["rain"], component_confidence["wind"]])), 3)
    return result


def _panns_score_to_intensity(score: float, labels: list[str], fallback: str) -> str:
    """Conservative zero-shot bucket mapping until site calibration exists."""
    if score < 0.08:
        return "none" if fallback == "none" else fallback
    if score < 0.20:
        return labels[1] if len(labels) > 1 else fallback
    if score < 0.42:
        return labels[2] if len(labels) > 2 else fallback
    return labels[-1]


def _load_panns_labels(panns_mod) -> list[str]:
    labels_obj = getattr(panns_mod, "labels", None)
    if isinstance(labels_obj, list):
        return [str(label) for label in labels_obj]
    if labels_obj is not None and hasattr(labels_obj, "labels"):
        return [str(label) for label in labels_obj.labels]

    labels_mod = importlib.import_module("panns_inference.labels")
    labels_obj = getattr(labels_mod, "labels", None)
    if labels_obj is None:
        labels_obj = getattr(labels_mod, "classes", None)
    if labels_obj is None:
        raise RuntimeError("Could not locate PANNs AudioSet labels")
    return [str(label) for label in labels_obj]


def _find_label_index(labels: list[str], target: str) -> int | None:
    target_norm = target.casefold()
    for idx, label in enumerate(labels):
        if label.casefold() == target_norm:
            return idx
    for idx, label in enumerate(labels):
        if target_norm in label.casefold():
            return idx
    return None


def _unavailable(reason: str) -> PannsEvidence:
    return PannsEvidence(
        available=False,
        reason=reason,
        scores={"rain": 0.0, "wind": 0.0, "thunder": 0.0},
        labels={"rain": {}, "wind": {}, "thunder": {}},
    )


def _restore_home(old_home: str | None) -> None:
    if old_home is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = old_home
