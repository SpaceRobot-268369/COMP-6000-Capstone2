"""Offline CLI for Layer E-B weather direct-detection analysis.

This smoke implementation establishes the analysis pipeline shape:
load audio, convert to mono, resample, split into windows, compute lightweight
features, and emit the schema-defined JSON. Model scores are placeholders until
CLAP/PANNs/YAMNet are wired in.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - local fallback for bare Python.
    yaml = None

try:
    from .audioset_scores import build_audioset_scorer
    from .gate_fusion import WEATHER_ELEMENTS, decide_weather_from_evidence
    from .model_scores import ALL_SCORE_KEYS, build_scorer
except ImportError:  # Allows direct script execution from repo root.
    from audioset_scores import build_audioset_scorer
    from gate_fusion import WEATHER_ELEMENTS, decide_weather_from_evidence
    from model_scores import ALL_SCORE_KEYS, build_scorer


ATTEMPT_ID = "murphy__mvp_1__weather_direct_detection"
ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ATTEMPT_ROOT / "params.yaml"

DEFAULT_PARAMS: dict[str, Any] = {
    "analysis_version": "e_b_weather_mvp_1",
    "sample_rate": 22050,
    "channels": 1,
    "window_s": 5.0,
    "hop_s": 2.5,
    "elements": ["rain", "wind", "thunder"],
    "thresholds": {
        "rain_present": 0.55,
        "wind_present": 0.55,
        "thunder_present": 0.60,
        "low_confidence_margin": 0.08,
        "clipping_ratio_warning": 0.001,
    },
    "fusion_weights": {
        "clap": 0.65,
        "audioset": 0.25,
        "feature_support": 0.10,
    },
    "prompts": {
        "rain": [
            "rain",
            "light rain",
            "steady rain",
            "heavy rain",
            "rain on leaves",
            "rainfall ambience",
        ],
        "wind": [
            "wind",
            "light wind",
            "strong wind",
            "wind in trees",
            "gusty wind",
        ],
        "thunder": [
            "thunder",
            "distant thunder",
            "thunder rumble",
            "thunderstorm",
        ],
        "none": [
            "quiet dry woodland ambience",
            "quiet outdoor ambience",
            "no weather sound",
        ],
        "contamination": {
            "bio": ["birdsong", "insects", "cicadas"],
            "human_machine": ["human voice", "machinery", "microphone handling noise"],
        },
    },
}


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    source_sample_rate: int
    source_channels: int

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples) / self.sample_rate)


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    if yaml is None:
        return DEFAULT_PARAMS
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        samples = (raw - 128.0) / 128.0
    elif sample_width == 2:
        raw = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        samples = raw / 32768.0
    elif sample_width == 3:
        byte_view = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((byte_view.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = byte_view
        sign = (byte_view[:, 2] & 0x80) != 0
        padded[sign, 3] = 0xFF
        raw = padded.view("<i4").reshape(-1).astype(np.float32)
        samples = raw / 8388608.0
    elif sample_width == 4:
        raw = np.frombuffer(frames, dtype="<i4").astype(np.float32)
        samples = raw / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        samples = samples.reshape(-1, channels)

    return np.clip(samples, -1.0, 1.0), sample_rate, channels


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return samples.mean(axis=1).astype(np.float32)


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    if len(samples) == 0:
        return samples.astype(np.float32, copy=False)

    duration_s = len(samples) / float(source_rate)
    target_len = max(1, int(round(duration_s * target_rate)))
    source_x = np.linspace(0.0, duration_s, num=len(samples), endpoint=False)
    target_x = np.linspace(0.0, duration_s, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def load_audio(path: Path, target_sample_rate: int) -> AudioData:
    samples, source_rate, source_channels = read_wav(path)
    mono = to_mono(samples)
    resampled = resample_linear(mono, source_rate, target_sample_rate)
    return AudioData(
        samples=np.clip(resampled, -1.0, 1.0),
        sample_rate=target_sample_rate,
        channels=1,
        source_sample_rate=source_rate,
        source_channels=source_channels,
    )


def iter_windows(samples: np.ndarray, sample_rate: int, window_s: float, hop_s: float):
    window_len = max(1, int(round(window_s * sample_rate)))
    hop_len = max(1, int(round(hop_s * sample_rate)))

    if len(samples) <= window_len:
        padded = np.zeros(window_len, dtype=np.float32)
        padded[: len(samples)] = samples
        yield 0.0, min(len(samples) / sample_rate, window_s), padded
        return

    for start in range(0, len(samples) - window_len + 1, hop_len):
        end = start + window_len
        yield start / sample_rate, end / sample_rate, samples[start:end]

    final_start = len(samples) - window_len
    if final_start > 0 and final_start % hop_len != 0:
        final_end = len(samples)
        yield final_start / sample_rate, final_end / sample_rate, samples[final_start:final_end]


def dbfs(value: float) -> float:
    if value <= 1e-12:
        return -120.0
    return float(20.0 * math.log10(value))


def spectral_features(window: np.ndarray, sample_rate: int) -> dict[str, float]:
    if len(window) == 0:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_flatness": 0.0,
            "spectral_entropy": 0.0,
            "low_20_700_ratio": 0.0,
            "high_2000_8000_ratio": 0.0,
        }

    tapered = window * np.hanning(len(window))
    spectrum = np.abs(np.fft.rfft(tapered)).astype(np.float64)
    freqs = np.fft.rfftfreq(len(tapered), d=1.0 / sample_rate)
    power = spectrum**2
    total_power = float(power.sum() + 1e-12)

    centroid = float((freqs * power).sum() / total_power)
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))
    probabilities = power / total_power
    entropy = float(-(probabilities * np.log2(probabilities + 1e-12)).sum())
    entropy_norm = entropy / max(1.0, math.log2(len(probabilities)))

    low_mask = (freqs >= 20.0) & (freqs <= 700.0)
    high_mask = (freqs >= 2000.0) & (freqs <= min(8000.0, sample_rate / 2.0))
    low_ratio = float(power[low_mask].sum() / total_power)
    high_ratio = float(power[high_mask].sum() / total_power)

    return {
        "spectral_centroid_hz": centroid,
        "spectral_flatness": flatness,
        "spectral_entropy": entropy_norm,
        "low_20_700_ratio": low_ratio,
        "high_2000_8000_ratio": high_ratio,
    }


def window_features(window: np.ndarray, sample_rate: int) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
    peak = float(np.max(np.abs(window))) if len(window) else 0.0
    clipping_ratio = float(np.mean(np.abs(window) >= 0.999)) if len(window) else 0.0
    features = {
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "clipping_ratio": clipping_ratio,
    }
    features.update(spectral_features(window, sample_rate))
    return features


def placeholder_scores(features: dict[str, float]) -> dict[str, float]:
    """Conservative placeholder until model scores are available."""
    rms_dbfs = features["rms_dbfs"]
    low_ratio = features["low_20_700_ratio"]
    high_ratio = features["high_2000_8000_ratio"]
    flatness = features["spectral_flatness"]

    audible = 1.0 if rms_dbfs > -55.0 else 0.0
    rain_hint = audible * min(0.35, 0.20 * flatness + 0.25 * high_ratio)
    wind_hint = audible * min(0.35, 0.30 * low_ratio + 0.10 * flatness)
    thunder_hint = audible * min(0.25, 0.25 * low_ratio)
    none_score = 0.75 if rms_dbfs <= -55.0 else 0.45

    return {
        "rain": float(rain_hint),
        "wind": float(wind_hint),
        "thunder": float(thunder_hint),
        "none": float(none_score),
        "bio_contamination": 0.0,
        "human_machine_contamination": 0.0,
    }


def combine_scores(
    model_scores: dict[str, float],
    audioset_scores: dict[str, float],
    feature_scores: dict[str, float],
    model_available: bool,
    audioset_available: bool,
    params: dict[str, Any],
) -> dict[str, float]:
    """Fuse model and feature evidence.

    The fused score is element-wise. This lets mixed weather such as rain+wind
    survive as two present elements instead of collapsing to one top class.
    """
    weights = params.get("fusion_weights", {})
    clap_weight = float(weights.get("clap", 0.65)) if model_available else 0.0
    audioset_weight = float(weights.get("audioset", 0.25)) if audioset_available else 0.0
    feature_weight = float(weights.get("feature_support", 0.10))
    total_weight = clap_weight + audioset_weight + feature_weight
    if total_weight <= 0.0:
        return dict(feature_scores)

    clap_weight /= total_weight
    audioset_weight /= total_weight
    feature_weight /= total_weight
    combined = {}
    for key in ALL_SCORE_KEYS:
        model_value = float(model_scores.get(key, 0.0))
        audioset_value = float(audioset_scores.get(key, 0.0))
        feature_value = float(feature_scores.get(key, 0.0))
        combined[key] = (
            (clap_weight * model_value)
            + (audioset_weight * audioset_value)
            + (feature_weight * feature_value)
        )
    return combined


def window_warnings(features: dict[str, float], params: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if features["clipping_ratio"] >= params["thresholds"]["clipping_ratio_warning"]:
        warnings.append("possible_clipping")
    return warnings


def aggregate_weather(window_results: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    elements = {}
    present_labels: list[str] = []
    element_coverages: dict[str, float] = {}
    element_peaks: dict[str, float] = {}

    for element in params["elements"]:
        threshold = float(params["thresholds"][f"{element}_present"])
        window_scores = [float(row["scores"][element]) for row in window_results]
        confidence = max(window_scores, default=0.0)
        coverage = (
            sum(score >= threshold for score in window_scores) / len(window_scores)
            if window_scores
            else 0.0
        )
        element_coverages[element] = float(coverage)
        element_peaks[element] = float(confidence)
        threshold = params["thresholds"][f"{element}_present"]
        present = confidence >= threshold
        intensity = "none"
        if present:
            if confidence >= 0.80 or coverage >= 0.65:
                intensity = "heavy"
            elif confidence >= 0.68 or coverage >= 0.35:
                intensity = "medium"
            else:
                intensity = "light"
            present_labels.append(element)
        elements[element] = {
            "present": present,
            "intensity": intensity,
            "confidence": round(float(confidence), 6),
            "coverage": round(float(coverage), 6),
        }

    warnings = sorted({warning for row in window_results for warning in row["warnings"]})
    if not present_labels:
        overall_label = "none"
        none = True
    else:
        overall_label = "+".join(present_labels)
        none = False
        if len(present_labels) >= 2:
            warnings.append("weather_mixed_with_ambient")
        if any(
            elements[element]["confidence"] < params["thresholds"][f"{element}_present"]
            + params["thresholds"]["low_confidence_margin"]
            for element in present_labels
        ):
            warnings.append("low_confidence")
        if (
            elements["wind"]["present"]
            and element_peaks.get("rain", 0.0)
            >= float(params["thresholds"]["rain_present"]) - 0.08
            and not elements["rain"]["present"]
        ):
            warnings.append("possible_rain_under_wind")
        if (
            elements["wind"]["present"]
            and element_peaks.get("thunder", 0.0)
            >= float(params["thresholds"]["thunder_present"]) - 0.10
            and not elements["thunder"]["present"]
        ):
            warnings.append("possible_wind_overload")

    return {
        "overall_label": overall_label,
        "none": none,
        "elements": elements,
        "warnings": sorted(set(warnings)),
    }


def _peak_scores(
    window_results: list[dict[str, Any]],
    score_section: str,
    nested_scores_key: str | None = None,
) -> dict[str, float]:
    peaks = {element: 0.0 for element in WEATHER_ELEMENTS}
    for row in window_results:
        section = row.get(score_section, {})
        scores = section.get(nested_scores_key, {}) if nested_scores_key else section
        for element in WEATHER_ELEMENTS:
            peaks[element] = max(peaks[element], float(scores.get(element, 0.0)))
    return peaks


def _coverage_from_evidence(
    window_results: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, float]:
    if not window_results:
        return {element: 0.0 for element in WEATHER_ELEMENTS}

    coverage: dict[str, float] = {}
    for element in WEATHER_ELEMENTS:
        threshold = float(params["thresholds"][f"{element}_present"])
        supported = 0
        for row in window_results:
            model_scores = row.get("model_scores", {}).get("scores", {})
            audioset_scores = row.get("audioset_scores", {}).get("scores", {})
            guard_scores = row.get("guard_scores", {}).get("scores", {})
            feature_scores = row.get("feature_scores", {})
            best_score = max(
                float(model_scores.get(element, 0.0)),
                float(audioset_scores.get(element, 0.0)),
                float(guard_scores.get(element, 0.0)),
                float(feature_scores.get(element, 0.0)),
            )
            if best_score >= threshold:
                supported += 1
        coverage[element] = supported / len(window_results)
    return coverage


def _weather_label_from_intensity(intensity: float) -> str:
    if intensity < 0.15:
        return "none"
    if intensity < 0.40:
        return "light"
    if intensity < 0.70:
        return "moderate"
    return "heavy"


def _window_element_evidence(row: dict[str, Any], element: str) -> float:
    model_scores = row.get("model_scores", {}).get("scores", {})
    audioset_scores = row.get("audioset_scores", {}).get("scores", {})
    guard_scores = row.get("guard_scores", {}).get("scores", {})
    feature_scores = row.get("feature_scores", {})
    return max(
        float(model_scores.get(element, 0.0)),
        float(audioset_scores.get(element, 0.0)),
        float(guard_scores.get(element, 0.0)),
        float(feature_scores.get(element, 0.0)),
    )


def _weather_variability(window_results: list[dict[str, Any]], element: str) -> float:
    if len(window_results) <= 1:
        return 0.0
    values = np.asarray(
        [_window_element_evidence(row, element) for row in window_results],
        dtype=np.float32,
    )
    if float(np.max(values)) <= 0.0:
        return 0.0
    return float(min(1.0, np.std(values) / max(0.10, float(np.max(values)))))


def weather_observations_from_decision(
    weather: dict[str, Any],
    window_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert the legacy gate result into the aggregator-facing E-B contract.

    The gate still owns presence/absence. This adapter exposes the same decision
    as continuous 0-1 summaries so the Layer E aggregator can consume E-B as an
    authoritative observation head.
    """
    observations: dict[str, Any] = {}
    elements = weather.get("elements", {})
    confidences: list[float] = []

    for element in ("wind", "rain"):
        decision = elements.get(element, {})
        confidence = float(decision.get("confidence", 0.0))
        coverage = float(decision.get("coverage", 0.0))
        present = bool(decision.get("present", False))
        intensity = confidence if present else min(0.14, confidence * 0.35)
        confidences.append(confidence)
        observations[element] = {
            "summary": {
                "intensity": round(float(intensity), 6),
                "variability": round(_weather_variability(window_results, element), 6),
                "coverage": round(coverage if present else 0.0, 6),
                "label": _weather_label_from_intensity(float(intensity)),
                "confidence": round(confidence, 6),
            }
        }

    thunder = elements.get("thunder", {})
    thunder_confidence = float(thunder.get("confidence", 0.0))
    thunder_present = bool(thunder.get("present", False))
    thunder_intensity = thunder_confidence if thunder_present else min(0.14, thunder_confidence * 0.35)
    confidences.append(thunder_confidence)
    observations["thunder"] = {
        "summary": {
            "intensity": round(float(thunder_intensity), 6),
            "variability": round(_weather_variability(window_results, "thunder"), 6),
            "coverage": round(float(thunder.get("coverage", 0.0)) if thunder_present else 0.0, 6),
            "label": _weather_label_from_intensity(float(thunder_intensity)),
            "confidence": round(thunder_confidence, 6),
        },
        "events": [],
        "mean_interval_s": None,
    }

    observations["confidence"] = round(float(np.mean(confidences)) if confidences else 0.0, 6)
    observations["derived_label"] = weather.get("overall_label", "none")
    observations["warnings"] = list(weather.get("warnings", []))
    return observations


def aggregate_weather_with_gates(
    window_results: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate window evidence, then apply transparent weather gates."""
    evidence = {
        "clap": _peak_scores(window_results, "model_scores", "scores"),
        "panns": _peak_scores(window_results, "audioset_scores", "scores"),
        "ast": _peak_scores(window_results, "guard_scores", "scores"),
        "features": _peak_scores(window_results, "feature_scores"),
    }
    coverage = _coverage_from_evidence(window_results, params)
    weather = decide_weather_from_evidence(evidence, coverage)
    window_warning_set = {
        warning
        for row in window_results
        for warning in row.get("warnings", [])
    }
    weather["warnings"] = sorted(set(weather["warnings"]) | window_warning_set)
    weather.setdefault("debug", {})["evidence"] = {
        channel: {
            element: round(float(score), 6)
            for element, score in scores.items()
        }
        for channel, scores in evidence.items()
    }
    return weather


def analyze(
    audio_path: Path,
    params: dict[str, Any],
    model_backend: str = "clap",
    audioset_backend: str = "panns",
    guard_backend: str = "none",
) -> dict[str, Any]:
    audio = load_audio(audio_path, int(params["sample_rate"]))
    scorer = build_scorer(params, model_backend)
    audioset_scorer = build_audioset_scorer(audioset_backend)
    guard_scorer = build_audioset_scorer(guard_backend)
    window_results = []
    model_available = False
    audioset_available = False
    guard_available = False
    for start_s, end_s, window in iter_windows(
        audio.samples,
        audio.sample_rate,
        float(params["window_s"]),
        float(params["hop_s"]),
    ):
        features = window_features(window, audio.sample_rate)
        feature_scores = placeholder_scores(features)
        model_result = scorer.score_window(window, audio.sample_rate)
        audioset_result = audioset_scorer.score_window(window, audio.sample_rate)
        guard_result = guard_scorer.score_window(window, audio.sample_rate)
        model_available = model_available or model_result.available
        audioset_available = audioset_available or audioset_result.available
        guard_available = guard_available or guard_result.available
        scores = combine_scores(
            model_result.scores,
            audioset_result.scores,
            feature_scores,
            model_result.available,
            audioset_result.available,
            params,
        )
        warnings = (
            window_warnings(features, params)
            + model_result.warnings
            + audioset_result.warnings
            + ([] if guard_backend == "none" else guard_result.warnings)
        )
        window_results.append(
            {
                "start_s": round(float(start_s), 6),
                "end_s": round(float(end_s), 6),
                "scores": {key: round(float(value), 6) for key, value in scores.items()},
                "model_scores": {
                    "available": model_result.available,
                    "backend": model_result.backend,
                    "scores": {
                        key: round(float(value), 6)
                        for key, value in model_result.scores.items()
                    },
                    "raw": model_result.raw,
                },
                "audioset_scores": {
                    "available": audioset_result.available,
                    "backend": audioset_result.backend,
                    "scores": {
                        key: round(float(value), 6)
                        for key, value in audioset_result.scores.items()
                    },
                    "raw": audioset_result.raw,
                },
                "guard_scores": {
                    "available": guard_result.available,
                    "backend": guard_result.backend,
                    "scores": {
                        key: round(float(value), 6)
                        for key, value in guard_result.scores.items()
                    },
                    "raw": guard_result.raw,
                },
                "feature_scores": {
                    key: round(float(value), 6)
                    for key, value in feature_scores.items()
                },
                "features": {key: round(float(value), 6) for key, value in features.items()},
                "warnings": sorted(set(warnings)),
            }
        )

    weather = aggregate_weather_with_gates(window_results, params)
    top_level_warnings = list(weather["warnings"])
    if audio.source_sample_rate != audio.sample_rate:
        top_level_warnings.append("unsupported_sample_rate_resampled")
        weather["warnings"] = sorted(set(top_level_warnings))
    observations = {
        "weather": weather_observations_from_decision(weather, window_results),
    }

    return {
        "attempt_id": ATTEMPT_ID,
        "analysis_version": params["analysis_version"],
        "audio": {
            "duration_s": round(audio.duration_s, 6),
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
            "source_sample_rate": audio.source_sample_rate,
            "source_channels": audio.source_channels,
        },
        "observations": observations,
        "weather": weather,
        "window_results": window_results,
        "debug": {
            "model_backend": model_backend,
            "model_scores_available": model_available,
            "audioset_backend": audioset_backend,
            "audioset_available": audioset_available,
            "guard_backend": guard_backend,
            "guard_available": guard_available,
            "note": "CLAP scorer is wired; it degrades safely when dependencies or model files are unavailable.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E-B weather analysis on a WAV file.")
    parser.add_argument("audio", type=Path, help="Input WAV file.")
    parser.add_argument("--out", type=Path, help="Output JSON path. Prints to stdout if omitted.")
    parser.add_argument("--params", type=Path, default=PARAMS_PATH, help="Params YAML path.")
    parser.add_argument(
        "--model-backend",
        choices=["clap", "none"],
        default="clap",
        help="Model scorer backend. 'clap' currently exposes an unavailable boundary.",
    )
    parser.add_argument(
        "--audioset-backend",
        choices=["panns", "none"],
        default="panns",
        help="AudioSet scorer backend. 'panns' safely degrades if unavailable.",
    )
    parser.add_argument(
        "--guard-backend",
        choices=["ast", "none"],
        default="none",
        help="Optional conservative guard scorer. AST is useful for thunder/wind cross-checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    result = analyze(
        args.audio,
        params,
        model_backend=args.model_backend,
        audioset_backend=args.audioset_backend,
        guard_backend=args.guard_backend,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
