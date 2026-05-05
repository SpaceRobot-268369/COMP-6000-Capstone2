"""Layer B weather sound engine for generation mode.

Layer B is retrieval-first and ML-assisted: it does not synthesize weather
from scratch. It scores real clips with lightweight audio features, then
selects suitable wind and rain layer candidates for later mixing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE_DIR = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a"
DEFAULT_ASSET_MANIFEST = RESOURCE_DIR / "weather_asset_manifest.csv"

INTENSITY_ORDER = ["none", "light", "medium", "strong"]
RAIN_INTENSITY_ORDER = ["none", "light", "dense"]


@dataclass(frozen=True)
class WeatherCandidate:
    path: Path
    row: dict
    score: float
    feature_score: float
    env_score: float
    context_score: float


def _to_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp((value - lo) / (hi - lo))


def _intensity_from_score(score: float, dense_label: bool = False) -> str:
    if score < 0.25:
        return "none"
    if score < 0.5:
        return "light"
    if dense_label:
        return "dense"
    if score < 0.75:
        return "medium"
    return "strong"


def _target_wind_intensity(env: dict) -> tuple[str, float]:
    wind = _to_float(env.get("wind_speed_ms"))
    gust = _to_float(env.get("wind_max_ms"))
    target = max(wind, gust * 0.65)
    if target < 2:
        return "none", 0.0
    if target < 6:
        return "light", _norm(target, 2, 6)
    if target < 10:
        return "medium", _norm(target, 6, 10)
    return "strong", _norm(target, 10, 16)


def _target_rain_intensity(env: dict) -> tuple[str, float]:
    hourly = _to_float(env.get("precipitation_mm"))
    daily = _to_float(env.get("precipitation_daily_mm"))
    humidity = _to_float(env.get("humidity_pct"), 50.0)
    recent_rain = max(0.0, 1.0 - _to_float(env.get("days_since_rain"), 30.0) / 14.0)
    target = max(hourly, daily / 8.0) + _norm(humidity, 75, 100) * 0.35 + recent_rain * 0.25
    if hourly <= 0 and daily < 1 and humidity < 85:
        return "none", 0.0
    if target < 2:
        return "light", _norm(target, 0.1, 2)
    return "dense", _norm(target, 2, 8)


def analyse_weather_features(path: Path, duration: float = 45.0, offset: float = 0.0) -> dict:
    """Estimate weather characteristics from audio.

    The scores are intentionally lightweight and explainable. They combine
    broadband/noise-like features for rain with low-frequency/gust features
    for wind.
    """
    import numpy as np
    import librosa

    y, sr = librosa.load(str(path), sr=22_050, mono=True, duration=duration, offset=max(0.0, offset))
    if y.size == 0:
        raise ValueError("empty audio")

    y = y.astype("float32")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms = np.maximum(rms, 1e-8)
    rms_db = float(librosa.amplitude_to_db(np.array([np.median(rms)]), ref=1.0)[0])

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = np.maximum(stft.sum(axis=0), 1e-12)
    low_ratio = float(stft[freqs < 700].sum(axis=0).mean() / total.mean())
    mid_ratio = float(stft[(freqs >= 700) & (freqs < 3500)].sum(axis=0).mean() / total.mean())
    high_ratio = float(stft[freqs >= 3500].sum(axis=0).mean() / total.mean())

    centroid = float(librosa.feature.spectral_centroid(S=stft, sr=sr)[0].mean())
    flatness = float(librosa.feature.spectral_flatness(S=stft)[0].mean())
    zcr = float(librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0].mean())

    rms_db_series = librosa.amplitude_to_db(rms, ref=np.median)
    gustiness = float(np.percentile(rms_db_series, 95) - np.percentile(rms_db_series, 50))
    transient_rate = float(np.mean(np.diff(rms_db_series, prepend=rms_db_series[0]) > 6.0))

    rain_score = (
        0.35 * _norm(high_ratio, 0.12, 0.45)
        + 0.30 * _norm(flatness, 0.02, 0.18)
        + 0.20 * _norm(centroid, 1800, 5200)
        + 0.15 * (1.0 - _norm(transient_rate, 0.04, 0.18))
    )
    wind_score = (
        0.35 * _norm(low_ratio + 0.5 * mid_ratio, 0.45, 0.9)
        + 0.30 * _norm(gustiness, 3.0, 14.0)
        + 0.20 * (1.0 - _norm(centroid, 1800, 5200))
        + 0.15 * (1.0 - _norm(high_ratio, 0.18, 0.55))
    )

    return {
        "rms_db": round(rms_db, 3),
        "low_ratio": round(low_ratio, 5),
        "mid_ratio": round(mid_ratio, 5),
        "high_ratio": round(high_ratio, 5),
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_flatness": round(flatness, 6),
        "zero_crossing_rate": round(zcr, 6),
        "gustiness_db": round(gustiness, 3),
        "transient_rate": round(transient_rate, 6),
        "wind_audio_score": round(_clamp(wind_score), 4),
        "rain_audio_score": round(_clamp(rain_score), 4),
        "wind_intensity_audio": _intensity_from_score(wind_score),
        "rain_intensity_audio": _intensity_from_score(rain_score, dense_label=True),
    }


def load_weather_assets(manifest_path: Path = DEFAULT_ASSET_MANIFEST) -> list[dict]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("analysis_status") == "ok"]


def _path_for_row(row: dict) -> Path:
    return PROJECT_ROOT / row["clip_path"]


def _same_context_score(env: dict, row: dict) -> float:
    score = 0.0
    if str(env.get("sample_bin", "")).lower() == str(row.get("sample_bin", "")).lower():
        score += 0.35

    q_month = _to_float(env.get("month"), 0)
    r_month = _to_float(row.get("month"), 0)
    if q_month and r_month:
        diff = abs(q_month - r_month) % 12
        score += 0.35 * (1.0 - min(diff, 12 - diff) / 6.0)

    if str(env.get("month_range", "")).lower() == str(row.get("month_range", "")).lower():
        score += 0.2

    return _clamp(score)


def _weather_env_score(kind: str, env: dict, row: dict) -> float:
    if kind == "wind":
        query_wind = max(_to_float(env.get("wind_speed_ms")), _to_float(env.get("wind_max_ms")) * 0.65)
        row_wind = max(_to_float(row.get("wind_speed_ms")), _to_float(row.get("wind_max_ms")) * 0.65)
        query_dir = _to_float(env.get("wind_direction_deg"), -1.0)
        row_dir = _to_float(row.get("wind_direction_deg"), -1.0)
        speed_score = 1.0 - _clamp(abs(query_wind - row_wind) / 10.0)
        if query_dir < 0 or row_dir < 0:
            return speed_score
        direction_diff = abs(query_dir - row_dir) % 360.0
        direction_score = 1.0 - min(direction_diff, 360.0 - direction_diff) / 180.0
        return _clamp(0.85 * speed_score + 0.15 * direction_score)

    query_rain = max(_to_float(env.get("precipitation_mm")), _to_float(env.get("precipitation_daily_mm")) / 8.0)
    row_rain = max(_to_float(row.get("precipitation_mm")), _to_float(row.get("precipitation_daily_mm")) / 8.0)
    humidity_score = 1.0 - _clamp(abs(_to_float(env.get("humidity_pct"), 50) - _to_float(row.get("humidity_pct"), 50)) / 60.0)
    return _clamp(0.7 * (1.0 - _clamp(abs(query_rain - row_rain) / 6.0)) + 0.3 * humidity_score)


def _intensity_distance(target: str, actual: str, order: list[str]) -> float:
    if target not in order or actual not in order:
        return 1.0
    return abs(order.index(target) - order.index(actual)) / max(len(order) - 1, 1)


def _select_layer(kind: str, env: dict, target_intensity: str, target_strength: float,
                  assets: list[dict], seed: Optional[int]) -> Optional[WeatherCandidate]:
    if target_intensity == "none":
        return None

    score_key = f"{kind}_audio_score"
    intensity_key = f"{kind}_intensity_audio"
    order = RAIN_INTENSITY_ORDER if kind == "rain" else INTENSITY_ORDER
    candidates: list[WeatherCandidate] = []

    for row in assets:
        path = _path_for_row(row)
        if not path.exists() or path.stat().st_size == 0:
            continue

        feature_score = _to_float(row.get(score_key))
        if feature_score < 0.25:
            continue

        actual_intensity = str(row.get(intensity_key, "none"))
        intensity_score = 1.0 - _intensity_distance(target_intensity, actual_intensity, order)
        env_score = _weather_env_score(kind, env, row)
        context_score = _same_context_score(env, row)
        total = (
            0.45 * feature_score
            + 0.25 * intensity_score
            + 0.20 * env_score
            + 0.10 * context_score
        )
        candidates.append(WeatherCandidate(path, row, total, feature_score, env_score, context_score))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item.score, reverse=True)
    shortlist = candidates[: min(5, len(candidates))]
    return random.Random(seed).choice(shortlist)


def _gain_for(kind: str, intensity: str, strength: float) -> float:
    if kind == "wind":
        ranges = {
            "light": (-24.0, -18.0),
            "medium": (-18.0, -10.0),
            "strong": (-10.0, -4.0),
        }
    else:
        ranges = {
            "light": (-22.0, -14.0),
            "dense": (-12.0, -5.0),
        }
    lo, hi = ranges.get(intensity, (-60.0, -60.0))
    return round(lo + (hi - lo) * _clamp(strength), 2)


def _stable_transform_seed(kind: str, env: dict, candidate: WeatherCandidate,
                           seed: Optional[int]) -> int:
    seed_payload = {
        "kind": kind,
        "seed": seed,
        "clip_path": str(candidate.path.relative_to(PROJECT_ROOT)),
        "recording_id": candidate.row.get("recording_id"),
        "clip_index": candidate.row.get("clip_index"),
        "env": {
            key: env.get(key)
            for key in (
                "wind_speed_ms",
                "wind_direction_deg",
                "wind_max_ms",
                "precipitation_mm",
                "precipitation_daily_mm",
                "days_since_rain",
                "humidity_pct",
                "month",
                "month_range",
                "sample_bin",
            )
        },
    }
    raw = json.dumps(seed_payload, sort_keys=True, default=str).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16)


def _transform_plan(kind: str, candidate: Optional[WeatherCandidate],
                    target_intensity: str, target_strength: float, env: dict,
                    seed: Optional[int], target_duration_sec: float = 30.0) -> Optional[dict]:
    if candidate is None or target_intensity == "none":
        return None

    variation_seed = _stable_transform_seed(kind, env, candidate, seed)
    rng = random.Random(variation_seed)
    row = candidate.row
    clip_duration = max(_to_float(row.get("clip_duration_seconds"), 300.0), 0.0)
    safe_duration = max(target_duration_sec, 1.0)
    max_offset = max(0.0, clip_duration - safe_duration)
    start_offset = rng.uniform(0.0, max_offset) if max_offset > 0 else 0.0
    loop_required = clip_duration < safe_duration

    strength = _clamp(target_strength)
    if kind == "wind":
        gain_jitter = 0.8 + 1.8 * strength
        time_stretch = rng.uniform(0.985 - 0.015 * strength, 1.015 + 0.015 * strength)
        highpass_hz = round(rng.uniform(70, 130) - 35 * strength)
        lowpass_hz = round(rng.uniform(5200, 8200) - 1200 * strength)
        density_scale = round(0.85 + 0.30 * strength + rng.uniform(-0.04, 0.04), 3)
        fade_in = rng.uniform(1.5, 3.5)
        fade_out = rng.uniform(2.0, 4.5)
    else:
        gain_jitter = 0.7 + 1.5 * strength
        time_stretch = rng.uniform(0.99 - 0.01 * strength, 1.01 + 0.01 * strength)
        highpass_hz = round(rng.uniform(260, 520) + 80 * strength)
        lowpass_hz = round(rng.uniform(8500, 11200))
        density_scale = round(0.75 + 0.55 * strength + rng.uniform(-0.05, 0.05), 3)
        fade_in = rng.uniform(0.8, 2.0)
        fade_out = rng.uniform(1.0, 2.5)

    gain_variation = rng.uniform(-gain_jitter, gain_jitter)

    return {
        "start_offset_sec": round(start_offset, 2),
        "target_duration_sec": round(safe_duration, 2),
        "loop_required": loop_required,
        "gain_variation_db": round(gain_variation, 2),
        "time_stretch": round(_clamp(time_stretch, 0.96, 1.04), 4),
        "highpass_hz": max(20, int(highpass_hz)),
        "lowpass_hz": max(1000, int(lowpass_hz)),
        "fade_in_sec": round(fade_in, 2),
        "fade_out_sec": round(fade_out, 2),
        "density_scale": round(_clamp(density_scale, 0.5, 1.5), 3),
        "variation_seed": variation_seed,
        "planning_note": (
            "Subtle seed-controlled transform hints only; audio processing is deferred to Layer D."
        ),
    }


def _layer_metadata(kind: str, candidate: Optional[WeatherCandidate], target_intensity: str,
                    target_strength: float, env: Optional[dict] = None,
                    seed: Optional[int] = None,
                    target_duration_sec: float = 30.0) -> dict:
    if candidate is None:
        return {
            "enabled": False,
            "target_intensity": target_intensity,
            "selected": None,
            "transform": None,
            "confidence": 0.0,
            "gain_db": -60.0,
        }

    row = candidate.row
    transform = _transform_plan(
        kind,
        candidate,
        target_intensity,
        target_strength,
        env or {},
        seed,
        target_duration_sec,
    )
    return {
        "enabled": True,
        "target_intensity": target_intensity,
        "audio_intensity": row.get(f"{kind}_intensity_audio"),
        "confidence": round(_clamp(candidate.score), 3),
        "feature_score": round(candidate.feature_score, 3),
        "env_score": round(candidate.env_score, 3),
        "context_score": round(candidate.context_score, 3),
        "gain_db": _gain_for(kind, target_intensity, target_strength),
        "transform": transform,
        "selected": {
            "clip_path": str(candidate.path.relative_to(PROJECT_ROOT)),
            "recording_id": row.get("recording_id"),
            "clip_index": int(_to_float(row.get("clip_index"), 0)),
            "month": int(_to_float(row.get("month"), 0)),
            "month_range": row.get("month_range"),
            "sample_bin": row.get("sample_bin"),
            "wind_direction_deg": _to_float(row.get("wind_direction_deg")),
            "wind_audio_score": _to_float(row.get("wind_audio_score")),
            "rain_audio_score": _to_float(row.get("rain_audio_score")),
            "spectral_centroid_hz": _to_float(row.get("spectral_centroid_hz")),
            "spectral_flatness": _to_float(row.get("spectral_flatness")),
            "gustiness_db": _to_float(row.get("gustiness_db")),
        },
    }


def prepare_weather_layers(env: dict, seed: Optional[int] = None,
                           manifest_path: Path = DEFAULT_ASSET_MANIFEST,
                           target_duration_sec: float = 30.0) -> dict:
    """Return Layer B weather layer plan and retrieval metadata."""
    assets = load_weather_assets(manifest_path)
    wind_intensity, wind_strength = _target_wind_intensity(env)
    rain_intensity, rain_strength = _target_rain_intensity(env)

    if not assets:
        return {
            "status": "unavailable",
            "asset_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "layers": {
                "wind": _layer_metadata("wind", None, wind_intensity, wind_strength),
                "rain": _layer_metadata("rain", None, rain_intensity, rain_strength),
            },
            "mix_hints": {"prepared_only": True},
            "explanation": (
                "Layer B found no analysed weather asset index. Run the weather "
                "asset preparation script after clips are downloaded."
            ),
        }

    wind = _select_layer("wind", env, wind_intensity, wind_strength, assets, seed)
    rain = _select_layer("rain", env, rain_intensity, rain_strength, assets, None if seed is None else seed + 17)
    wind_meta = _layer_metadata("wind", wind, wind_intensity, wind_strength, env, seed, target_duration_sec)
    rain_meta = _layer_metadata(
        "rain",
        rain,
        rain_intensity,
        rain_strength,
        env,
        None if seed is None else seed + 17,
        target_duration_sec,
    )
    enabled = [name for name, meta in (("wind", wind_meta), ("rain", rain_meta)) if meta["enabled"]]

    if not enabled and wind_intensity == "none" and rain_intensity == "none":
        status = "no_weather_needed"
        explanation = "Layer B did not select wind or rain because requested conditions are calm and dry."
    elif not enabled:
        status = "no_matching_assets"
        explanation = (
            "Layer B detected weather conditions, but no analysed real clip matched "
            "the requested intensity closely enough."
        )
    else:
        status = "prepared"
        explanation = (
            "Layer B selected real weather clips using audio feature scores "
            "combined with the requested weather parameters. Final mixing is "
            "deferred to Layer D."
        )

    return {
        "status": status,
        "asset_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "layers": {"wind": wind_meta, "rain": rain_meta},
        "mix_hints": {
            "prepared_only": True,
            "transform_planning_only": True,
            "wind_gain_db": wind_meta["gain_db"],
            "rain_gain_db": rain_meta["gain_db"],
            "target_duration_sec": round(max(target_duration_sec, 1.0), 2),
        },
        "explanation": explanation,
    }
