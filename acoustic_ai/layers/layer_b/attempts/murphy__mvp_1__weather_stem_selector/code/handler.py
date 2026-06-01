"""Layer B MVP weather stem selector.

This handler does not train or synthesize a full weather model. It selects a
short weather-only stem from the curated Layer B asset index, trims a
seed-controlled segment, applies simple RMS normalization, and returns the WAV
for frontend preview / Layer D handoff.
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


_REPO_ROOT = Path(__file__).resolve().parents[6]
_ALLOWED_WEATHER = {"rain", "wind", "thunder", "storm"}
_ALLOWED_INTENSITY = {"light", "medium", "heavy"}
_USABLE_AUDIT_STATUSES = {
    "library_seed",
    "approved_from_audit002",
    "maybe_from_audit002",
    "yes",
    "maybe",
}
_INTENSITY_FALLBACKS = {
    "light": ["light", "medium", "heavy"],
    "medium": ["medium", "light", "heavy"],
    "heavy": ["heavy", "medium", "light"],
}


@dataclass(frozen=True)
class WeatherAsset:
    row: dict[str, str]
    path: Path


@dataclass(frozen=True)
class WeatherStemState:
    assets: list[WeatherAsset]
    params: dict[str, Any]


def load(checkpoint_dir: Path | None, params: dict, extra: dict | None = None) -> WeatherStemState:
    del checkpoint_dir, extra
    index_path = _resolve_path(str(params.get("asset_index", "")))
    if not index_path.is_file():
        raise FileNotFoundError(f"Layer B asset index not found: {index_path}")

    assets: list[WeatherAsset] = []
    with index_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("layer_d_use") == "reject":
                continue
            if row.get("human_audit_status") not in _USABLE_AUDIT_STATUSES:
                continue
            path = _resolve_path(row.get("clip_path", ""))
            if path.is_file():
                assets.append(WeatherAsset(row=row, path=path))

    if not assets:
        raise FileNotFoundError(f"No materialized Layer B audio assets found in {index_path}")
    return WeatherStemState(assets=assets, params=dict(params))


def generate(state: WeatherStemState, seed: int | None = None, **runtime_params) -> dict:
    weather_type = _clean_choice(
        runtime_params.get("weather_type"),
        default=str(state.params.get("default_weather_type", "rain")),
        allowed=_ALLOWED_WEATHER,
        field="weather_type",
    )
    intensity = _clean_choice(
        runtime_params.get("intensity"),
        default=str(state.params.get("default_intensity", "medium")),
        allowed=_ALLOWED_INTENSITY,
        field="intensity",
    )
    duration_s = _clean_duration(runtime_params.get("duration_s"), state.params)
    run_seed = int(seed if seed is not None else 42)

    candidates, fallback_used, fallback_reason = _select_candidates(
        state.assets,
        weather_type=weather_type,
        intensity=intensity,
    )
    rng = random.Random(f"{weather_type}:{intensity}:{duration_s}:{run_seed}")
    asset = candidates[rng.randrange(len(candidates))]
    audio, sample_rate, segment = _read_seeded_segment(asset.path, duration_s, rng)
    normalized, gain, loudness = _normalize(
        audio,
        intensity=intensity,
        target_rms_cfg=state.params.get("target_rms") or {},
        peak_ceiling=float(state.params.get("peak_ceiling", 0.95)),
    )

    wav_bytes = _encode_wav(normalized, sample_rate)
    row = asset.row
    metadata = {
        "seed": run_seed,
        "audio": {
            "duration_s": segment["actual_duration_s"],
            "sample_rate": sample_rate,
            "channels": int(normalized.shape[1]) if normalized.ndim == 2 else 1,
            "rms": loudness["rms_after"],
            "peak": loudness["peak_after"],
            "gain": gain,
        },
        "layer_b": {
            "requested": {
                "weather_type": weather_type,
                "intensity": intensity,
                "duration_s": duration_s,
            },
            "selected": {
                "asset_id": row.get("asset_id"),
                "clip_path": row.get("clip_path"),
                "source_type": row.get("source_type"),
                "primary_weather": row.get("primary_weather"),
                "layer_d_role": row.get("layer_d_role"),
                "layer_d_use": row.get("layer_d_use"),
                "human_audit_status": row.get("human_audit_status"),
                "components": {
                    "rain": _truthy(row.get("has_rain", "")),
                    "wind": _truthy(row.get("has_wind", "")),
                    "thunder": _truthy(row.get("has_thunder", "")),
                },
                "intensity": {
                    "rain": row.get("rain_intensity"),
                    "wind": row.get("wind_intensity"),
                    "thunder": row.get("thunder_intensity"),
                },
                "segment_start_s": segment["start_s"],
                "segment_end_s": segment["end_s"],
                "asset_duration_s": segment["asset_duration_s"],
                "looped_to_duration": segment["looped_to_duration"],
            },
            "fallback": {
                "used": fallback_used,
                "reason": fallback_reason,
                "candidate_count": len(candidates),
            },
        },
    }
    return {"wav_bytes": wav_bytes, "mel_db": None, "metadata": metadata}


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _clean_choice(value: Any, *, default: str, allowed: set[str], field: str) -> str:
    cleaned = str(value or default).strip().lower()
    if cleaned not in allowed:
        raise ValueError(f"{field} must be one of: {', '.join(sorted(allowed))}")
    return cleaned


def _clean_duration(value: Any, params: dict[str, Any]) -> float:
    if value in (None, ""):
        value = params.get("default_duration_s", 10.0)
    duration_s = float(value)
    if duration_s <= 0 or duration_s > 30:
        raise ValueError("duration_s must be greater than 0 and at most 30 seconds.")
    return duration_s


def _select_candidates(
    assets: list[WeatherAsset],
    *,
    weather_type: str,
    intensity: str,
) -> tuple[list[WeatherAsset], bool, str]:
    weather_matches = [asset for asset in assets if _matches_weather(asset.row, weather_type)]
    if not weather_matches:
        raise FileNotFoundError(f"No materialized Layer B assets for weather_type={weather_type!r}")

    exact_weather = [asset for asset in weather_matches if _is_exact_weather(asset.row, weather_type)]
    fallback_notes: list[str] = []
    if exact_weather:
        weather_matches = exact_weather
    else:
        fallback_notes.append(f"no pure {weather_type}; used mixed component asset")

    field = _intensity_field(weather_type)
    for candidate_intensity in _INTENSITY_FALLBACKS[intensity]:
        exact = [
            asset
            for asset in weather_matches
            if asset.row.get(field) == candidate_intensity
        ]
        if exact:
            if candidate_intensity != intensity:
                fallback_notes.append(f"no {intensity} {weather_type}; used {candidate_intensity}")
            fallback_used = bool(fallback_notes)
            reason = "; ".join(fallback_notes)
            return _prefer_primary(exact), fallback_used, reason

    fallback_notes.append(f"no intensity match for {intensity} {weather_type}; used any {weather_type}")
    return _prefer_primary(weather_matches), True, "; ".join(fallback_notes)


def _matches_weather(row: dict[str, str], weather_type: str) -> bool:
    primary = row.get("primary_weather", "")
    if weather_type == "storm":
        return primary == "storm" or (_truthy(row.get("has_rain", "")) and _truthy(row.get("has_thunder", "")))
    if primary == weather_type:
        return True
    component_field = {
        "rain": "has_rain",
        "wind": "has_wind",
        "thunder": "has_thunder",
    }[weather_type]
    return _truthy(row.get(component_field, ""))


def _is_exact_weather(row: dict[str, str], weather_type: str) -> bool:
    primary = row.get("primary_weather", "")
    if weather_type == "storm":
        return primary == "storm"
    return primary == weather_type


def _intensity_field(weather_type: str) -> str:
    if weather_type == "storm":
        return "thunder_intensity"
    return f"{weather_type}_intensity"


def _prefer_primary(assets: list[WeatherAsset]) -> list[WeatherAsset]:
    primary = [asset for asset in assets if asset.row.get("layer_d_use") == "primary"]
    return primary or assets


def _read_seeded_segment(path: Path, duration_s: float, rng: random.Random) -> tuple[np.ndarray, int, dict[str, Any]]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"Asset contains no audio: {path}")

    target_frames = max(1, int(round(duration_s * sample_rate)))
    asset_frames = int(audio.shape[0])
    asset_duration_s = asset_frames / sample_rate
    if asset_frames > target_frames:
        max_start = asset_frames - target_frames
        start_frame = rng.randint(0, max_start)
        segment = audio[start_frame:start_frame + target_frames]
        looped = False
    else:
        start_frame = 0
        if asset_frames < target_frames:
            repeats = int(np.ceil(target_frames / asset_frames))
            segment = np.tile(audio, (repeats, 1))[:target_frames]
            looped = True
        else:
            segment = audio
            looped = False

    end_frame = start_frame + min(target_frames, asset_frames - start_frame)
    info = {
        "start_s": round(start_frame / sample_rate, 3),
        "end_s": round(end_frame / sample_rate, 3),
        "asset_duration_s": round(asset_duration_s, 3),
        "actual_duration_s": round(segment.shape[0] / sample_rate, 3),
        "looped_to_duration": looped,
    }
    return segment.astype(np.float32, copy=False), sample_rate, info


def _normalize(
    audio: np.ndarray,
    *,
    intensity: str,
    target_rms_cfg: dict[str, Any],
    peak_ceiling: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    target_rms = float(target_rms_cfg.get(intensity, target_rms_cfg.get("medium", 0.05)))
    rms_before = float(np.sqrt(np.mean(np.square(audio))) + 1e-12)
    peak_before = float(np.max(np.abs(audio)) + 1e-12)
    gain = target_rms / rms_before
    if peak_before * gain > peak_ceiling:
        gain = peak_ceiling / peak_before
    out = np.clip(audio * gain, -peak_ceiling, peak_ceiling).astype(np.float32)
    return out, float(gain), {
        "rms_before": rms_before,
        "peak_before": peak_before,
        "rms_after": float(np.sqrt(np.mean(np.square(out))) + 1e-12),
        "peak_after": float(np.max(np.abs(out)) + 1e-12),
    }


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
