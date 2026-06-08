"""Layer B MVP weather stem selector.

This handler does not train or synthesize a full weather model. It selects a
short weather-only stem from the curated Layer B asset index, trims a
seed-controlled segment, applies simple RMS normalization, and returns the WAV
for frontend preview / Layer D handoff.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


_REPO_ROOT = Path(__file__).resolve().parents[6]
_ALLOWED_WEATHER = {"rain", "wind", "rain+wind"}
_ALLOWED_INTENSITY = {"light", "medium", "heavy"}
_USABLE_AUDIT_STATUSES = {
    "library_seed",
    "approved_from_audit002",
    "maybe_from_audit002",
    "site_ready",
    "yes",
    "maybe",
}
_INTENSITY_FALLBACKS = {
    "light": ["light", "medium", "heavy"],
    "medium": ["medium", "light", "heavy"],
    "heavy": ["heavy", "medium", "light"],
}
_MIN_SITE_CANDIDATES_FOR_SITE_ONLY = 3
_MEL_BINS = 128
_FFT_SIZE = 1024
_HOP_LENGTH = 256


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
    asset_bank = str(params.get("asset_bank", "")).strip()
    if asset_bank:
        assets = _load_json_bank(_resolve_path(asset_bank))
        return WeatherStemState(assets=assets, params=dict(params))

    index_path = _resolve_path(str(params.get("asset_index", "")))
    if not index_path.is_file():
        raise FileNotFoundError(f"Layer B asset index not found: {index_path}")

    assets: list[WeatherAsset] = []
    import csv
    with index_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("layer_d_use") == "reject":
                continue
            if not _is_usable_audit_status(row.get("human_audit_status", "")):
                continue
            path = _resolve_path(row.get("clip_path", ""))
            if path.is_file():
                assets.append(WeatherAsset(row=row, path=path))

    if not assets:
        raise FileNotFoundError(f"No materialized Layer B audio assets found in {index_path}")
    return WeatherStemState(assets=assets, params=dict(params))


def _load_json_bank(bank_root: Path) -> list[WeatherAsset]:
    index_path = bank_root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Layer B asset bank index not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)

    assets: list[WeatherAsset] = []
    for item in doc.get("assets", []):
        audio_path = str(item.get("audio_path", ""))
        row = dict(item.get("attributes") or {})
        row["asset_id"] = str(item.get("id", row.get("asset_id", "")))
        row["clip_path"] = audio_path
        path = (bank_root / audio_path).resolve()
        if path.is_file():
            assets.append(WeatherAsset(row=row, path=path))

    if not assets:
        raise FileNotFoundError(f"No materialized Layer B audio assets found in {index_path}")
    return assets


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

    mel_db = _compute_log_mel_db(normalized, sample_rate)
    wav_bytes = _encode_wav(normalized, sample_rate)
    row = asset.row
    prompt = _build_retrieval_prompt(
        row=row,
        weather_type=weather_type,
        intensity=intensity,
        duration_s=duration_s,
        retrieval_seed=run_seed,
        segment=segment,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )
    metadata = {
        "seed": run_seed,
        "retrieval_seed": run_seed,
        "prompt": prompt,
        "prompt_locked": False,
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
    return {"wav_bytes": wav_bytes, "mel_db": mel_db, "metadata": metadata}


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

    for candidate_intensity in _INTENSITY_FALLBACKS[intensity]:
        exact = [
            asset
            for asset in weather_matches
            if _matches_intensity(asset.row, weather_type, candidate_intensity)
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
    if weather_type == "rain+wind":
        return (
            primary == "rain+wind"
            or (_truthy(row.get("has_rain", "")) and _truthy(row.get("has_wind", "")))
        )
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


def _matches_intensity(row: dict[str, str], weather_type: str, intensity: str) -> bool:
    if weather_type == "rain+wind":
        return row.get("rain_intensity") == intensity or row.get("wind_intensity") == intensity
    return row.get(_intensity_field(weather_type)) == intensity


def _intensity_field(weather_type: str) -> str:
    if weather_type == "rain+wind":
        return "rain_intensity"
    if weather_type == "storm":
        return "thunder_intensity"
    return f"{weather_type}_intensity"


def _prefer_primary(assets: list[WeatherAsset]) -> list[WeatherAsset]:
    primary = [asset for asset in assets if asset.row.get("layer_d_use") == "primary"]
    preferred = primary or assets
    site = [asset for asset in preferred if asset.row.get("source_type") == "site"]
    if len(site) >= _MIN_SITE_CANDIDATES_FOR_SITE_ONLY:
        return site
    return preferred


def _build_retrieval_prompt(
    *,
    row: dict[str, str],
    weather_type: str,
    intensity: str,
    duration_s: float,
    retrieval_seed: int,
    segment: dict[str, Any],
    fallback_used: bool,
    fallback_reason: str,
) -> str:
    components = [
        name for name, present in (
            ("rain", _truthy(row.get("has_rain", ""))),
            ("wind", _truthy(row.get("has_wind", ""))),
            ("thunder", _truthy(row.get("has_thunder", ""))),
        )
        if present
    ]
    component_text = "+".join(components) if components else row.get("primary_weather", "weather")
    source = row.get("source_type") or "unknown source"
    role = row.get("layer_d_role") or "weather stem"
    asset_id = row.get("asset_id") or "unknown asset"
    selected_intensity = row.get(_intensity_field(weather_type), intensity)
    prompt = (
        f"{duration_s:g}-second Layer B weather stem retrieval, "
        f"target {weather_type} / {intensity}, selected {source} {role} asset {asset_id}, "
        f"components {component_text}, asset intensity {selected_intensity}, "
        f"retrieval_seed {retrieval_seed}, start {segment['start_s']:.3f}s"
    )
    if fallback_used and fallback_reason:
        prompt += f", fallback note: {fallback_reason}"
    return prompt


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


def _compute_log_mel_db(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    mono = mono.astype(np.float32, copy=False)
    if mono.size == 0:
        return np.zeros((_MEL_BINS, 1), dtype=np.float32)

    if mono.size < _FFT_SIZE:
        mono = np.pad(mono, (0, _FFT_SIZE - mono.size))

    frames = _frame_signal(mono, frame_size=_FFT_SIZE, hop_length=_HOP_LENGTH).astype(np.float64)
    window = np.hanning(_FFT_SIZE).astype(np.float64)
    spectrum = np.fft.rfft(frames * window[None, :], n=_FFT_SIZE, axis=1)
    power = np.maximum(np.abs(spectrum) ** 2, 1e-12)
    power = np.nan_to_num(power, nan=1e-12, posinf=1e12, neginf=1e-12)
    mel_filter = _mel_filterbank(sample_rate, n_fft=_FFT_SIZE, n_mels=_MEL_BINS).astype(np.float64)
    mel_power = np.maximum(np.einsum("mf,tf->mt", mel_filter, power, optimize=True), 1e-12)
    mel_power = np.nan_to_num(mel_power, nan=1e-12, posinf=1e12, neginf=1e-12)
    mel_db = 10.0 * np.log10(mel_power)
    mel_db -= float(np.max(mel_db))
    return np.clip(mel_db, -80.0, 0.0).astype(np.float32)


def _frame_signal(audio: np.ndarray, *, frame_size: int, hop_length: int) -> np.ndarray:
    if audio.size <= frame_size:
        return audio[:frame_size][None, :]
    frame_count = 1 + int(np.ceil((audio.size - frame_size) / hop_length))
    padded_size = frame_size + (frame_count - 1) * hop_length
    if audio.size < padded_size:
        audio = np.pad(audio, (0, padded_size - audio.size))
    frames = np.empty((frame_count, frame_size), dtype=np.float32)
    for i in range(frame_count):
        start = i * hop_length
        frames[i] = audio[start:start + frame_size]
    return frames


def _mel_filterbank(sample_rate: int, *, n_fft: int, n_mels: int) -> np.ndarray:
    min_hz = 50.0
    max_hz = sample_rate / 2.0
    mel_points = np.linspace(_hz_to_mel(min_hz), _hz_to_mel(max_hz), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2)
        if center > left:
            filters[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_usable_audit_status(value: str) -> bool:
    statuses = {part.strip() for part in str(value).split(";") if part.strip()}
    return bool(statuses & _USABLE_AUDIT_STATUSES)
