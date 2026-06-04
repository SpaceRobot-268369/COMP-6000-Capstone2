"""Layer D mixer contracts and fixed runtime format."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from scipy import signal

from .audio_format import normalize_audio_format
from .audio_metrics import audio_metrics


MIX_SAMPLE_RATE = 22_050
MIX_CHANNELS = 1
EXPORT_SUBTYPE = "PCM_16"
PEAK_CEILING = 0.95
WEATHER_CROSSFADE_S = 0.1
EVENT_ACTIVITY_WINDOW_S = 0.05
EVENT_ACTIVITY_THRESHOLD = 1e-4
EVENT_BOUNDARY_FADE_S = 0.15

LayerRole = Literal["ambient", "weather", "event"]
LAYER_GAIN_DB: dict[LayerRole, float] = {
    "ambient": 0.0,
    "weather": -12.0,
    "event": -12.0,
}


@dataclass(frozen=True)
class LayerStem:
    """One source stem handed to Layer D before format normalization."""

    role: LayerRole
    audio: np.ndarray
    sample_rate: int
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventPlacement:
    """An event stem and its requested start time on the final timeline."""

    stem: LayerStem
    start_s: float


@dataclass(frozen=True)
class MixRequest:
    """All inputs required to render one Layer D soundscape."""

    ambient: LayerStem
    duration_s: float
    weather: LayerStem | None = None
    events: tuple[EventPlacement, ...] = ()
    event_activity_envelope: bool = True
    event_boundary_fade_s: float = EVENT_BOUNDARY_FADE_S
    event_gain_db: float = LAYER_GAIN_DB["event"]
    event_bandpass_hz: tuple[float, float] | None = None
    weather_gain_db: float = LAYER_GAIN_DB["weather"]
    peak_ceiling: float = PEAK_CEILING


@dataclass(frozen=True)
class MixResult:
    """In-memory Layer D output before WAV serialization."""

    audio: np.ndarray
    sample_rate: int
    explanation: dict[str, Any]


def render_mix(request: MixRequest) -> MixResult:
    """Prepare all requested layers and render one complete Layer D mix."""

    ambient = prepare_ambient_stem(request.ambient, request.duration_s)
    weather = (
        prepare_weather_stem(request.weather, request.duration_s)
        if request.weather is not None
        else None
    )
    event_timeline = (
        prepare_event_timeline(
            request.events,
            request.duration_s,
            apply_activity_envelope=request.event_activity_envelope,
            boundary_fade_s=request.event_boundary_fade_s,
            bandpass_hz=request.event_bandpass_hz,
        )
        if request.events
        else None
    )
    result = mix_aligned_stems(
        ambient,
        weather=weather,
        events=((event_timeline,) if event_timeline is not None else ()),
        gain_db_overrides={
            "weather": request.weather_gain_db,
            "event": request.event_gain_db,
        },
        peak_ceiling=request.peak_ceiling,
    )
    return MixResult(
        audio=result.audio,
        sample_rate=result.sample_rate,
        explanation={
            **result.explanation,
            "duration_s": request.duration_s,
            "event_activity_envelope": request.event_activity_envelope,
            "event_boundary_fade_s": request.event_boundary_fade_s,
            "event_gain_db": request.event_gain_db,
            "event_bandpass_hz": request.event_bandpass_hz,
            "weather_gain_db": request.weather_gain_db,
            "peak_ceiling": request.peak_ceiling,
            "prepared_layers": {
                "ambient": ambient.metadata,
                "weather": weather.metadata if weather is not None else None,
                "events": event_timeline.metadata if event_timeline is not None else None,
            },
        },
    )


def prepare_ambient_stem(ambient: LayerStem, duration_s: float) -> LayerStem:
    """Normalize an ambient stem and trim or repeat it to the target duration."""

    if ambient.role != "ambient":
        raise ValueError("prepare_ambient_stem requires an ambient stem")
    target_frames = _duration_to_frames(duration_s)
    normalized = normalize_audio_format(
        ambient.audio,
        ambient.sample_rate,
        target_sample_rate=MIX_SAMPLE_RATE,
        target_channels=MIX_CHANNELS,
    )
    fitted, duration_operation = _fit_repeating_audio(normalized.audio, target_frames)
    return LayerStem(
        role="ambient",
        audio=fitted,
        sample_rate=MIX_SAMPLE_RATE,
        source_id=ambient.source_id,
        metadata={
            **ambient.metadata,
            "format_operations": list(normalized.operations),
            "duration_operation": duration_operation,
            "target_duration_s": duration_s,
        },
    )


def prepare_weather_stem(
    weather: LayerStem,
    duration_s: float,
    *,
    crossfade_s: float = WEATHER_CROSSFADE_S,
) -> LayerStem:
    """Normalize weather and crossfade-loop it to the target duration."""

    if weather.role != "weather":
        raise ValueError("prepare_weather_stem requires a weather stem")
    target_frames = _duration_to_frames(duration_s)
    crossfade_frames = _duration_to_frames(crossfade_s)
    normalized = normalize_audio_format(
        weather.audio,
        weather.sample_rate,
        target_sample_rate=MIX_SAMPLE_RATE,
        target_channels=MIX_CHANNELS,
    )
    fitted, duration_operation, applied_crossfade_frames = _fit_weather_audio(
        normalized.audio,
        target_frames,
        crossfade_frames,
    )
    return LayerStem(
        role="weather",
        audio=fitted,
        sample_rate=MIX_SAMPLE_RATE,
        source_id=weather.source_id,
        metadata={
            **weather.metadata,
            "format_operations": list(normalized.operations),
            "duration_operation": duration_operation,
            "target_duration_s": duration_s,
            "crossfade_frames": applied_crossfade_frames,
            "crossfade_s": applied_crossfade_frames / MIX_SAMPLE_RATE,
        },
    )


def prepare_event_timeline(
    events: tuple[EventPlacement, ...],
    duration_s: float,
    *,
    apply_activity_envelope: bool = True,
    boundary_fade_s: float = EVENT_BOUNDARY_FADE_S,
    bandpass_hz: tuple[float, float] | None = None,
) -> LayerStem:
    """Normalize and place event stems on one full-length event timeline."""

    target_frames = _duration_to_frames(duration_s)
    timeline = np.zeros((target_frames, MIX_CHANNELS), dtype=np.float32)
    placement_rows = []
    for placement in events:
        if placement.stem.role != "event":
            raise ValueError("prepare_event_timeline requires event stems")
        if not np.isfinite(placement.start_s) or placement.start_s < 0:
            raise ValueError("event start_s must be a non-negative finite value")

        normalized = normalize_audio_format(
            placement.stem.audio,
            placement.stem.sample_rate,
            target_sample_rate=MIX_SAMPLE_RATE,
            target_channels=MIX_CHANNELS,
        )
        event_audio, bandpass_metadata = _apply_event_bandpass(
            normalized.audio,
            bandpass_hz,
        )
        if apply_activity_envelope:
            event_audio, activity_metadata = _apply_event_activity_envelope(
                event_audio,
                boundary_fade_s=boundary_fade_s,
            )
        else:
            activity_metadata = {"applied": False}
        start_frame = int(round(placement.start_s * MIX_SAMPLE_RATE))
        available_frames = max(0, target_frames - start_frame)
        written_frames = min(event_audio.shape[0], available_frames)
        if written_frames:
            timeline[start_frame : start_frame + written_frames] += event_audio[:written_frames]
        placement_rows.append(
            {
                "source_id": placement.stem.source_id,
                "requested_start_s": placement.start_s,
                "start_frame": start_frame,
                "source_frames": event_audio.shape[0],
                "written_frames": written_frames,
                "trimmed_at_end": written_frames < event_audio.shape[0],
                "format_operations": list(normalized.operations),
                "bandpass": bandpass_metadata,
                "activity_envelope": activity_metadata,
            }
        )

    return LayerStem(
        role="event",
        audio=timeline,
        sample_rate=MIX_SAMPLE_RATE,
        source_id="event_timeline",
        metadata={
            "target_duration_s": duration_s,
            "activity_envelope_applied": apply_activity_envelope,
            "bandpass_hz": bandpass_hz,
            "placements": placement_rows,
        },
    )


def export_mix_result(
    result: MixResult,
    wav_path: Path | str,
    explanation_path: Path | str,
) -> dict[str, Any]:
    """Write a validated mix result as PCM16 WAV plus explanation JSON."""

    _validate_mix_result(result)
    wav_output = Path(wav_path)
    explanation_output = Path(explanation_path)
    wav_output.parent.mkdir(parents=True, exist_ok=True)
    explanation_output.parent.mkdir(parents=True, exist_ok=True)

    sf.write(
        wav_output,
        result.audio,
        result.sample_rate,
        format="WAV",
        subtype=EXPORT_SUBTYPE,
    )
    exported_explanation = {
        **result.explanation,
        "export": {
            "wav_path": str(wav_output),
            "sample_rate": result.sample_rate,
            "channels": MIX_CHANNELS,
            "subtype": EXPORT_SUBTYPE,
            "metrics": audio_metrics(result.audio, result.sample_rate),
        },
    }
    explanation_output.write_text(
        json.dumps(exported_explanation, indent=2),
        encoding="utf-8",
    )
    return exported_explanation


def mix_aligned_stems(
    ambient: LayerStem,
    *,
    weather: LayerStem | None = None,
    events: tuple[LayerStem, ...] = (),
    gain_db_overrides: dict[LayerRole, float] | None = None,
    peak_ceiling: float = PEAK_CEILING,
) -> MixResult:
    """Mix already-normalized, full-timeline stems with fixed layer gains."""

    stems = (ambient,) + ((weather,) if weather is not None else ()) + events
    _validate_aligned_stems(stems)

    mixed = np.zeros_like(ambient.audio, dtype=np.float32)
    layer_rows = []
    gain_db_overrides = gain_db_overrides or {}
    for stem in stems:
        gain_db = float(gain_db_overrides.get(stem.role, LAYER_GAIN_DB[stem.role]))
        if not np.isfinite(gain_db):
            raise ValueError(f"{stem.role} gain must be finite")
        gain_linear = float(10.0 ** (gain_db / 20.0))
        mixed += stem.audio.astype(np.float32, copy=False) * gain_linear
        layer_rows.append(
            {
                "role": stem.role,
                "source_id": stem.source_id,
                "gain_db": gain_db,
                "gain_linear": gain_linear,
            }
        )

    mixed, peak_protection = _apply_peak_ceiling(mixed, peak_ceiling)

    return MixResult(
        audio=mixed,
        sample_rate=MIX_SAMPLE_RATE,
        explanation={
            "runtime_format": "22050_hz_mono_float32",
            "layers": layer_rows,
            "processing": ["fixed_gain_sum", "peak_ceiling"],
            "peak_protection": peak_protection,
        },
    )


def _apply_peak_ceiling(
    audio: np.ndarray,
    peak_ceiling: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not np.isfinite(peak_ceiling) or not 0.0 < peak_ceiling <= 1.0:
        raise ValueError("peak_ceiling must be finite and in the range (0, 1]")
    input_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    scale = peak_ceiling / input_peak if input_peak > peak_ceiling else 1.0
    protected = np.asarray(audio * scale, dtype=np.float32)
    output_peak = float(np.max(np.abs(protected))) if protected.size else 0.0
    return protected, {
        "ceiling": peak_ceiling,
        "applied": scale < 1.0,
        "scale": float(scale),
        "input_peak": input_peak,
        "output_peak": output_peak,
    }


def _duration_to_frames(duration_s: float) -> int:
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be a positive finite value")
    frames = int(round(duration_s * MIX_SAMPLE_RATE))
    if frames < 1:
        raise ValueError("duration_s must resolve to at least one frame")
    return frames


def _fit_repeating_audio(audio: np.ndarray, target_frames: int) -> tuple[np.ndarray, str]:
    source_frames = audio.shape[0]
    if source_frames > target_frames:
        return np.ascontiguousarray(audio[:target_frames], dtype=np.float32), "trim"
    if source_frames < target_frames:
        repeats = int(np.ceil(target_frames / source_frames))
        repeated = np.tile(audio, (repeats, 1))[:target_frames]
        return np.ascontiguousarray(repeated, dtype=np.float32), "loop"
    return np.ascontiguousarray(audio, dtype=np.float32), "none"


def _fit_weather_audio(
    audio: np.ndarray,
    target_frames: int,
    requested_crossfade_frames: int,
) -> tuple[np.ndarray, str, int]:
    source_frames = audio.shape[0]
    if source_frames > target_frames:
        fitted = np.ascontiguousarray(audio[:target_frames], dtype=np.float32)
        return fitted, "trim", 0
    if source_frames == target_frames:
        return np.ascontiguousarray(audio, dtype=np.float32), "none", 0

    if source_frames < 2:
        repeated = np.tile(audio, (target_frames, 1))[:target_frames]
        return np.ascontiguousarray(repeated, dtype=np.float32), "loop", 0

    crossfade_frames = min(requested_crossfade_frames, max(1, source_frames // 2))
    fade_in = np.linspace(
        0.0,
        1.0,
        crossfade_frames + 2,
        dtype=np.float32,
    )[1:-1, None]
    fade_out = 1.0 - fade_in
    fitted = np.asarray(audio, dtype=np.float32)
    while fitted.shape[0] < target_frames:
        overlap = fitted[-crossfade_frames:] * fade_out + audio[:crossfade_frames] * fade_in
        fitted = np.concatenate((fitted[:-crossfade_frames], overlap, audio[crossfade_frames:]))
    return (
        np.ascontiguousarray(fitted[:target_frames], dtype=np.float32),
        "loop_crossfade",
        crossfade_frames,
    )


def _apply_event_activity_envelope(
    audio: np.ndarray,
    *,
    boundary_fade_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not np.isfinite(boundary_fade_s) or boundary_fade_s <= 0:
        raise ValueError("event boundary_fade_s must be a positive finite value")
    window_frames = max(1, int(round(EVENT_ACTIVITY_WINDOW_S * MIX_SAMPLE_RATE)))
    fade_frames = max(1, int(round(boundary_fade_s * MIX_SAMPLE_RATE)))
    frame_count = int(np.ceil(audio.shape[0] / window_frames))
    padded_frames = frame_count * window_frames
    padded = np.pad(audio[:, 0], (0, padded_frames - audio.shape[0]))
    windows = padded.reshape(frame_count, window_frames)
    rms = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1))
    active = rms > EVENT_ACTIVITY_THRESHOLD

    envelope = np.zeros(audio.shape[0], dtype=np.float32)
    padded_active = np.pad(active.astype(np.int8), (1, 1))
    run_starts = np.flatnonzero(np.diff(padded_active) == 1)
    run_ends = np.flatnonzero(np.diff(padded_active) == -1)
    for start_window, end_window in zip(run_starts, run_ends):
        coarse_start = int(start_window * window_frames)
        coarse_end = min(int(end_window * window_frames), audio.shape[0])
        active_samples = np.flatnonzero(
            np.abs(audio[coarse_start:coarse_end, 0]) > EVENT_ACTIVITY_THRESHOLD
        )
        if not active_samples.size:
            continue
        start_frame = coarse_start + int(active_samples[0])
        end_frame = coarse_start + int(active_samples[-1]) + 1
        envelope[start_frame:end_frame] = 1.0
        run_frames = end_frame - start_frame
        applied_fade_frames = min(fade_frames, max(1, run_frames // 2))
        progress = np.linspace(0.0, 1.0, applied_fade_frames, dtype=np.float32)
        fade_in = progress * progress * (3.0 - 2.0 * progress)
        fade_out = fade_in[::-1]
        envelope[start_frame : start_frame + applied_fade_frames] *= fade_in
        envelope[end_frame - applied_fade_frames : end_frame] *= fade_out

    processed = np.asarray(audio * envelope[:, None], dtype=np.float32)
    return processed, {
        "window_s": EVENT_ACTIVITY_WINDOW_S,
        "threshold_rms": EVENT_ACTIVITY_THRESHOLD,
        "boundary_fade_s": boundary_fade_s,
        "fade_curve": "smoothstep",
        "active_region_count": int(len(run_starts)),
        "active_window_ratio": float(np.mean(active)),
    }


def _apply_event_bandpass(
    audio: np.ndarray,
    bandpass_hz: tuple[float, float] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if bandpass_hz is None:
        return np.asarray(audio, dtype=np.float32), {"applied": False}
    low_hz, high_hz = map(float, bandpass_hz)
    nyquist = MIX_SAMPLE_RATE / 2.0
    if not 0.0 < low_hz < high_hz < nyquist:
        raise ValueError(f"event bandpass must satisfy 0 < low < high < {nyquist}")
    sos = signal.butter(
        4,
        (low_hz, high_hz),
        btype="bandpass",
        fs=MIX_SAMPLE_RATE,
        output="sos",
    )
    minimum_frames = 28
    if audio.shape[0] < minimum_frames:
        return np.asarray(audio, dtype=np.float32), {
            "applied": False,
            "reason": "source_too_short",
            "minimum_frames": minimum_frames,
            "source_frames": int(audio.shape[0]),
            "low_hz": low_hz,
            "high_hz": high_hz,
        }
    filtered = signal.sosfiltfilt(sos, audio[:, 0]).astype(np.float32)[:, None]
    return filtered, {
        "applied": True,
        "filter": "butterworth_zero_phase",
        "order": 4,
        "low_hz": low_hz,
        "high_hz": high_hz,
    }


def _validate_mix_result(result: MixResult) -> None:
    if result.sample_rate != MIX_SAMPLE_RATE:
        raise ValueError(f"mix result must use {MIX_SAMPLE_RATE} Hz")
    if result.audio.ndim != 2 or result.audio.shape[1] != MIX_CHANNELS:
        raise ValueError(f"mix result must be shaped (frames, {MIX_CHANNELS})")
    if result.audio.shape[0] == 0:
        raise ValueError("mix result must not be empty")
    if not np.isfinite(result.audio).all():
        raise ValueError("mix result contains NaN or infinite values")


def _validate_aligned_stems(stems: tuple[LayerStem, ...]) -> None:
    expected_frames = stems[0].audio.shape[0]
    for stem in stems:
        if stem.sample_rate != MIX_SAMPLE_RATE:
            raise ValueError(f"{stem.source_id} must use {MIX_SAMPLE_RATE} Hz")
        if stem.audio.ndim != 2 or stem.audio.shape[1] != MIX_CHANNELS:
            raise ValueError(f"{stem.source_id} must be shaped (frames, {MIX_CHANNELS})")
        if stem.audio.shape[0] != expected_frames:
            raise ValueError("all stems must have the same frame count")
        if not np.isfinite(stem.audio).all():
            raise ValueError(f"{stem.source_id} contains NaN or infinite values")
