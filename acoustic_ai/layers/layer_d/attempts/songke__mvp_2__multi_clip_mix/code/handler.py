"""Registry handler for the Layer D multi-clip mixer MVP implementation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import librosa
import numpy as np
import soundfile as sf

from .audio_format import normalize_audio_format
from .audio_mixer import (
    MIX_CHANNELS,
    MIX_SAMPLE_RATE,
    EventPlacement,
    LayerStem,
    MixRequest,
    prepare_weather_stem,
    render_mix,
)


@dataclass(frozen=True)
class MixerState:
    params: dict[str, Any]


def load(
    checkpoint_dir: Path | None,
    params: dict,
    extra: dict | None = None,
) -> MixerState:
    del checkpoint_dir, extra
    return MixerState(params=dict(params))


def generate(
    state: MixerState,
    seed: int | None = None,
    *,
    ambient_wav_bytes: bytes | None = None,
    weather_wav_bytes: bytes | None = None,
    event_wav_bytes: bytes | None = None,
    weather_clips: Sequence[Mapping[str, Any]] | None = None,
    event_clips: Sequence[Mapping[str, Any]] | None = None,
    event_start_s: float = 0.0,
    duration_s: float | None = None,
    placement_seed: int | None = None,
    **_ignored: object,
) -> dict:
    """Mix upstream A/B/C WAV bytes into the final Layer D output."""

    del seed
    if ambient_wav_bytes is None:
        raise ValueError("Layer D requires ambient_wav_bytes from Layer A")

    params = state.params
    resolved_duration_s = float(
        duration_s if duration_s is not None else params.get("default_duration_s", 30.0)
    )
    ambient = _decode_stem(ambient_wav_bytes, role="ambient", source_id="layer_a")
    resolved_placement_seed = _resolve_placement_seed(placement_seed, params)
    weather, weather_contract = _resolve_weather_input(
        weather_clips=weather_clips,
        weather_wav_bytes=weather_wav_bytes,
        duration_s=resolved_duration_s,
        placement_seed=resolved_placement_seed,
        layer_gain_db=float(params.get("weather_gain_db", -2.0)),
    )
    events, event_contract = _resolve_event_input(
        event_clips=event_clips,
        event_wav_bytes=event_wav_bytes,
        event_start_s=event_start_s,
        duration_s=resolved_duration_s,
        placement_seed=resolved_placement_seed,
        layer_gain_db=float(params.get("event_gain_db", -8.0)),
    )
    bandpass = params.get("event_bandpass_hz")
    event_bandpass_hz = (
        (float(bandpass[0]), float(bandpass[1]))
        if isinstance(bandpass, (list, tuple)) and len(bandpass) == 2
        else None
    )
    result = render_mix(
        MixRequest(
            ambient=ambient,
            weather=weather,
            events=events,
            duration_s=resolved_duration_s,
            event_activity_envelope=bool(params.get("event_activity_envelope", True)),
            event_boundary_fade_s=float(params.get("event_boundary_fade_s", 1.0)),
            event_gain_db=float(params.get("event_gain_db", -8.0)),
            event_bandpass_hz=event_bandpass_hz,
            weather_gain_db=float(params.get("weather_gain_db", -2.0)),
            peak_ceiling=float(params.get("peak_ceiling", 0.95)),
        )
    )
    layer_d_metadata = {
        **result.explanation,
        "attempt_contract": "multi_clip_mix_v2",
        "multi_clip_enabled": bool(weather_clips or event_clips),
        "placement_seed": placement_seed,
        "input_contract": {
            "weather": weather_contract,
            "events": event_contract,
        },
        "placed_clips": _placed_clips_metadata(weather_contract, event_contract),
    }
    wav_bytes = _encode_wav(result.audio, result.sample_rate)
    return {
        "wav_bytes": wav_bytes,
        "mel_db": _mel_db(result.audio[:, 0], result.sample_rate),
        "metadata": {
            "audio": {
                "duration_s": resolved_duration_s,
                "sample_rate": result.sample_rate,
                "channels": 1,
                "subtype": "PCM_16",
            },
            "layer_d": layer_d_metadata,
        },
    }


def _resolve_weather_input(
    *,
    weather_clips: Sequence[Mapping[str, Any]] | None,
    weather_wav_bytes: bytes | None,
    duration_s: float,
    placement_seed: int,
    layer_gain_db: float,
) -> tuple[LayerStem | None, dict[str, Any]]:
    if weather_clips is not None:
        clips = list(weather_clips)
        if not clips:
            return None, {"mode": "multi_clip", "clip_count": 0, "supported_clip_count": 0}
        rng = np.random.default_rng(placement_seed)
        target_frames = _duration_to_frames(duration_s)
        timeline = np.zeros((target_frames, MIX_CHANNELS), dtype=np.float32)
        clip_rows = []
        placement_count = 0
        for clip_index, clip in enumerate(clips):
            weather_type = str(clip.get("weather_type") or f"weather_{clip_index}")
            stem = _decode_stem(
                _clip_wav_bytes(clip, source_id="layer_b"),
                role="weather",
                source_id=weather_type,
            )
            continuous = bool(clip.get("continuous", True))
            gain = _resolve_clip_gain(
                clip.get("gain_db"),
                layer_gain_db=layer_gain_db,
                field=f"weather_clips[{clip_index}].gain_db",
            )
            if continuous:
                prepared = prepare_weather_stem(stem, duration_s)
                prepared_audio = _apply_gain_delta(
                    prepared.audio,
                    clip_gain_db=gain["applied_gain_db"],
                    layer_gain_db=layer_gain_db,
                )
                timeline += prepared_audio
                onset_values: list[float] | None = None
                placement_random = False
                source_duration_s = stem.audio.shape[0] / stem.sample_rate
                clip_rows.append(
                    {
                        "weather_type": clip.get("weather_type"),
                        "continuous": True,
                        "onsets_s": onset_values,
                        "placement_random": placement_random,
                        "placement_seed": None,
                        "source_duration_s": source_duration_s,
                        "gain_db": clip.get("gain_db"),
                        **gain,
                        "change": clip.get("change"),
                        "placement_count": 0,
                        "prepared": prepared.metadata,
                    }
                )
                continue

            onsets = clip.get("onsets_s")
            placement_random = onsets is None
            if placement_random:
                source_duration_s = stem.audio.shape[0] / stem.sample_rate
                onset_values = [
                    _random_onset_s(
                        rng,
                        duration_s=duration_s,
                        clip_duration_s=source_duration_s,
                    )
                ]
            else:
                onset_values = _coerce_onsets(onsets, field=f"weather_clips[{clip_index}].onsets_s")
                source_duration_s = stem.audio.shape[0] / stem.sample_rate
            placement_rows = _place_weather_clip(
                timeline,
                stem,
                onsets_s=onset_values,
                duration_s=duration_s,
                clip_gain_db=gain["applied_gain_db"],
                layer_gain_db=layer_gain_db,
            )
            placement_count += len(placement_rows)
            clip_rows.append(
                {
                    "weather_type": clip.get("weather_type"),
                    "continuous": False,
                    "onsets_s": onset_values,
                    "placement_random": placement_random,
                    "placement_seed": placement_seed if placement_random else None,
                    "source_duration_s": source_duration_s,
                    "gain_db": clip.get("gain_db"),
                    **gain,
                    "change": clip.get("change"),
                    "placement_count": len(placement_rows),
                    "placements": placement_rows,
                }
            )
        stem = LayerStem(
            role="weather",
            audio=np.ascontiguousarray(timeline, dtype=np.float32),
            sample_rate=MIX_SAMPLE_RATE,
            source_id="weather_timeline",
            metadata={
                "target_duration_s": duration_s,
                "clips": clip_rows,
                "placement_count": placement_count,
            },
        )
        return stem, {
            "mode": "multi_clip",
            "clip_count": len(clips),
            "placement_count": placement_count,
            "clips": clip_rows,
        }
    if weather_wav_bytes is None:
        return None, {"mode": "legacy_single_stem", "clip_count": 0}
    return (
        _decode_stem(weather_wav_bytes, role="weather", source_id="layer_b"),
        {"mode": "legacy_single_stem", "clip_count": 1},
    )


def _place_weather_clip(
    timeline: np.ndarray,
    stem: LayerStem,
    *,
    onsets_s: Sequence[float],
    duration_s: float,
    clip_gain_db: float,
    layer_gain_db: float,
) -> list[dict[str, Any]]:
    normalized = normalize_audio_format(
        stem.audio,
        stem.sample_rate,
        target_sample_rate=MIX_SAMPLE_RATE,
        target_channels=MIX_CHANNELS,
    )
    target_frames = _duration_to_frames(duration_s)
    placement_rows = []
    for onset_s in onsets_s:
        if not np.isfinite(onset_s) or onset_s < 0:
            raise ValueError("weather onset_s must be a non-negative finite value")
        start_frame = int(round(onset_s * MIX_SAMPLE_RATE))
        available_frames = max(0, target_frames - start_frame)
        written_frames = min(normalized.audio.shape[0], available_frames)
        if written_frames:
            clip_audio = _apply_gain_delta(
                normalized.audio[:written_frames],
                clip_gain_db=clip_gain_db,
                layer_gain_db=layer_gain_db,
            )
            timeline[start_frame : start_frame + written_frames] += clip_audio
        placement_rows.append(
            {
                "source_id": stem.source_id,
                "requested_start_s": float(onset_s),
                "start_frame": start_frame,
                "source_frames": normalized.audio.shape[0],
                "written_frames": written_frames,
                "trimmed_at_end": written_frames < normalized.audio.shape[0],
                "format_operations": list(normalized.operations),
            }
        )
    return placement_rows


def _resolve_event_input(
    *,
    event_clips: Sequence[Mapping[str, Any]] | None,
    event_wav_bytes: bytes | None,
    event_start_s: float,
    duration_s: float,
    placement_seed: int,
    layer_gain_db: float,
) -> tuple[tuple[EventPlacement, ...], dict[str, Any]]:
    if event_clips is not None:
        rng = np.random.default_rng(placement_seed)
        placements: list[EventPlacement] = []
        clip_rows = []
        for clip_index, clip in enumerate(event_clips):
            stem = _decode_stem(
                _clip_wav_bytes(clip, source_id="layer_c"),
                role="event",
                source_id=str(clip.get("species") or f"layer_c_event_{clip_index}"),
            )
            gain = _resolve_clip_gain(
                clip.get("gain_db"),
                layer_gain_db=layer_gain_db,
                field=f"event_clips[{clip_index}].gain_db",
            )
            stem = _scale_stem_for_layer_gain_override(
                stem,
                clip_gain_db=gain["applied_gain_db"],
                layer_gain_db=layer_gain_db,
            )
            onsets = clip.get("onsets_s")
            placement_random = onsets is None
            if placement_random:
                onset_values = [
                    _random_onset_s(
                        rng,
                        duration_s=duration_s,
                        clip_duration_s=stem.audio.shape[0] / stem.sample_rate,
                    )
                ]
            else:
                onset_values = _coerce_onsets(onsets, field=f"event_clips[{clip_index}].onsets_s")
            for onset_s in onset_values:
                placements.append(EventPlacement(stem=stem, start_s=onset_s))
            clip_rows.append(
                {
                    "species": clip.get("species"),
                    "onsets_s": onset_values,
                    "placement_random": placement_random,
                    "placement_seed": placement_seed if placement_random else None,
                    "source_duration_s": stem.audio.shape[0] / stem.sample_rate,
                    "gain_db": clip.get("gain_db"),
                    **gain,
                    "placement_count": len(onset_values),
                }
            )
        return (
            tuple(placements),
            {
                "mode": "multi_clip",
                "clip_count": len(event_clips),
                "placement_count": len(placements),
                "clips": clip_rows,
            },
        )
    if event_wav_bytes is None:
        return (), {"mode": "legacy_single_stem", "clip_count": 0, "placement_count": 0}
    return (
        (
            EventPlacement(
                stem=_decode_stem(event_wav_bytes, role="event", source_id="layer_c"),
                start_s=float(event_start_s),
            ),
        ),
        {"mode": "legacy_single_stem", "clip_count": 1, "placement_count": 1},
    )


def _placed_clips_metadata(
    weather_contract: Mapping[str, Any],
    event_contract: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "weather": [
            _weather_clip_summary(clip)
            for clip in weather_contract.get("clips", [])
            if isinstance(clip, Mapping)
        ],
        "events": [
            _event_clip_summary(clip)
            for clip in event_contract.get("clips", [])
            if isinstance(clip, Mapping)
        ],
    }


def _weather_clip_summary(clip: Mapping[str, Any]) -> dict[str, Any]:
    weather_type = clip.get("weather_type")
    continuous = bool(clip.get("continuous", False))
    return {
        "kind": "weather",
        "label": weather_type,
        "weather_type": weather_type,
        "continuous": continuous,
        "onsets_s": clip.get("onsets_s"),
        "placement_random": bool(clip.get("placement_random", False)),
        "placement_seed": clip.get("placement_seed"),
        "applied_gain_db": clip.get("applied_gain_db"),
        "gain_override": bool(clip.get("gain_override", False)),
        "layer_default_gain_db": clip.get("layer_default_gain_db"),
        "source_duration_s": clip.get("source_duration_s"),
        "placement_count": int(clip.get("placement_count") or 0),
        "placements": list(clip.get("placements") or []),
        "change": clip.get("change"),
    }


def _event_clip_summary(clip: Mapping[str, Any]) -> dict[str, Any]:
    species = clip.get("species")
    return {
        "kind": "event",
        "label": species,
        "species": species,
        "onsets_s": clip.get("onsets_s"),
        "placement_random": bool(clip.get("placement_random", False)),
        "placement_seed": clip.get("placement_seed"),
        "applied_gain_db": clip.get("applied_gain_db"),
        "gain_override": bool(clip.get("gain_override", False)),
        "layer_default_gain_db": clip.get("layer_default_gain_db"),
        "source_duration_s": clip.get("source_duration_s"),
        "placement_count": int(clip.get("placement_count") or 0),
    }


def _clip_wav_bytes(clip: Mapping[str, Any], *, source_id: str) -> bytes:
    wav = clip.get("wav", clip.get("wav_bytes"))
    if not isinstance(wav, (bytes, bytearray)):
        raise ValueError(f"{source_id} clip requires wav bytes")
    return bytes(wav)


def _resolve_clip_gain(
    value: object,
    *,
    layer_gain_db: float,
    field: str,
) -> dict[str, Any]:
    if value is None:
        gain_db = layer_gain_db
        override = False
    else:
        gain_db = float(value)
        override = True
    if not np.isfinite(gain_db):
        raise ValueError(f"{field} must be a finite dB value")
    return {
        "applied_gain_db": gain_db,
        "gain_override": override,
        "layer_default_gain_db": layer_gain_db,
        "pre_layer_gain_delta_db": gain_db - layer_gain_db,
    }


def _scale_stem_for_layer_gain_override(
    stem: LayerStem,
    *,
    clip_gain_db: float,
    layer_gain_db: float,
) -> LayerStem:
    return LayerStem(
        role=stem.role,
        audio=_apply_gain_delta(
            stem.audio,
            clip_gain_db=clip_gain_db,
            layer_gain_db=layer_gain_db,
        ),
        sample_rate=stem.sample_rate,
        source_id=stem.source_id,
        metadata=stem.metadata,
    )


def _apply_gain_delta(
    audio: np.ndarray,
    *,
    clip_gain_db: float,
    layer_gain_db: float,
) -> np.ndarray:
    delta_db = clip_gain_db - layer_gain_db
    gain_linear = float(10.0 ** (delta_db / 20.0))
    return np.asarray(audio * gain_linear, dtype=np.float32)


def _resolve_placement_seed(placement_seed: int | None, params: Mapping[str, Any]) -> int:
    if placement_seed is not None:
        return int(placement_seed)
    return int(params.get("placement_seed_default", 42))


def _random_onset_s(
    rng: np.random.Generator,
    *,
    duration_s: float,
    clip_duration_s: float,
) -> float:
    latest_start_s = max(0.0, duration_s - clip_duration_s)
    if latest_start_s == 0.0:
        return 0.0
    return float(rng.uniform(0.0, latest_start_s))


def _duration_to_frames(duration_s: float) -> int:
    frames = int(round(float(duration_s) * MIX_SAMPLE_RATE))
    if frames < 1:
        raise ValueError("duration_s must resolve to at least one frame")
    return frames


def _coerce_onsets(value: object, *, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a list of onset seconds")
    onsets = [float(item) for item in value]
    if not onsets:
        raise ValueError(f"{field} must contain at least one onset")
    return onsets


def _decode_stem(wav_bytes: bytes, *, role: str, source_id: str) -> LayerStem:
    try:
        audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{source_id} WAV could not be decoded") from exc
    return LayerStem(
        role=role,
        audio=audio,
        sample_rate=int(sample_rate),
        source_id=source_id,
    )


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _mel_db(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)
