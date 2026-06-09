"""Registry handler for the Layer D layered mixer MVP implementation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .audio_mixer import EventPlacement, LayerStem, MixRequest, render_mix


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
    event_start_s: float = 0.0,
    duration_s: float | None = None,
    **_ignored: object,
) -> dict:
    """Mix upstream A/B/C WAV bytes into the final Layer D output."""

    if ambient_wav_bytes is None:
        raise ValueError("Layer D requires ambient_wav_bytes from Layer A")

    params = state.params
    resolved_duration_s = float(
        duration_s if duration_s is not None else params.get("default_duration_s", 30.0)
    )
    ambient = _decode_stem(ambient_wav_bytes, role="ambient", source_id="layer_a")
    weather = (
        _decode_stem(weather_wav_bytes, role="weather", source_id="layer_b")
        if weather_wav_bytes is not None
        else None
    )
    if event_wav_bytes is not None:
        event_stem = _decode_stem(event_wav_bytes, role="event", source_id="layer_c")
        start_s = _resolve_event_start_s(
            params,
            seed=seed,
            event_stem=event_stem,
            duration_s=resolved_duration_s,
            explicit_start_s=event_start_s,
        )
        events = (EventPlacement(stem=event_stem, start_s=start_s),)
    else:
        events = ()
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
            event_gain_db=float(params.get("event_gain_db", -18.0)),
            event_bandpass_hz=event_bandpass_hz,
            weather_gain_db=float(params.get("weather_gain_db", -12.0)),
            peak_ceiling=float(params.get("peak_ceiling", 0.95)),
        )
    )
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
            "layer_d": result.explanation,
        },
    }


def _resolve_event_start_s(
    params: dict[str, Any],
    *,
    seed: int | None,
    event_stem: LayerStem,
    duration_s: float,
    explicit_start_s: float,
) -> float:
    """Decide where the event lands on the final timeline.

    ``event_placement: "random"`` (default) draws a seeded onset inside
    ``event_start_window_s`` (default [2, 10]), clamped so the event still fits
    before ``duration_s`` (no end-trim). ``"fixed"`` honors ``explicit_start_s``.
    Seeding off the shared run seed keeps placement reproducible: same seed +
    same inputs -> same onset.
    """

    mode = str(params.get("event_placement", "random")).lower()
    if mode != "random":
        return max(0.0, float(explicit_start_s))

    window = params.get("event_start_window_s", [2.0, 10.0])
    try:
        lo, hi = float(window[0]), float(window[1])
    except (TypeError, IndexError, ValueError):
        lo, hi = 2.0, 10.0

    event_len_s = event_stem.audio.shape[0] / float(event_stem.sample_rate)
    # Latest start that still lets the whole event play before the end.
    latest_start = max(0.0, duration_s - event_len_s)
    hi = min(hi, latest_start)
    lo = min(max(0.0, lo), hi)

    rng = np.random.default_rng(seed)
    return float(rng.uniform(lo, hi))


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
