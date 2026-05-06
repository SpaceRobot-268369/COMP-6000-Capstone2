"""Layer D audio mixer for generation mode.

Layer D consumes the retrieval/planning outputs from Layers A, B, and C and
renders a short browser-playable WAV. It does not train or generate new calls;
it only loads local clips, applies bounded transform hints, and mixes them.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SR = 22_050
DEFAULT_DURATION_SEC = 30.0


def _to_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = PROJECT_ROOT / path_value
    if not path.exists() or path.stat().st_size == 0:
        return None
    return path


def _load_mono(path: Path, sr: int, offset: float = 0.0,
               duration: Optional[float] = None) -> np.ndarray:
    import librosa

    y, _ = librosa.load(
        str(path),
        sr=sr,
        mono=True,
        offset=max(0.0, offset),
        duration=duration,
    )
    return y.astype("float32", copy=False)


def _fit_duration(y: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        return np.zeros(0, dtype="float32")
    if y.size == 0:
        return np.zeros(samples, dtype="float32")
    if y.size >= samples:
        return y[:samples].astype("float32", copy=False)

    reps = int(math.ceil(samples / max(y.size, 1)))
    return np.tile(y, reps)[:samples].astype("float32", copy=False)


def _apply_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    rate = _clamp(rate, 0.96, 1.04)
    if y.size < 2048 or abs(rate - 1.0) < 0.002:
        return y
    try:
        import librosa

        return librosa.effects.time_stretch(y, rate=rate).astype("float32", copy=False)
    except Exception:
        return y


def _apply_pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    semitones = _clamp(semitones, -0.5, 0.5)
    if y.size < 2048 or abs(semitones) < 0.01:
        return y
    try:
        import librosa

        return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype("float32", copy=False)
    except Exception:
        return y


def _apply_filters(y: np.ndarray, sr: int, highpass_hz: float, lowpass_hz: float) -> np.ndarray:
    if y.size < 32:
        return y
    try:
        from scipy.signal import butter, sosfiltfilt

        nyquist = sr / 2.0
        out = y
        hp = _clamp(highpass_hz, 0.0, nyquist - 100.0)
        lp = _clamp(lowpass_hz, 100.0, nyquist - 50.0)
        if hp > 20.0:
            sos = butter(2, hp / nyquist, btype="highpass", output="sos")
            out = sosfiltfilt(sos, out).astype("float32", copy=False)
        if lp < nyquist - 80.0:
            sos = butter(2, lp / nyquist, btype="lowpass", output="sos")
            out = sosfiltfilt(sos, out).astype("float32", copy=False)
        return out
    except Exception:
        return y


def _apply_fades(y: np.ndarray, sr: int, fade_in_sec: float, fade_out_sec: float) -> np.ndarray:
    out = y.astype("float32", copy=True)
    if out.size == 0:
        return out
    fade_in = min(int(max(0.0, fade_in_sec) * sr), out.size)
    fade_out = min(int(max(0.0, fade_out_sec) * sr), out.size)
    if fade_in > 1:
        out[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype="float32")
    if fade_out > 1:
        out[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype="float32")
    return out


def _to_stereo(y: np.ndarray, pan: float = 0.0) -> np.ndarray:
    pan = _clamp(pan, -1.0, 1.0)
    left = math.cos((pan + 1.0) * math.pi / 4.0)
    right = math.sin((pan + 1.0) * math.pi / 4.0)
    stereo = np.vstack([y * left, y * right])
    return stereo.astype("float32", copy=False)


def _mix_at(base: np.ndarray, layer: np.ndarray, start_sample: int) -> None:
    if layer.size == 0 or start_sample >= base.shape[1]:
        return
    start = max(0, start_sample)
    end = min(base.shape[1], start + layer.shape[1])
    if end <= start:
        return
    base[:, start:end] += layer[:, : end - start]


def _spectrogram_png_b64(y: np.ndarray, sr: int) -> str:
    try:
        import librosa
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mono = y.mean(axis=0)
        mel = librosa.feature.melspectrogram(y=mono, sr=sr, n_mels=128, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
        ax.set_title("Layer D Mixed Spectrogram")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def _wav_b64(y: np.ndarray, sr: int) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, y.T, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _prepare_weather_layer(layer: dict, sr: int, target_samples: int) -> tuple[Optional[np.ndarray], Optional[str]]:
    selected = layer.get("selected") or {}
    transform = layer.get("transform") or {}
    path = _resolve_path(selected.get("clip_path"))
    if not layer.get("enabled") or path is None:
        return None, "weather asset missing or disabled"

    target_duration = _to_float(transform.get("target_duration_sec"), DEFAULT_DURATION_SEC)
    target_duration = min(max(target_duration, 1.0), target_samples / sr)
    y = _load_mono(
        path,
        sr,
        offset=_to_float(transform.get("start_offset_sec")),
        duration=target_duration + 1.0,
    )
    y = _apply_time_stretch(y, _to_float(transform.get("time_stretch"), 1.0))
    y = _fit_duration(y, int(target_duration * sr))
    y = _apply_filters(
        y,
        sr,
        _to_float(transform.get("highpass_hz")),
        _to_float(transform.get("lowpass_hz"), sr / 2.0 - 100),
    )

    gain_db = _to_float(layer.get("gain_db"), -18.0) + _to_float(transform.get("gain_variation_db"))
    density = _clamp(_to_float(transform.get("density_scale"), 1.0), 0.5, 1.5)
    y = y * _db_to_amp(gain_db) * density
    y = _apply_fades(
        y,
        sr,
        _to_float(transform.get("fade_in_sec"), 1.0),
        _to_float(transform.get("fade_out_sec"), 1.0),
    )
    return _to_stereo(y, _to_float(transform.get("pan"), 0.0)), None


def _prepare_event_layer(event: dict, sr: int) -> tuple[Optional[np.ndarray], Optional[str]]:
    selected = event.get("selected") or {}
    transform = event.get("transform") or {}
    path = _resolve_path(selected.get("clip_path"))
    if not event.get("enabled") or path is None:
        return None, "event asset missing or disabled"

    target_duration = _to_float(
        transform.get("target_duration_sec"),
        _to_float((event.get("schedule") or {}).get("duration_seconds"), 3.0),
    )
    target_duration = _clamp(target_duration, 0.25, 12.0)
    y = _load_mono(
        path,
        sr,
        offset=_to_float(transform.get("source_offset_sec")),
        duration=target_duration + 0.5,
    )
    y = _apply_time_stretch(y, _to_float(transform.get("time_stretch"), 1.0))
    y = _apply_pitch_shift(y, sr, _to_float(transform.get("pitch_shift_semitones")))
    y = _fit_duration(y, int(target_duration * sr))
    y = _apply_filters(
        y,
        sr,
        _to_float(transform.get("highpass_hz")),
        _to_float(transform.get("lowpass_hz"), sr / 2.0 - 100),
    )
    gain_db = _to_float(event.get("gain_db"), -12.0) + _to_float(transform.get("gain_variation_db"))
    y = y * _db_to_amp(gain_db)
    y = _apply_fades(
        y,
        sr,
        _to_float(transform.get("fade_in_sec"), 0.05),
        _to_float(transform.get("fade_out_sec"), 0.1),
    )
    return _to_stereo(y, _to_float(transform.get("pan"), 0.0)), None


def mix_generation_layers(layer_a_response: dict, weather: dict, events: dict,
                          target_duration_sec: float = DEFAULT_DURATION_SEC,
                          sample_rate: int = DEFAULT_SR) -> dict:
    """Mix Layer A, B, and C outputs into a final WAV response fragment."""
    target_duration_sec = max(1.0, float(target_duration_sec))
    target_samples = int(target_duration_sec * sample_rate)
    limitations: list[str] = []
    included_layers: list[str] = []

    selected = layer_a_response.get("selected") or {}
    ambient_path = _resolve_path(selected.get("clip_path"))
    if ambient_path is None:
        return {
            "status": "unavailable",
            "final_audio_b64": "",
            "final_audio_mime": "audio/wav",
            "final_audio_ext": "wav",
            "final_image_b64": "",
            "mixer": {
                "target_duration_sec": target_duration_sec,
                "sample_rate": sample_rate,
                "included_layers": [],
                "clipping_prevention": {"applied": False},
                "limitations": ["Layer A ambient bed could not be loaded."],
            },
        }

    ambient = _load_mono(ambient_path, sample_rate, duration=target_duration_sec)
    ambient = _fit_duration(ambient, target_samples)
    ambient = _apply_fades(ambient, sample_rate, 0.2, 0.6)
    mix = _to_stereo(ambient * 0.82)
    included_layers.append("ambient_bed")

    for kind, layer in (weather.get("layers") or {}).items():
        try:
            prepared, limitation = _prepare_weather_layer(layer, sample_rate, target_samples)
            if prepared is None:
                if layer.get("enabled") and limitation:
                    limitations.append(f"Layer B {kind}: {limitation}.")
                continue
            _mix_at(mix, prepared, 0)
            included_layers.append(f"weather_{kind}")
        except Exception as exc:
            limitations.append(f"Layer B {kind} skipped: {exc}")

    for index, event in enumerate(events.get("events") or []):
        try:
            prepared, limitation = _prepare_event_layer(event, sample_rate)
            if prepared is None:
                if limitation:
                    limitations.append(f"Layer C event {index + 1}: {limitation}.")
                continue
            start_sec = _to_float((event.get("schedule") or {}).get("start_seconds"))
            _mix_at(mix, prepared, int(start_sec * sample_rate))
            included_layers.append("event")
        except Exception as exc:
            limitations.append(f"Layer C event {index + 1} skipped: {exc}")

    peak_before = float(np.max(np.abs(mix))) if mix.size else 0.0
    normalization_gain_db = 0.0
    normalization_applied = False
    if peak_before > 0.98:
        gain = 0.98 / peak_before
        mix *= gain
        normalization_gain_db = round(20.0 * math.log10(gain), 3)
        normalization_applied = True
    elif 0.0 < peak_before < 0.25:
        gain = min(0.65 / peak_before, 31.62)  # cap boost at about +30 dB
        mix *= gain
        normalization_gain_db = round(20.0 * math.log10(gain), 3)
        normalization_applied = True

    peak_after = float(np.max(np.abs(mix))) if mix.size else 0.0
    final_audio_b64 = _wav_b64(mix, sample_rate)
    final_image_b64 = _spectrogram_png_b64(mix, sample_rate)

    return {
        "status": "mixed",
        "final_audio_b64": final_audio_b64,
        "final_audio_mime": "audio/wav",
        "final_audio_ext": "wav",
        "final_image_b64": final_image_b64,
        "mixer": {
            "target_duration_sec": round(target_duration_sec, 2),
            "sample_rate": sample_rate,
            "included_layers": included_layers,
            "clipping_prevention": {
                "applied": normalization_applied,
                "peak_before": round(peak_before, 4),
                "peak_after": round(peak_after, 4),
                "normalization_gain_db": normalization_gain_db,
            },
            "limitations": limitations,
        },
        "explanation": (
            "Layer D mixed the retrieved ambient bed with prepared weather and "
            "biological event layers using local clips and bounded transform hints."
        ),
    }
