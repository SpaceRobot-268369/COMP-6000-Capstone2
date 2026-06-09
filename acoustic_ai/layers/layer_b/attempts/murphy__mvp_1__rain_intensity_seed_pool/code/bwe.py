"""Bandwidth-extension postprocess for the Layer B rain AudioLDM2 attempt.

This module is a production-shaped copy of the validated offline prototype in
``spectrum_diagnostic_20260606/bwe_prototype/scripts/bwe_prototype.py``.
The algorithm and locked default parameters should stay aligned with the
accepted ``phase4_bwe_mixed_24k_next_trial_m2p75`` trial.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import signal


OUTPUT_SR = 24000
N_FFT = 2048
HOP = 512
EPS = 1e-12


@dataclass(frozen=True)
class BweParams:
    cross_over_hz: float = 8000.0
    output_sample_rate_hz: int = OUTPUT_SR
    sbr_source_low_hz: float = 4000.0
    sbr_source_high_hz: float = 7900.0
    sbr_shift_hz: float = 4000.0
    sbr_target_low_hz: float = 8000.0
    sbr_target_high_hz: float = 11800.0
    noise_low_hz: float = 8000.0
    noise_high_hz: float = 11800.0
    envelope_source_low_hz: float = 3000.0
    envelope_source_high_hz: float = 7900.0
    envelope_smooth_ms: float = 35.0
    noise_to_sbr_rms_ratio: float = 0.35
    target_drop_8_11_minus_2_8_db: float = -10.985337681240505
    post_bwe_highband_trim_db: float = -2.75
    denoise_enabled: bool = False


DEFAULT_BWE_PARAMS = BweParams()


def params_from_dict(values: dict[str, Any] | None) -> BweParams:
    """Build BWE params from a registry/params dict while keeping defaults locked."""
    if not values:
        return DEFAULT_BWE_PARAMS
    allowed = {field.name for field in BweParams.__dataclass_fields__.values()}
    normalized = {key: value for key, value in values.items() if key in allowed}
    if "output_sr" in values:
        normalized["output_sample_rate_hz"] = values["output_sr"]
    return BweParams(**normalized)


def resample_audio(y: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return y.astype(np.float32)
    gcd = np.gcd(source_sr, target_sr)
    return signal.resample_poly(y, target_sr // gcd, source_sr // gcd).astype(np.float32)


def linear_phase_lowpass(y: np.ndarray, sr: int, cutoff_hz: float = 8000.0) -> np.ndarray:
    nyquist = sr / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff {cutoff_hz} Hz must be below Nyquist {nyquist} Hz")
    taps = signal.firwin(numtaps=1025, cutoff=cutoff_hz, fs=sr, window="hann")
    return signal.filtfilt(taps, [1.0], y).astype(np.float32)


def linear_phase_highpass(
    y: np.ndarray, sr: int, cutoff_hz: float, numtaps: int = 1025
) -> np.ndarray:
    nyquist = sr / 2.0
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"invalid highpass cutoff {cutoff_hz} Hz for Nyquist {nyquist}")
    taps = signal.firwin(
        numtaps=numtaps,
        cutoff=cutoff_hz,
        fs=sr,
        window="hann",
        pass_zero=False,
    )
    return signal.filtfilt(taps, [1.0], y).astype(np.float32)


def linear_phase_bandpass(
    y: np.ndarray, sr: int, low_hz: float, high_hz: float, numtaps: int = 1025
) -> np.ndarray:
    nyquist = sr / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(f"invalid bandpass {low_hz}-{high_hz} Hz for Nyquist {nyquist}")
    taps = signal.firwin(
        numtaps=numtaps,
        cutoff=[low_hz, high_hz],
        fs=sr,
        window="hann",
        pass_zero=False,
    )
    return signal.filtfilt(taps, [1.0], y).astype(np.float32)


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + EPS))


def match_rms(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_rms = rms(source)
    if source_rms <= 0:
        return source
    return (source * (rms(reference) / source_rms)).astype(np.float32)


def amplitude_envelope(y: np.ndarray, sr: int, smooth_ms: float = 35.0) -> np.ndarray:
    analytic = signal.hilbert(y)
    env = np.abs(analytic).astype(np.float32)
    win = max(3, int(sr * smooth_ms / 1000.0))
    if win % 2 == 0:
        win += 1
    kernel = np.hanning(win).astype(np.float32)
    kernel /= kernel.sum()
    env = np.convolve(env, kernel, mode="same").astype(np.float32)
    mean = float(env.mean())
    if mean > 0:
        env /= mean
    return np.clip(env, 0.0, 4.0).astype(np.float32)


def band_mean(db: np.ndarray, freqs: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(mask):
        raise ValueError(f"no FFT bins in band {lo_hz}-{hi_hz} Hz")
    return float(db[mask].mean())


def band_power_db_from_audio(y: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> float:
    freqs, _times, zxx = signal.stft(
        y,
        fs=sr,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2
    db = 10.0 * np.log10(power.mean(axis=1) + EPS)
    return band_mean(db, freqs, lo_hz, hi_hz)


def synthesize_sbr_highband(y_24k: np.ndarray, sr: int, params: BweParams) -> np.ndarray:
    source_4_8 = linear_phase_bandpass(
        y_24k, sr, params.sbr_source_low_hz, params.sbr_source_high_hz
    )
    analytic = signal.hilbert(source_4_8)
    t = np.arange(len(source_4_8), dtype=np.float64) / sr
    shifted = np.real(
        analytic * np.exp(2j * np.pi * params.sbr_shift_hz * t)
    ).astype(np.float32)
    return linear_phase_bandpass(
        shifted, sr, params.sbr_target_low_hz, params.sbr_target_high_hz
    )


def synthesize_noise_fill(
    seed: int | None, length: int, sr: int, envelope: np.ndarray, params: BweParams
) -> np.ndarray:
    rng_seed = int(seed if seed is not None else 0)
    rng = np.random.default_rng(rng_seed)
    noise = rng.standard_normal(length).astype(np.float32)
    noise = linear_phase_bandpass(noise, sr, params.noise_low_hz, params.noise_high_hz)
    noise *= envelope
    return linear_phase_bandpass(noise, sr, params.noise_low_hz, params.noise_high_hz)


def synthesize_highband_candidate(
    y_24k: np.ndarray, sr: int, seed: int | None, params: BweParams
) -> tuple[np.ndarray, dict[str, float]]:
    env_source = linear_phase_bandpass(
        y_24k, sr, params.envelope_source_low_hz, params.envelope_source_high_hz
    )
    env = amplitude_envelope(env_source, sr, smooth_ms=params.envelope_smooth_ms)
    sbr = synthesize_sbr_highband(y_24k, sr, params)
    noise = synthesize_noise_fill(seed, len(y_24k), sr, env, params)
    noise = match_rms(noise, sbr) * params.noise_to_sbr_rms_ratio
    highband = linear_phase_bandpass(
        sbr + noise, sr, params.sbr_target_low_hz, params.sbr_target_high_hz
    )
    return highband, {
        "sbr_rms": rms(sbr),
        "noise_rms": rms(noise),
        "highband_rms": rms(highband),
    }


def apply_bwe(
    audio: np.ndarray,
    source_sr: int,
    *,
    seed: int | None = None,
    params: BweParams | None = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Apply only the validated BWE mix stage.

    Downstream 80 Hz highpass, RMS matching, fade, and peak limiting are kept
    outside this function so the caller can enforce the full postprocess order.
    """
    params = params or DEFAULT_BWE_PARAMS
    output_sr = int(params.output_sample_rate_hz)
    y_24k = resample_audio(np.asarray(audio, dtype=np.float32), source_sr, output_sr)
    input_mid_db = band_power_db_from_audio(y_24k, output_sr, 2000.0, 8000.0)
    input_high_db = band_power_db_from_audio(y_24k, output_sr, 8000.0, 11025.0)
    lowband = linear_phase_lowpass(y_24k, output_sr, cutoff_hz=params.cross_over_hz)
    highband, highband_stats = synthesize_highband_candidate(
        y_24k, output_sr, seed, params
    )

    low_mid_db = band_power_db_from_audio(lowband, output_sr, 2000.0, 8000.0)
    high_db_before = band_power_db_from_audio(highband, output_sr, 8000.0, 11025.0)
    target_high_db = low_mid_db + params.target_drop_8_11_minus_2_8_db
    high_gain_db = (
        target_high_db
        - high_db_before
        + params.post_bwe_highband_trim_db
    )
    high_gain = float(10.0 ** (high_gain_db / 20.0))

    highband_matched = linear_phase_highpass(
        highband * high_gain, output_sr, params.cross_over_hz
    )
    mixed = (lowband + highband_matched).astype(np.float32)

    high_db_after = band_power_db_from_audio(
        highband_matched, output_sr, 8000.0, 11025.0
    )
    mixed_mid_db = band_power_db_from_audio(mixed, output_sr, 2000.0, 8000.0)
    mixed_high_db = band_power_db_from_audio(mixed, output_sr, 8000.0, 11025.0)
    metadata = {
        "enabled": True,
        "parameters": asdict(params),
        "input_sample_rate_hz": source_sr,
        "output_sample_rate_hz": output_sr,
        "target_drop_8_11_minus_2_8_db": params.target_drop_8_11_minus_2_8_db,
        "input_2_8khz_db": input_mid_db,
        "input_8_11khz_db": input_high_db,
        "lowband_2_8khz_db": low_mid_db,
        "target_high_8_11khz_db": target_high_db,
        "highband_8_11khz_before_db": high_db_before,
        "highband_gain_db": high_gain_db,
        "post_bwe_highband_trim_db": params.post_bwe_highband_trim_db,
        "highband_8_11khz_after_db": high_db_after,
        "mixed_2_8khz_db": mixed_mid_db,
        "mixed_8_11khz_db": mixed_high_db,
        "mixed_drop_8_11_minus_2_8_db": mixed_high_db - mixed_mid_db,
        "band_energy": {
            "before_bwe": {
                "2_8khz_db": input_mid_db,
                "8_11khz_db": input_high_db,
                "drop_8_11_minus_2_8_db": input_high_db - input_mid_db,
            },
            "after_bwe": {
                "2_8khz_db": mixed_mid_db,
                "8_11khz_db": mixed_high_db,
                "drop_8_11_minus_2_8_db": mixed_high_db - mixed_mid_db,
            },
        },
        **highband_stats,
    }
    return mixed, output_sr, metadata
