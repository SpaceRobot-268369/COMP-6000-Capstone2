#!/usr/bin/env python3
"""Offline BWE prototype harness for Layer B rain showcase samples.

Phase 0 only establishes an isolated prototype entrypoint. Later phases should
fill in calibration, synthesis, gain matching, and diagnostic plotting here.
This script deliberately writes only under the BWE workspace.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


TARGET_SR = 22050
PHASE2_SR = 24000
N_FFT = 2048
HOP = 512
EPS = 1e-12


def default_attempt_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    attempt_dir = default_attempt_dir()
    bwe_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Offline BWE prototype harness for rain showcase samples."
    )
    parser.add_argument(
        "--showcase-dir",
        type=Path,
        default=attempt_dir / "showcase",
        help="Read-only showcase directory containing seed_*_generated/audio.wav.",
    )
    parser.add_argument(
        "--training-pool-manifest",
        type=Path,
        default=Path.cwd()
        / "debug/murphy_layer_b_rain_smoke_training_pool_v0_20260606/caption_manifest.csv",
        help="Read-only real-rain reference manifest for later calibration phases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=bwe_dir / "outputs",
        help="BWE output directory. Must be inside the bwe_prototype workspace.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=bwe_dir / "figures",
        help="Diagnostic figure directory. Must be inside the bwe_prototype workspace.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned inputs/outputs without writing generated WAV files.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=(0, 1, 2, 3, 4, 5),
        help=(
            "Prototype phase to run. Phase 1 calibrates real-rain target; "
            "phase 2 upsamples showcase audio and isolates 0-8 kHz low band; "
            "phase 3 synthesizes an envelope-modulated 8-12 kHz high band; "
            "phase 4 gain-matches and mixes the BWE prototype; "
            "phase 5 reruns before/after/real spectrum diagnostics."
        ),
    )
    parser.add_argument(
        "--bwe-parameters",
        type=Path,
        default=None,
        help="Optional BWE parameter JSON for phase 4 trial settings.",
    )
    parser.add_argument(
        "--trial-suffix",
        type=str,
        default="",
        help="Optional suffix for phase 4/5 output dirs, e.g. _next_trial_m2p75.",
    )
    return parser.parse_args()


def ensure_inside(child: Path, parent: Path) -> None:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    if parent_resolved not in (child_resolved, *child_resolved.parents):
        raise SystemExit(f"Refusing to write outside BWE workspace: {child}")


def read_mono(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        y = signal.resample_poly(y, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    if len(y) == 0:
        raise ValueError(f"empty audio: {path}")
    return y, sr


def read_mono_native(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if len(y) == 0:
        raise ValueError(f"empty audio: {path}")
    return y, sr


def resample_audio(y: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return y.astype(np.float32)
    gcd = np.gcd(source_sr, target_sr)
    return signal.resample_poly(y, target_sr // gcd, source_sr // gcd).astype(np.float32)


def linear_phase_lowpass(y: np.ndarray, sr: int, cutoff_hz: float = 8000.0) -> np.ndarray:
    nyquist = sr / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff {cutoff_hz} Hz must be below Nyquist {nyquist} Hz")
    # Odd tap count gives integer group delay. filtfilt keeps phase effectively zero
    # for this offline diagnostic prototype.
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


def mean_psd_db(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y, sr = read_mono(path)
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
    return freqs, db


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


def apply_fade(y: np.ndarray, sr: int, fade_ms: float = 20.0) -> np.ndarray:
    n = min(len(y) // 2, int(sr * fade_ms / 1000.0))
    if n <= 1:
        return y.astype(np.float32)
    out = y.astype(np.float32).copy()
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    out[:n] *= fade_in
    out[-n:] *= fade_out
    return out


def peak_limit(y: np.ndarray, ceiling: float = 0.98) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(y)))
    if peak <= ceiling:
        return y.astype(np.float32), 1.0
    gain = ceiling / peak
    return (y * gain).astype(np.float32), gain


def run_phase1(args: argparse.Namespace, bwe_dir: Path) -> dict:
    with args.training_pool_manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in training pool manifest: {args.training_pool_manifest}")

    wav_paths = [Path.cwd() / row["audio_path"] for row in rows]
    for wav_path in wav_paths:
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty real-rain WAV: {wav_path}")

    per_file_rows = []
    psd_rows = []
    freqs_ref = None
    for wav_path in wav_paths:
        freqs, db = mean_psd_db(wav_path)
        if freqs_ref is None:
            freqs_ref = freqs
        elif not np.array_equal(freqs_ref, freqs):
            raise SystemExit("Frequency bins differ across real-rain files")
        psd_rows.append(db)
        band_2_8 = band_mean(db, freqs, 2000, 8000)
        band_8_11 = band_mean(db, freqs, 8000, 11025)
        per_file_rows.append(
            {
                "audio_path": str(wav_path.relative_to(Path.cwd())),
                "band_2_8khz_db": band_2_8,
                "band_8_11khz_db": band_8_11,
                "drop_8_11_minus_2_8_db": band_8_11 - band_2_8,
            }
        )

    psd = np.stack(psd_rows)
    band_2_8_values = np.array([row["band_2_8khz_db"] for row in per_file_rows])
    band_8_11_values = np.array([row["band_8_11khz_db"] for row in per_file_rows])
    drop_values = band_8_11_values - band_2_8_values

    target = {
        "phase": 1,
        "status": "real_rain_target_calibrated",
        "training_pool_manifest": str(args.training_pool_manifest),
        "real_rain_count": len(wav_paths),
        "sample_rate_hz": TARGET_SR,
        "n_fft": N_FFT,
        "hop_length": HOP,
        "anchor_bands": {
            "mid_band_2_8khz": {
                "mean_db": float(band_2_8_values.mean()),
                "std_db": float(band_2_8_values.std()),
                "min_db": float(band_2_8_values.min()),
                "max_db": float(band_2_8_values.max()),
            },
            "high_band_8_11khz": {
                "mean_db": float(band_8_11_values.mean()),
                "std_db": float(band_8_11_values.std()),
                "min_db": float(band_8_11_values.min()),
                "max_db": float(band_8_11_values.max()),
            },
        },
        "target_slope": {
            "high_minus_mid_db": float(drop_values.mean()),
            "std_db": float(drop_values.std()),
            "interpretation": "BWE 8-11 kHz band should sit this many dB below the 2-8 kHz band.",
        },
    }

    target_path = bwe_dir / "phase1_real_rain_target.json"
    target_path.write_text(json.dumps(target, indent=2) + "\n")

    csv_path = bwe_dir / "phase1_real_rain_per_file_bands.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audio_path",
                "band_2_8khz_db",
                "band_8_11khz_db",
                "drop_8_11_minus_2_8_db",
            ],
        )
        writer.writeheader()
        writer.writerows(per_file_rows)

    mean_psd_path = bwe_dir / "phase1_real_rain_mean_psd.csv"
    with mean_psd_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frequency_hz", "mean_psd_db", "std_psd_db"])
        writer.writeheader()
        mean_psd = psd.mean(axis=0)
        std_psd = psd.std(axis=0)
        for freq, mean_db, std_db in zip(freqs_ref, mean_psd, std_psd):
            writer.writerow(
                {
                    "frequency_hz": float(freq),
                    "mean_psd_db": float(mean_db),
                    "std_psd_db": float(std_db),
                }
            )

    return target


def run_phase2(args: argparse.Namespace, bwe_dir: Path, showcase_wavs: list[Path]) -> dict:
    phase1_target = bwe_dir / "phase1_real_rain_target.json"
    if not phase1_target.exists():
        raise SystemExit(f"Run phase 1 first; missing {phase1_target}")

    phase2_dir = bwe_dir / "phase2_lowband_24k"
    if phase2_dir.exists():
        raise SystemExit(f"output dir already exists, refusing to overwrite: {phase2_dir}")

    upsampled_dir = phase2_dir / "upsampled_24k"
    lowband_dir = phase2_dir / "lowband_0_8k_24k"
    upsampled_dir.mkdir(parents=True)
    lowband_dir.mkdir(parents=True)

    rows = []
    for wav_path in showcase_wavs:
        seed_name = wav_path.parent.name
        y, source_sr = read_mono_native(wav_path)
        y_24k = resample_audio(y, source_sr, PHASE2_SR)
        lowband = linear_phase_lowpass(y_24k, PHASE2_SR, cutoff_hz=8000.0)

        seed_up_dir = upsampled_dir / seed_name
        seed_low_dir = lowband_dir / seed_name
        seed_up_dir.mkdir(parents=True)
        seed_low_dir.mkdir(parents=True)
        upsampled_path = seed_up_dir / "audio_24k.wav"
        lowband_path = seed_low_dir / "audio_lowband_0_8k_24k.wav"

        sf.write(upsampled_path, y_24k, PHASE2_SR, subtype="PCM_16")
        sf.write(lowband_path, lowband, PHASE2_SR, subtype="PCM_16")

        rows.append(
            {
                "seed": seed_name,
                "source_audio_path": str(wav_path),
                "source_sample_rate_hz": source_sr,
                "output_sample_rate_hz": PHASE2_SR,
                "duration_s": len(y_24k) / PHASE2_SR,
                "lowpass_cutoff_hz": 8000,
                "upsampled_audio_path": str(upsampled_path),
                "lowband_audio_path": str(lowband_path),
                "source_peak": float(np.max(np.abs(y))),
                "upsampled_peak": float(np.max(np.abs(y_24k))),
                "lowband_peak": float(np.max(np.abs(lowband))),
            }
        )

    manifest_path = phase2_dir / "phase2_lowband_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "phase": 2,
        "status": "showcase_upsampled_and_lowband_isolated",
        "showcase_count": len(rows),
        "output_sample_rate_hz": PHASE2_SR,
        "lowpass_cutoff_hz": 8000,
        "phase2_dir": str(phase2_dir),
        "upsampled_dir": str(upsampled_dir),
        "lowband_dir": str(lowband_dir),
        "manifest": str(manifest_path),
        "notes": [
            "No high-frequency synthesis or mixing is performed in phase 2.",
            "Original showcase files are read-only and unchanged.",
        ],
    }
    (phase2_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def synthesize_sbr_highband(y_24k: np.ndarray, sr: int) -> np.ndarray:
    source_4_8 = linear_phase_bandpass(y_24k, sr, 4000.0, 7900.0)
    analytic = signal.hilbert(source_4_8)
    # Frequency shift by +4 kHz maps 4-8 kHz texture into 8-12 kHz.
    t = np.arange(len(source_4_8), dtype=np.float64) / sr
    shifted = np.real(analytic * np.exp(2j * np.pi * 4000.0 * t)).astype(np.float32)
    return linear_phase_bandpass(shifted, sr, 8000.0, 11800.0)


def synthesize_noise_fill(seed_name: str, length: int, sr: int, envelope: np.ndarray) -> np.ndarray:
    seed_int = int("".join(ch for ch in seed_name if ch.isdigit()) or "0")
    rng = np.random.default_rng(seed_int)
    noise = rng.standard_normal(length).astype(np.float32)
    noise = linear_phase_bandpass(noise, sr, 8000.0, 11800.0)
    noise *= envelope
    return linear_phase_bandpass(noise, sr, 8000.0, 11800.0)


def run_phase3(args: argparse.Namespace, bwe_dir: Path) -> dict:
    phase2_dir = bwe_dir / "phase2_lowband_24k"
    phase2_manifest = phase2_dir / "phase2_lowband_manifest.csv"
    if not phase2_manifest.exists():
        raise SystemExit(f"Run phase 2 first; missing {phase2_manifest}")

    phase3_dir = bwe_dir / "phase3_highband_synthesis_24k"
    if phase3_dir.exists():
        raise SystemExit(f"output dir already exists, refusing to overwrite: {phase3_dir}")

    with phase2_manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 10:
        raise SystemExit(f"Expected 10 phase 2 rows, found {len(rows)}")

    sbr_dir = phase3_dir / "sbr_copy_8_12k"
    noise_dir = phase3_dir / "noise_fill_8_12k"
    highband_dir = phase3_dir / "highband_candidate_8_12k"
    for directory in (sbr_dir, noise_dir, highband_dir):
        directory.mkdir(parents=True)

    out_rows = []
    for row in rows:
        seed = row["seed"]
        upsampled_path = Path(row["upsampled_audio_path"])
        lowband_path = Path(row["lowband_audio_path"])
        if not upsampled_path.exists() or not lowband_path.exists():
            raise SystemExit(f"Missing phase 2 input for {seed}")

        y_24k, sr = read_mono_native(upsampled_path)
        lowband, low_sr = read_mono_native(lowband_path)
        if sr != PHASE2_SR or low_sr != PHASE2_SR:
            raise SystemExit(f"Unexpected sample rate for {seed}: {sr}, {low_sr}")

        env_source = linear_phase_bandpass(y_24k, sr, 3000.0, 7900.0)
        env = amplitude_envelope(env_source, sr, smooth_ms=35.0)
        sbr = synthesize_sbr_highband(y_24k, sr)
        noise = synthesize_noise_fill(seed, len(y_24k), sr, env)
        noise = match_rms(noise, sbr) * 0.35
        highband = linear_phase_bandpass(sbr + noise, sr, 8000.0, 11800.0)

        seed_sbr_dir = sbr_dir / seed
        seed_noise_dir = noise_dir / seed
        seed_high_dir = highband_dir / seed
        for directory in (seed_sbr_dir, seed_noise_dir, seed_high_dir):
            directory.mkdir(parents=True)
        sbr_path = seed_sbr_dir / "audio_sbr_copy_8_12k.wav"
        noise_path = seed_noise_dir / "audio_noise_fill_8_12k.wav"
        highband_path = seed_high_dir / "audio_highband_candidate_8_12k.wav"

        sf.write(sbr_path, sbr, sr, subtype="PCM_16")
        sf.write(noise_path, noise, sr, subtype="PCM_16")
        sf.write(highband_path, highband, sr, subtype="PCM_16")

        out_rows.append(
            {
                "seed": seed,
                "sample_rate_hz": sr,
                "sbr_source_band_hz": "4000-7900",
                "sbr_target_band_hz": "8000-11800",
                "noise_band_hz": "8000-11800",
                "envelope_source_band_hz": "3000-7900",
                "envelope_smooth_ms": 35,
                "noise_to_sbr_rms_ratio": 0.35,
                "sbr_rms": rms(sbr),
                "noise_rms": rms(noise),
                "highband_rms": rms(highband),
                "sbr_path": str(sbr_path),
                "noise_path": str(noise_path),
                "highband_path": str(highband_path),
            }
        )

    manifest_path = phase3_dir / "phase3_highband_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {
        "phase": 3,
        "status": "highband_candidates_synthesized",
        "showcase_count": len(out_rows),
        "sample_rate_hz": PHASE2_SR,
        "sbr_strategy": "4-8 kHz analytic frequency shift to 8-12 kHz",
        "noise_strategy": "8-12 kHz shaped white noise modulated by 3-8 kHz envelope",
        "noise_to_sbr_rms_ratio": 0.35,
        "phase3_dir": str(phase3_dir),
        "manifest": str(manifest_path),
        "notes": [
            "No final BWE mixing or target-slope gain matching is performed in phase 3.",
            "Highband files are intermediate candidates for phase 4.",
        ],
    }
    (phase3_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_phase4(args: argparse.Namespace, bwe_dir: Path) -> dict:
    phase1_target_path = bwe_dir / "phase1_real_rain_target.json"
    phase2_manifest = bwe_dir / "phase2_lowband_24k/phase2_lowband_manifest.csv"
    phase3_manifest = bwe_dir / "phase3_highband_synthesis_24k/phase3_highband_manifest.csv"
    for required in (phase1_target_path, phase2_manifest, phase3_manifest):
        if not required.exists():
            raise SystemExit(f"missing prerequisite for phase 4: {required}")

    phase4_dir = bwe_dir / f"phase4_bwe_mixed_24k{args.trial_suffix}"
    if phase4_dir.exists():
        raise SystemExit(f"output dir already exists, refusing to overwrite: {phase4_dir}")
    output_dir = phase4_dir / "outputs"
    output_dir.mkdir(parents=True)

    target = json.loads(phase1_target_path.read_text())
    target_drop_db = float(target["target_slope"]["high_minus_mid_db"])
    bwe_parameters = {}
    if args.bwe_parameters is not None:
        if not args.bwe_parameters.exists():
            raise SystemExit(f"BWE parameter file not found: {args.bwe_parameters}")
        bwe_parameters = json.loads(args.bwe_parameters.read_text())
    post_bwe_highband_trim_db = float(
        bwe_parameters.get("post_bwe_highband_trim_db", 0.0)
    )

    with phase2_manifest.open(newline="") as f:
        phase2_rows = {row["seed"]: row for row in csv.DictReader(f)}
    with phase3_manifest.open(newline="") as f:
        phase3_rows = {row["seed"]: row for row in csv.DictReader(f)}
    if sorted(phase2_rows) != sorted(phase3_rows):
        raise SystemExit("phase 2 and phase 3 seed sets differ")

    rows = []
    for seed in sorted(phase2_rows):
        lowband_path = Path(phase2_rows[seed]["lowband_audio_path"])
        upsampled_path = Path(phase2_rows[seed]["upsampled_audio_path"])
        highband_path = Path(phase3_rows[seed]["highband_path"])
        for path in (lowband_path, upsampled_path, highband_path):
            if not path.exists() or path.stat().st_size == 0:
                raise SystemExit(f"missing or empty phase 4 input: {path}")

        lowband, sr = read_mono_native(lowband_path)
        upsampled, up_sr = read_mono_native(upsampled_path)
        highband, high_sr = read_mono_native(highband_path)
        if sr != PHASE2_SR or up_sr != PHASE2_SR or high_sr != PHASE2_SR:
            raise SystemExit(f"unexpected sample rates for {seed}: {sr}, {up_sr}, {high_sr}")

        low_mid_db = band_power_db_from_audio(lowband, sr, 2000.0, 8000.0)
        high_db_before = band_power_db_from_audio(highband, sr, 8000.0, 11025.0)
        target_high_db = low_mid_db + target_drop_db
        high_gain_db = target_high_db - high_db_before + post_bwe_highband_trim_db
        high_gain = float(10.0 ** (high_gain_db / 20.0))

        highband_matched = linear_phase_highpass(highband * high_gain, sr, 8000.0)
        mixed = lowband + highband_matched
        # Fixed post order requested: BWE -> 80 Hz highpass -> RMS match -> fade.
        mixed_hp = linear_phase_highpass(mixed, sr, 80.0, numtaps=513)
        mixed_rms_matched = match_rms(mixed_hp, upsampled)
        mixed_faded = apply_fade(mixed_rms_matched, sr, fade_ms=20.0)
        mixed_limited, limiter_gain = peak_limit(mixed_faded, ceiling=0.98)

        out_seed_dir = output_dir / seed
        out_seed_dir.mkdir(parents=True)
        out_wav = out_seed_dir / "audio_bwe_24k.wav"
        metadata_path = out_seed_dir / "metadata.json"
        sf.write(out_wav, mixed_limited, sr, subtype="PCM_16")

        high_db_after = band_power_db_from_audio(highband_matched, sr, 8000.0, 11025.0)
        mixed_mid_db = band_power_db_from_audio(mixed_limited, sr, 2000.0, 8000.0)
        mixed_high_db = band_power_db_from_audio(mixed_limited, sr, 8000.0, 11025.0)
        metadata = {
            "seed": seed,
            "phase": 4,
            "sample_rate_hz": sr,
            "target_drop_8_11_minus_2_8_db": target_drop_db,
            "lowband_2_8khz_db": low_mid_db,
            "target_high_8_11khz_db": target_high_db,
            "highband_8_11khz_before_db": high_db_before,
            "highband_gain_db": high_gain_db,
            "post_bwe_highband_trim_db": post_bwe_highband_trim_db,
            "highband_8_11khz_after_db": high_db_after,
            "mixed_2_8khz_db": mixed_mid_db,
            "mixed_8_11khz_db": mixed_high_db,
            "mixed_drop_8_11_minus_2_8_db": mixed_high_db - mixed_mid_db,
            "post_order": ["BWE mix", "80 Hz highpass", "RMS match", "20 ms fade", "peak limit if needed"],
            "limiter_gain": limiter_gain,
            "input_lowband_path": str(lowband_path),
            "input_highband_path": str(highband_path),
            "output_wav": str(out_wav),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append(
            {
                "seed": seed,
                "output_wav": str(out_wav),
                "metadata": str(metadata_path),
                "target_drop_db": target_drop_db,
                "lowband_2_8khz_db": low_mid_db,
                "target_high_8_11khz_db": target_high_db,
                "highband_8_11khz_before_db": high_db_before,
                "highband_gain_db": high_gain_db,
                "post_bwe_highband_trim_db": post_bwe_highband_trim_db,
                "highband_8_11khz_after_db": high_db_after,
                "mixed_2_8khz_db": mixed_mid_db,
                "mixed_8_11khz_db": mixed_high_db,
                "mixed_drop_db": mixed_high_db - mixed_mid_db,
                "limiter_gain": limiter_gain,
            }
        )

    manifest_path = phase4_dir / "phase4_bwe_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mixed_drop = np.array([row["mixed_drop_db"] for row in rows], dtype=np.float64)
    summary = {
        "phase": 4,
        "status": "bwe_mixed_and_gain_matched",
        "showcase_count": len(rows),
        "sample_rate_hz": PHASE2_SR,
        "target_drop_8_11_minus_2_8_db": target_drop_db,
        "bwe_parameters": str(args.bwe_parameters) if args.bwe_parameters else None,
        "post_bwe_highband_trim_db": post_bwe_highband_trim_db,
        "mean_mixed_drop_8_11_minus_2_8_db": float(mixed_drop.mean()),
        "std_mixed_drop_db": float(mixed_drop.std()),
        "phase4_dir": str(phase4_dir),
        "outputs_dir": str(output_dir),
        "manifest": str(manifest_path),
        "notes": [
            "This is still an offline prototype output.",
            "Generation pipeline files and original showcase WAVs are unchanged.",
        ],
    }
    (phase4_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def collect_psd(paths: list[Path], target_sr: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = []
    psd_rows = []
    freqs_ref = None
    for path in paths:
        y, sr = read_mono_native(path)
        y = resample_audio(y, sr, target_sr)
        freqs, _times, zxx = signal.stft(
            y,
            fs=target_sr,
            nperseg=N_FFT,
            noverlap=N_FFT - HOP,
            nfft=N_FFT,
            boundary=None,
            padded=False,
        )
        power = np.abs(zxx) ** 2
        db = 10.0 * np.log10(power.mean(axis=1) + EPS)
        if freqs_ref is None:
            freqs_ref = freqs
        elif not np.array_equal(freqs_ref, freqs):
            raise SystemExit("Frequency bins differ during phase 5")
        psd_rows.append(db)
        rows.append(
            {
                "path": path,
                "duration_s": len(y) / target_sr,
                "peak": float(np.max(np.abs(y))),
                "band_2_8khz_db": band_mean(db, freqs, 2000.0, 8000.0),
                "band_8_11khz_db": band_mean(db, freqs, 8000.0, 11025.0),
            }
        )
    return freqs_ref, np.stack(psd_rows), rows


def run_phase5(
    args: argparse.Namespace, bwe_dir: Path, showcase_wavs: list[Path]
) -> dict:
    phase4_outputs = bwe_dir / f"phase4_bwe_mixed_24k{args.trial_suffix}/outputs"
    if not phase4_outputs.exists():
        raise SystemExit(f"Run phase 4 first; missing {phase4_outputs}")

    phase5_dir = bwe_dir / f"phase5_retest{args.trial_suffix}"
    if phase5_dir.exists():
        raise SystemExit(f"output dir already exists, refusing to overwrite: {phase5_dir}")
    figures_dir = phase5_dir / "figures"
    per_seed_dir = figures_dir / "per_seed"
    figures_dir.mkdir(parents=True)
    per_seed_dir.mkdir(parents=True)

    with args.training_pool_manifest.open(newline="") as f:
        training_rows = list(csv.DictReader(f))
    real_wavs = [Path.cwd() / row["audio_path"] for row in training_rows]
    bwe_wavs = [
        phase4_outputs / path.parent.name / "audio_bwe_24k.wav"
        for path in showcase_wavs
    ]
    for path in real_wavs + showcase_wavs + bwe_wavs:
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"missing or empty phase 5 input: {path}")

    freqs, real_psd, real_rows = collect_psd(real_wavs, PHASE2_SR)
    _freqs, before_psd, before_rows = collect_psd(showcase_wavs, PHASE2_SR)
    _freqs, after_psd, after_rows = collect_psd(bwe_wavs, PHASE2_SR)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mean_std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return arr.mean(axis=0), arr.std(axis=0)

    real_mean, real_std = mean_std(real_psd)
    before_mean, before_std = mean_std(before_psd)
    after_mean, after_std = mean_std(after_psd)

    plt.figure(figsize=(12, 7))
    plt.plot(freqs, real_mean, label="real rain training mean", linewidth=2)
    plt.fill_between(freqs, real_mean - real_std, real_mean + real_std, alpha=0.12)
    plt.plot(freqs, before_mean, label="before BWE showcase mean", linewidth=2)
    plt.fill_between(freqs, before_mean - before_std, before_mean + before_std, alpha=0.12)
    plt.plot(freqs, after_mean, label="after BWE showcase mean", linewidth=2)
    plt.fill_between(freqs, after_mean - after_std, after_mean + after_std, alpha=0.12)
    plt.axvspan(8000, 11025, color="gold", alpha=0.18, label="8-11 kHz target band")
    plt.xlim(0, PHASE2_SR / 2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean STFT power (dB)")
    plt.title("BWE retest: before / after / real mean spectrum")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    mean_spectrum_path = figures_dir / "phase5_mean_spectrum_before_after_real.png"
    plt.savefig(mean_spectrum_path, dpi=180)
    plt.close()

    groups = {
        "real": real_rows,
        "before": before_rows,
        "after_bwe": after_rows,
    }
    band_rows = []
    for name, rows in groups.items():
        b2 = np.array([row["band_2_8khz_db"] for row in rows])
        b8 = np.array([row["band_8_11khz_db"] for row in rows])
        band_rows.append(
            {
                "group": name,
                "count": len(rows),
                "mean_2_8khz_db": float(b2.mean()),
                "std_2_8khz_db": float(b2.std()),
                "mean_8_11khz_db": float(b8.mean()),
                "std_8_11khz_db": float(b8.std()),
                "drop_8_11_minus_2_8_db": float((b8 - b2).mean()),
            }
        )

    labels = ["2-8 kHz", "8-11 kHz", "drop 8-11 minus 2-8"]
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(10, 6))
    for offset, row in zip([-width, 0, width], band_rows):
        vals = [
            row["mean_2_8khz_db"],
            row["mean_8_11khz_db"],
            row["drop_8_11_minus_2_8_db"],
        ]
        plt.bar(x + offset, vals, width, label=row["group"])
    plt.xticks(x, labels)
    plt.ylabel("dB")
    plt.title("BWE retest band metrics")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    band_plot_path = figures_dir / "phase5_band_metrics_before_after_real.png"
    plt.savefig(band_plot_path, dpi=180)
    plt.close()

    per_seed_rows = []
    for before_path, after_path, before, after in zip(
        showcase_wavs, bwe_wavs, before_rows, after_rows
    ):
        seed = before_path.parent.name
        per_seed_rows.append(
            {
                "seed": seed,
                "before_path": str(before_path),
                "after_path": str(after_path),
                "before_2_8khz_db": before["band_2_8khz_db"],
                "before_8_11khz_db": before["band_8_11khz_db"],
                "before_drop_db": before["band_8_11khz_db"] - before["band_2_8khz_db"],
                "after_2_8khz_db": after["band_2_8khz_db"],
                "after_8_11khz_db": after["band_8_11khz_db"],
                "after_drop_db": after["band_8_11khz_db"] - after["band_2_8khz_db"],
            }
        )
        _freqs, before_psd_one, _ = collect_psd([before_path], PHASE2_SR)
        _freqs, after_psd_one, _ = collect_psd([after_path], PHASE2_SR)
        plt.figure(figsize=(10, 5))
        plt.plot(freqs, before_psd_one[0], label="before")
        plt.plot(freqs, after_psd_one[0], label="after BWE")
        plt.plot(freqs, real_mean, label="real mean", alpha=0.7)
        plt.axvspan(8000, 11025, color="gold", alpha=0.18)
        plt.xlim(0, PHASE2_SR / 2)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Mean STFT power (dB)")
        plt.title(f"{seed}: before vs after BWE")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(per_seed_dir / f"{seed}_before_after_real.png", dpi=160)
        plt.close()

    band_csv = phase5_dir / "phase5_band_metrics.csv"
    with band_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(band_rows[0].keys()))
        writer.writeheader()
        writer.writerows(band_rows)

    per_seed_csv = phase5_dir / "phase5_per_seed_metrics.csv"
    with per_seed_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    real_8_11 = band_rows[0]["mean_8_11khz_db"]
    before_8_11 = band_rows[1]["mean_8_11khz_db"]
    after_8_11 = band_rows[2]["mean_8_11khz_db"]
    before_gap = before_8_11 - real_8_11
    after_gap = after_8_11 - real_8_11
    improvement = abs(before_gap) - abs(after_gap)
    passes_band_gate = abs(after_gap) <= 5.0

    summary = {
        "phase": 5,
        "status": "bwe_retest_complete",
        "real_count": len(real_rows),
        "before_count": len(before_rows),
        "after_count": len(after_rows),
        "real_mean_8_11khz_db": real_8_11,
        "before_mean_8_11khz_db": before_8_11,
        "after_mean_8_11khz_db": after_8_11,
        "before_gap_vs_real_8_11_db": before_gap,
        "after_gap_vs_real_8_11_db": after_gap,
        "absolute_gap_improvement_db": improvement,
        "passes_8_11_band_gate_pm5db": passes_band_gate,
        "phase5_dir": str(phase5_dir),
        "figures": {
            "mean_spectrum": str(mean_spectrum_path),
            "band_metrics": str(band_plot_path),
            "per_seed_dir": str(per_seed_dir),
        },
        "csv": {
            "band_metrics": str(band_csv),
            "per_seed_metrics": str(per_seed_csv),
        },
        "notes": [
            "Subjective A/B listening is not automated in this phase.",
            "No production pipeline files were modified.",
        ],
    }
    (phase5_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (phase5_dir / "REPORT.md").write_text(
        "# BWE phase 5 retest\n\n"
        f"- Real samples: {len(real_rows)}\n"
        f"- Before samples: {len(before_rows)}\n"
        f"- After-BWE samples: {len(after_rows)}\n"
        f"- Real mean 8-11 kHz: {real_8_11:.2f} dB\n"
        f"- Before mean 8-11 kHz: {before_8_11:.2f} dB\n"
        f"- After-BWE mean 8-11 kHz: {after_8_11:.2f} dB\n"
        f"- Before gap vs real: {before_gap:.2f} dB\n"
        f"- After gap vs real: {after_gap:.2f} dB\n"
        f"- Absolute gap improvement: {improvement:.2f} dB\n"
        f"- Passes +/-5 dB 8-11 kHz gate: {passes_band_gate}\n\n"
        "Figures:\n"
        f"- `{mean_spectrum_path}`\n"
        f"- `{band_plot_path}`\n"
        f"- `{per_seed_dir}`\n"
    )
    return summary


def main() -> None:
    args = parse_args()
    bwe_dir = Path(__file__).resolve().parents[1]
    ensure_inside(args.output_dir, bwe_dir)
    ensure_inside(args.figures_dir, bwe_dir)

    showcase_wavs = sorted(args.showcase_dir.glob("seed_*_generated/audio.wav"))
    if len(showcase_wavs) != 10:
        raise SystemExit(
            f"Expected 10 showcase WAVs under {args.showcase_dir}, "
            f"found {len(showcase_wavs)}"
        )

    if not args.training_pool_manifest.exists():
        raise SystemExit(
            f"Training pool manifest not found: {args.training_pool_manifest}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == 1:
        target = run_phase1(args, bwe_dir)
        print(json.dumps(target, indent=2))
        return

    if args.phase == 2:
        summary = run_phase2(args, bwe_dir, showcase_wavs)
        print(json.dumps(summary, indent=2))
        return

    if args.phase == 3:
        summary = run_phase3(args, bwe_dir)
        print(json.dumps(summary, indent=2))
        return

    if args.phase == 4:
        summary = run_phase4(args, bwe_dir)
        print(json.dumps(summary, indent=2))
        return

    if args.phase == 5:
        summary = run_phase5(args, bwe_dir, showcase_wavs)
        print(json.dumps(summary, indent=2))
        return

    plan = {
        "phase": 0,
        "status": "workspace_ready",
        "showcase_count": len(showcase_wavs),
        "showcase_inputs": [str(path) for path in showcase_wavs],
        "training_pool_manifest": str(args.training_pool_manifest),
        "output_dir": str(args.output_dir),
        "figures_dir": str(args.figures_dir),
        "writes_pipeline_files": False,
        "notes": [
            "Phase 0 only validates isolated inputs and output locations.",
            "BWE DSP is intentionally not implemented in this phase.",
        ],
    }
    plan_path = bwe_dir / "phase0_workspace_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
