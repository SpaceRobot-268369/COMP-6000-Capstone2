#!/usr/bin/env python3
"""Validate possible site thunder clips with spectrogram/envelope features.

This is a conservative second-pass validator after CLAP retrieval. CLAP can put
thunder, storm wind, low rumble, and microphone overload close together; this
script checks whether the audio has the transient low-frequency burst shape that
is more consistent with thunder than continuous wind.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path

import numpy as np


POLICY_VERSION = "site_weather_thunder_validator_v0.2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except ValueError:
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM wav, got sample width {sample_width}")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def frame_audio(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    frame_size = max(512, int(sample_rate * 0.10))
    hop_size = max(256, int(sample_rate * 0.05))
    if samples.size < frame_size:
        padded = np.zeros(frame_size, dtype=np.float32)
        padded[: samples.size] = samples
        return padded[None, :], hop_size

    starts = range(0, samples.size - frame_size + 1, hop_size)
    frames = np.stack([samples[start : start + frame_size] for start in starts])
    return frames, hop_size


def band_ratio(
    power: np.ndarray,
    freqs: np.ndarray,
    low_hz: float,
    high_hz: float,
    *,
    total_low_hz: float = 20.0,
    total_high_hz: float = 6000.0,
) -> float:
    band_mask = (freqs >= low_hz) & (freqs < high_hz)
    total_mask = (freqs >= total_low_hz) & (freqs < total_high_hz)
    band = float(power[:, band_mask].sum())
    total = float(power[:, total_mask].sum())
    return band / max(total, 1e-12)


def longest_true_span(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def audio_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    clipping_ratio = float(np.mean(np.abs(samples) >= 0.999)) if samples.size else 0.0
    frames, hop_size = frame_audio(samples, sample_rate)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    windowed = frames * window[None, :]

    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    median_db = float(np.median(rms_db))
    peak_index = int(np.argmax(rms_db))
    peak_db = float(rms_db[peak_index])
    peak_to_median_db = peak_db - median_db

    active_threshold = median_db + max(6.0, min(14.0, peak_to_median_db * 0.45))
    active = rms_db >= active_threshold
    active_ratio = float(np.mean(active))
    longest_active_seconds = longest_true_span(active) * hop_size / sample_rate

    smooth_db = np.convolve(rms_db, np.ones(3) / 3.0, mode="same")
    onset_count = int(np.sum(np.diff(smooth_db) > 5.5))

    post_start = peak_index + int(0.5 * sample_rate / hop_size)
    post_end = min(len(rms_db), peak_index + int(2.5 * sample_rate / hop_size))
    if post_start < post_end:
        decay_drop_db = peak_db - float(np.median(rms_db[post_start:post_end]))
    else:
        decay_drop_db = 0.0

    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    low_ratio = band_ratio(spectrum, freqs, 20.0, 180.0)
    low_mid_ratio = band_ratio(spectrum, freqs, 180.0, 700.0)
    high_ratio = band_ratio(spectrum, freqs, 1500.0, 6000.0)

    norm_spectrum = spectrum / np.maximum(spectrum.sum(axis=1, keepdims=True), 1e-12)
    if norm_spectrum.shape[0] > 1:
        flux = np.maximum(np.diff(norm_spectrum, axis=0), 0.0).sum(axis=1)
        spectral_flux_mean = float(np.mean(flux))
        spectral_flux_p95 = float(np.percentile(flux, 95))
    else:
        spectral_flux_mean = 0.0
        spectral_flux_p95 = 0.0

    duration_seconds = samples.size / sample_rate if sample_rate else 0.0
    peak_time_seconds = peak_index * hop_size / sample_rate

    return {
        "validator_duration_seconds": round(duration_seconds, 3),
        "validator_peak_time_seconds": round(peak_time_seconds, 3),
        "validator_rms_median_dbfs": round(median_db, 3),
        "validator_rms_peak_dbfs": round(peak_db, 3),
        "validator_peak_to_median_db": round(peak_to_median_db, 3),
        "validator_active_ratio": round(active_ratio, 6),
        "validator_longest_active_seconds": round(longest_active_seconds, 3),
        "validator_onset_count": onset_count,
        "validator_decay_drop_db": round(decay_drop_db, 3),
        "validator_low_freq_ratio": round(low_ratio, 6),
        "validator_low_mid_ratio": round(low_mid_ratio, 6),
        "validator_high_freq_ratio": round(high_ratio, 6),
        "validator_spectral_flux_mean": round(spectral_flux_mean, 6),
        "validator_spectral_flux_p95": round(spectral_flux_p95, 6),
        "validator_clipping_ratio": round(clipping_ratio, 6),
    }


def metadata_support(row: dict[str, str]) -> dict[str, bool]:
    target = row.get("retrieval_target", "")
    clap_label = row.get("clap_weather_label", "")
    pool_category = row.get("pool_category", "")
    pool_label = row.get("pool_label", "")
    env_bucket = row.get("env_bucket", "")
    return {
        "target_thunder": target == "thunder",
        "clap_thunder": clap_label == "thunder",
        "storm_pool": "storm" in pool_category,
        "storm_env": env_bucket == "storm_env_prior",
        "mixed_weather_pool": pool_label == "rain+wind",
    }


def classify_validator(row: dict[str, str], features: dict[str, float]) -> dict[str, object]:
    clipping = float(features["validator_clipping_ratio"])
    peak_to_median = float(features["validator_peak_to_median_db"])
    active_ratio = float(features["validator_active_ratio"])
    low_ratio = float(features["validator_low_freq_ratio"])
    onset_count = float(features["validator_onset_count"])
    decay_drop = float(features["validator_decay_drop_db"])
    longest_active = float(features["validator_longest_active_seconds"])

    transient_score = clamp((peak_to_median - 8.0) / 12.0)
    low_score = clamp((low_ratio - 0.22) / 0.30)
    onset_score = clamp(onset_count / 2.0)
    decay_score = clamp((decay_drop - 4.0) / 10.0)

    if active_ratio < 0.02:
        burst_shape_score = 0.25
    elif active_ratio <= 0.45:
        burst_shape_score = 1.0
    elif active_ratio <= 0.70:
        burst_shape_score = 0.45
    else:
        burst_shape_score = 0.0

    continuity_penalty = clamp((longest_active - 5.0) / 5.0)
    clipping_penalty = clamp(clipping / 0.005)

    thunder_burst_score = (
        0.30 * transient_score
        + 0.25 * low_score
        + 0.20 * onset_score
        + 0.15 * decay_score
        + 0.10 * burst_shape_score
        - 0.20 * continuity_penalty
        - 0.35 * clipping_penalty
    )
    thunder_burst_score = clamp(thunder_burst_score)

    support = metadata_support(row)
    has_thunder_context = (
        support["target_thunder"]
        or support["clap_thunder"]
        or support["storm_pool"]
        or support["storm_env"]
    )
    has_only_mixed_context = support["mixed_weather_pool"] and not has_thunder_context
    overload_risk = clipping >= 0.002 or (clipping > 0.0 and active_ratio > 0.15)

    if overload_risk:
        label = "ambiguous_overload_or_thunder"
        reason = "low_frequency_burst_with_clipping_or_peak_risk"
    elif thunder_burst_score >= 0.72 and has_thunder_context:
        label = "possible_thunder_burst"
        reason = "transient_low_frequency_burst"
    elif thunder_burst_score >= 0.62 and has_only_mixed_context:
        label = "possible_storm_rumble_not_thunder"
        reason = "burst_shape_without_thunder_context"
    elif thunder_burst_score >= 0.45:
        label = "ambiguous_storm_rumble"
        reason = "partial_thunder_shape_but_not_decisive"
    else:
        label = "likely_wind_or_rain"
        reason = "continuous_or_weak_transient_shape"

    return {
        "validator_label": label,
        "validator_reason": reason,
        "validator_thunder_burst_score": round(thunder_burst_score, 6),
        "validator_has_thunder_context": has_thunder_context,
        "validator_has_only_mixed_context": has_only_mixed_context,
        "validator_policy_version": POLICY_VERSION,
    }


def should_validate(row: dict[str, str], all_rows: bool) -> bool:
    if all_rows:
        return True
    target = row.get("retrieval_target", "")
    clap_label = row.get("clap_weather_label", "")
    pool_category = row.get("pool_category", "")
    pool_label = row.get("pool_label", "")
    clipping = parse_float(row, "analysis_clipping_ratio")
    return (
        target == "thunder"
        or clap_label == "thunder"
        or "storm" in pool_category
        or pool_label == "rain+wind"
        or clipping >= 0.001
    )


def validate_rows(rows: list[dict[str, str]], *, all_rows: bool) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if not should_validate(row, all_rows):
            continue
        wav_path = Path(row.get("wav_path", ""))
        result: dict[str, object] = dict(row)
        if not wav_path.exists():
            result.update(
                {
                    "validator_label": "missing_audio",
                    "validator_reason": "wav_path_not_found",
                    "validator_policy_version": POLICY_VERSION,
                }
            )
            output.append(result)
            continue
        try:
            samples, sample_rate = load_wav(wav_path)
            features = audio_features(samples, sample_rate)
            result.update(features)
            result.update(classify_validator(row, features))
        except Exception as exc:  # Keep batch jobs inspectable instead of failing all rows.
            result.update(
                {
                    "validator_label": "validator_error",
                    "validator_reason": f"{type(exc).__name__}: {exc}",
                    "validator_policy_version": POLICY_VERSION,
                }
            )
        output.append(result)
    return output


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    summary: dict[str, object] = {
        "policy_version": POLICY_VERSION,
        "total_validated_rows": len(rows),
        "validator_label_counts": {},
        "validator_reason_counts": {},
    }
    for row in rows:
        for field, key in [
            ("validator_label", "validator_label_counts"),
            ("validator_reason", "validator_reason_counts"),
        ]:
            value = str(row.get(field, ""))
            counts = summary[key]  # type: ignore[index]
            counts[value] = counts.get(value, 0) + 1
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all-rows", action="store_true")
    args = parser.parse_args()

    rows = validate_rows(read_csv(args.manifest), all_rows=args.all_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "thunder_validator_manifest.csv", rows)
    write_summary(args.output_dir / "summary.json", rows)
    (args.output_dir / "policy_version.txt").write_text(POLICY_VERSION + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} thunder validator rows to {args.output_dir}")


if __name__ == "__main__":
    main()
