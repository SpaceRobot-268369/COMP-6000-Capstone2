#!/usr/bin/env python3
"""Build a CLAP-first site-weather retrieval batch.

This is the second MVP pass after the env-only audit. Env metadata proposes
where to look, but CLAP/audio embeddings decide the audible weather label and
ranking.

Run this on the server, not on a local laptop. It downloads only selected S3
coarse webm chunks, exports short WAV windows, embeds those windows with CLAP,
and writes a retrieval manifest for listening audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import wave
from pathlib import Path
from typing import Iterable

import numpy as np


POLICY_VERSION = "site_weather_clap_retrieval_v0.2"
DEFAULT_BUCKET = "eco-acoustic-data.store.adelaideuni.cloud"
DEFAULT_SOURCE_PREFIX = "dataset/original/site_257_bowra-dry-a/downloaded_clips"
COARSE_CLIP_SECONDS = 300.0
CLAP_ANALYSIS_SAMPLE_RATE_HZ = 48000
LAYER_D_TARGET_SAMPLE_RATE_HZ = 22050

S3_LISTING_RE = re.compile(
    r"^\S+\s+\S+\s+(?P<size>\d+)\s+"
    r"(?P<path>site_257_item_(?P<item_id>\d+)/"
    r"site_257_item_(?P=item_id)_clip_(?P<clip_num>\d+)\.webm)$"
)

WEATHER_PROMPTS = {
    "rain": [
        "natural rain ambience",
        "steady forest rain",
        "heavy rain in nature",
        "light rain ambience",
    ],
    "wind": [
        "natural wind ambience",
        "wind through trees",
        "strong forest wind",
        "light breeze in trees",
    ],
    "thunder": [
        "natural thunder",
        "distant rolling thunder",
        "thunderstorm ambience",
    ],
}

TARGET_ORDER = ["rain", "wind", "thunder"]

CONTAMINATION_PROMPTS = {
    "bird_or_insect": [
        "loud insects and cicadas",
        "birds calling",
        "frogs calling",
        "wildlife chorus",
    ],
    "human_or_machine": [
        "human speech",
        "traffic noise",
        "engine noise",
        "microphone handling noise",
    ],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except ValueError:
        return default


def parse_s3_listing(path: Path) -> dict[str, list[dict[str, object]]]:
    by_item: dict[str, list[dict[str, object]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = S3_LISTING_RE.match(line.strip())
            if not match:
                continue
            item_id = match.group("item_id")
            by_item.setdefault(item_id, []).append(
                {
                    "item_id": item_id,
                    "clip_num": int(match.group("clip_num")),
                    "path": match.group("path"),
                    "size_bytes": int(match.group("size")),
                }
            )

    for clips in by_item.values():
        clips.sort(key=lambda item: int(item["clip_num"]))
    return by_item


def stable_int(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def parse_target_quotas(text: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        target, _, value = item.partition("=")
        target = target.strip()
        if target not in WEATHER_PROMPTS:
            raise ValueError(f"Unknown target quota weather type: {target}")
        quotas[target] = int(value.strip())
    return quotas


def env_prior(row: dict[str, str], weather_type: str) -> float:
    precipitation = parse_float(row, "precipitation_mm")
    daily_precip = parse_float(row, "precipitation_daily_mm")
    wind = parse_float(row, "wind_speed_ms")
    wind_max = parse_float(row, "wind_max_ms")

    if weather_type == "rain":
        if precipitation >= 5:
            return 0.90
        if precipitation >= 2:
            return 0.75
        if precipitation > 0:
            return 0.45
        if daily_precip >= 2:
            return 0.25
        return 0.0

    if weather_type == "wind":
        wind_signal = max(wind, wind_max)
        if wind >= 6:
            return 0.80
        if wind >= 2:
            return 0.55
        if wind_signal >= 6:
            return 0.30
        return 0.0

    if weather_type == "thunder":
        # No direct thunder labels in current env data. Keep this weak.
        if precipitation >= 5 and max(wind, wind_max) >= 6:
            return 0.35
        if precipitation >= 5:
            return 0.20
        return 0.0

    return 0.0


def candidate_env_bucket(row: dict[str, str]) -> str:
    rain_prior = env_prior(row, "rain")
    wind_prior = env_prior(row, "wind")
    thunder_prior = env_prior(row, "thunder")
    if thunder_prior >= 0.35:
        return "storm_env_prior"
    if rain_prior >= wind_prior and rain_prior > 0:
        return "rain_env_prior"
    if wind_prior > 0:
        return "wind_env_prior"
    return "low_weather_prior"


def choose_windows(
    *,
    recording_id: str,
    seed_context: str,
    clips: list[dict[str, object]],
    recording_duration: float,
    window_seconds: float,
    windows_per_recording: int,
) -> Iterable[tuple[dict[str, object], float, float]]:
    usable = [
        clip
        for clip in clips
        if (int(clip["clip_num"]) - 1) * COARSE_CLIP_SECONDS + window_seconds
        <= recording_duration
    ]
    if not usable:
        usable = clips

    for variant in range(windows_per_recording):
        index = stable_int(recording_id, seed_context, "clap_retrieval", variant) % len(usable)
        coarse = usable[index]
        coarse_start = (int(coarse["clip_num"]) - 1) * COARSE_CLIP_SECONDS
        remaining = max(0.0, min(COARSE_CLIP_SECONDS, recording_duration - coarse_start))
        max_inner_start = max(0.0, remaining - window_seconds)
        inner_start = 0.0
        if max_inner_start > 0:
            inner_start = float(
                stable_int(recording_id, seed_context, variant, "offset")
                % int(max_inner_start + 1)
            )
        recording_start = coarse_start + inner_start
        yield coarse, inner_start, recording_start


def build_window_rows(
    *,
    items_rows: list[dict[str, str]],
    env_rows: list[dict[str, str]],
    listing: dict[str, list[dict[str, object]]],
    window_seconds: float,
    windows_per_recording: int,
    max_recordings_per_env_bucket: int,
    balanced: bool,
    target_quotas: dict[str, int],
    max_recordings_per_target: int,
) -> list[dict[str, object]]:
    items_by_id = {row["id"]: row for row in items_rows}
    eligible_env = [
        row
        for row in env_rows
        if row["recording_id"] in items_by_id and row["recording_id"] in listing
    ]

    if balanced:
        selected_env_targets: list[tuple[dict[str, str], str]] = []
        for target in TARGET_ORDER:
            if target_quotas.get(target, 0) <= 0:
                continue
            target_rows = [
                row
                for row in eligible_env
                if env_prior(row, target) > (0.0 if target != "thunder" else 0.19)
            ]
            target_rows.sort(
                key=lambda row: (
                    -env_prior(row, target),
                    row.get("recorded_date_utc", ""),
                    row["recording_id"],
                )
            )
            for row in target_rows[:max_recordings_per_target]:
                selected_env_targets.append((row, target))
        return build_rows_from_selected_env(
            selected_env_targets=selected_env_targets,
            items_by_id=items_by_id,
            listing=listing,
            window_seconds=window_seconds,
            windows_per_recording=windows_per_recording,
        )

    eligible_env.sort(
        key=lambda row: (
            candidate_env_bucket(row),
            -max(env_prior(row, weather) for weather in WEATHER_PROMPTS),
            row.get("recorded_date_utc", ""),
            row["recording_id"],
        )
    )

    bucket_counts: dict[str, int] = {}
    selected_env_targets: list[tuple[dict[str, str], str]] = []
    for row in eligible_env:
        bucket = candidate_env_bucket(row)
        if bucket == "low_weather_prior":
            continue
        if bucket_counts.get(bucket, 0) >= max_recordings_per_env_bucket:
            continue
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        selected_env_targets.append((row, "auto"))

    return build_rows_from_selected_env(
        selected_env_targets=selected_env_targets,
        items_by_id=items_by_id,
        listing=listing,
        window_seconds=window_seconds,
        windows_per_recording=windows_per_recording,
    )


def build_rows_from_selected_env(
    *,
    selected_env_targets: list[tuple[dict[str, str], str]],
    items_by_id: dict[str, dict[str, str]],
    listing: dict[str, list[dict[str, object]]],
    window_seconds: float,
    windows_per_recording: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    used_clip_ids: set[str] = set()
    for env_row, retrieval_target in selected_env_targets:
        recording_id = env_row["recording_id"]
        item = items_by_id[recording_id]
        recording_duration = parse_float(item, "duration_seconds")
        env_bucket = candidate_env_bucket(env_row)
        for coarse, inner_start, recording_start in choose_windows(
            recording_id=recording_id,
            seed_context=retrieval_target,
            clips=listing[recording_id],
            recording_duration=recording_duration,
            window_seconds=window_seconds,
            windows_per_recording=windows_per_recording,
        ):
            recording_end = recording_start + window_seconds
            clip_id = f"site257_{recording_id}_{int(recording_start):06d}_{int(recording_end):06d}"
            if clip_id in used_clip_ids:
                continue
            used_clip_ids.add(clip_id)
            rows.append(
                {
                    "clip_id": clip_id,
                    "site_id": item.get("site_id", "257"),
                    "recording_id": recording_id,
                    "item_id": recording_id,
                    "recorded_date_utc": env_row.get("recorded_date_utc", item.get("recorded_date", "")),
                    "sample_bin": env_row.get("sample_bin", item.get("sample_bin", "")),
                    "sample_local_date": env_row.get("sample_local_date", item.get("sample_local_date", "")),
                    "retrieval_target": retrieval_target,
                    "target_env_prior": env_prior(env_row, retrieval_target)
                    if retrieval_target != "auto"
                    else "",
                    "env_bucket": env_bucket,
                    "s3_key": coarse["path"],
                    "coarse_clip_num": f"{int(coarse['clip_num']):03d}",
                    "coarse_size_bytes": coarse["size_bytes"],
                    "coarse_inner_start_seconds": round(inner_start, 3),
                    "recording_start_offset_seconds": round(recording_start, 3),
                    "duration_seconds": window_seconds,
                    "precipitation_mm": parse_float(env_row, "precipitation_mm"),
                    "precipitation_daily_mm": parse_float(env_row, "precipitation_daily_mm"),
                    "wind_speed_ms": parse_float(env_row, "wind_speed_ms"),
                    "wind_max_ms": parse_float(env_row, "wind_max_ms"),
                    "humidity_pct": parse_float(env_row, "humidity_pct"),
                    "temperature_c": parse_float(env_row, "temperature_c"),
                }
            )

    return rows


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def add_wav_quality_fields(row: dict[str, object], wav_path: Path) -> None:
    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count)

    row["analysis_wav_sample_rate_hz"] = sample_rate
    row["analysis_wav_channels"] = channels
    row["analysis_wav_duration_seconds"] = round(frame_count / sample_rate, 3) if sample_rate else ""

    if sample_width != 2 or not frames:
        row["analysis_rms_dbfs"] = ""
        row["analysis_peak_dbfs"] = ""
        row["analysis_clipping_ratio"] = ""
        return

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        row["analysis_rms_dbfs"] = ""
        row["analysis_peak_dbfs"] = ""
        row["analysis_clipping_ratio"] = ""
        return

    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    clipping_ratio = float(np.mean(np.abs(samples) >= 0.999))
    row["analysis_rms_dbfs"] = round(20.0 * math.log10(max(rms, 1e-9)), 3)
    row["analysis_peak_dbfs"] = round(20.0 * math.log10(max(peak, 1e-9)), 3)
    row["analysis_clipping_ratio"] = round(clipping_ratio, 6)


def export_windows(
    rows: list[dict[str, object]],
    *,
    output_dir: Path,
    bucket: str,
    source_prefix: str,
    aws_region: str,
    mp3_previews: bool,
) -> None:
    wav_dir = output_dir / "windows_wav"
    preview_dir = output_dir / "previews"
    cache_dir = output_dir / "cache"
    wav_dir.mkdir(parents=True, exist_ok=True)
    if mp3_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        s3_key = str(row["s3_key"])
        source_uri = f"s3://{bucket}/{source_prefix}/{s3_key}"
        local_source = cache_dir / s3_key.replace("/", "__")
        wav_path = wav_dir / f"{index:04d}_{row['clip_id']}.wav"

        if not local_source.exists():
            run_command(
                [
                    "aws",
                    "s3",
                    "cp",
                    source_uri,
                    str(local_source),
                    "--only-show-errors",
                    "--region",
                    aws_region,
                ]
            )

        if not wav_path.exists():
            run_command(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    str(row["coarse_inner_start_seconds"]),
                    "-t",
                    str(row["duration_seconds"]),
                    "-i",
                    str(local_source),
                    "-ac",
                    "1",
                    "-ar",
                    str(CLAP_ANALYSIS_SAMPLE_RATE_HZ),
                    "-sample_fmt",
                    "s16",
                    str(wav_path),
                ]
            )

        row["wav_path"] = str(wav_path)
        row["source_s3_uri"] = source_uri
        row["analysis_asset_role"] = "clap_scoring_window"
        row["preview_asset_only"] = "true"
        row["layer_d_asset_status"] = "pending_22050hz_export"
        row["layer_d_target_sample_rate_hz"] = LAYER_D_TARGET_SAMPLE_RATE_HZ
        row["layer_d_target_channels"] = 1
        row["layer_d_recommended_format"] = "wav"
        add_wav_quality_fields(row, wav_path)

        if mp3_previews:
            mp3_path = preview_dir / f"{index:04d}_{row['clip_id']}.mp3"
            if not mp3_path.exists():
                run_command(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(wav_path),
                        "-b:a",
                        "96k",
                        str(mp3_path),
                    ]
                )
            row["preview_path"] = str(mp3_path)


def cosine_similarity_matrix(audio_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    audio = audio_embeddings / np.linalg.norm(audio_embeddings, axis=1, keepdims=True)
    text = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    return audio @ text.T


def load_clap_model():
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    return model


def score_with_clap(rows: list[dict[str, object]], *, audio_batch_size: int) -> None:
    model = load_clap_model()
    audio_files = [str(row["wav_path"]) for row in rows]
    audio_embedding_batches = []
    audio_batch_size = max(1, audio_batch_size)
    for start in range(0, len(audio_files), audio_batch_size):
        batch = audio_files[start : start + audio_batch_size]
        audio_embedding_batches.append(
            np.asarray(model.get_audio_embedding_from_filelist(x=batch, use_tensor=False))
        )
    audio_embeddings = np.vstack(audio_embedding_batches)

    prompt_items: list[tuple[str, str, str]] = []
    for weather_type, prompts in WEATHER_PROMPTS.items():
        for prompt in prompts:
            prompt_items.append(("weather", weather_type, prompt))
    for contamination_type, prompts in CONTAMINATION_PROMPTS.items():
        for prompt in prompts:
            prompt_items.append(("contamination", contamination_type, prompt))

    text_prompts = [prompt for _, _, prompt in prompt_items]
    try:
        text_embeddings = np.asarray(model.get_text_embedding(text_prompts, use_tensor=False))
    except TypeError:
        text_embeddings = np.asarray(model.get_text_embedding(text_prompts))
    similarities = cosine_similarity_matrix(audio_embeddings, text_embeddings)

    for row, scores in zip(rows, similarities):
        grouped: dict[str, list[float]] = {}
        for (_, label, _), score in zip(prompt_items, scores):
            grouped.setdefault(label, []).append(float(score))

        weather_scores = {
            weather_type: max(grouped.get(weather_type, [0.0]))
            for weather_type in WEATHER_PROMPTS
        }
        contamination_scores = {
            contamination_type: max(grouped.get(contamination_type, [0.0]))
            for contamination_type in CONTAMINATION_PROMPTS
        }
        top_weather = max(weather_scores, key=weather_scores.get)
        top_contamination = max(contamination_scores, key=contamination_scores.get)

        retrieval_target = str(row.get("retrieval_target", "auto"))
        scoring_weather = (
            retrieval_target if retrieval_target in WEATHER_PROMPTS else top_weather
        )
        target_weather_score = weather_scores[scoring_weather]
        other_weather_score = max(
            score
            for weather_type, score in weather_scores.items()
            if weather_type != scoring_weather
        )
        target_vs_other_weather_margin = target_weather_score - other_weather_score
        clap_weather_score = weather_scores[top_weather]
        contamination_score = contamination_scores[top_contamination]
        weather_margin = target_weather_score - contamination_score
        env_score = (
            parse_float({k: str(v) for k, v in row.items()}, "target_env_prior")
            if retrieval_target in WEATHER_PROMPTS
            else env_prior({k: str(v) for k, v in row.items()}, top_weather)
        )
        quality_score = max(0.0, min(1.0, 0.5 + weather_margin))
        final_score = (
            0.65 * target_weather_score
            + 0.20 * quality_score
            + 0.15 * env_score
            - 0.20 * max(0.0, contamination_score - target_weather_score)
            - 0.10 * max(0.0, other_weather_score - target_weather_score)
        )

        row["clap_rain_score"] = round(weather_scores["rain"], 6)
        row["clap_wind_score"] = round(weather_scores["wind"], 6)
        row["clap_thunder_score"] = round(weather_scores["thunder"], 6)
        row["contamination_label"] = top_contamination
        row["contamination_score"] = round(contamination_score, 6)
        row["clap_weather_label"] = top_weather
        row["clap_weather_score"] = round(clap_weather_score, 6)
        row["target_clap_score"] = round(target_weather_score, 6)
        row["target_vs_other_weather_margin"] = round(target_vs_other_weather_margin, 6)
        row["weather_margin"] = round(weather_margin, 6)
        row["env_prior_for_clap_label"] = round(env_score, 6)
        row["audio_quality_proxy"] = round(quality_score, 6)
        row["final_score"] = round(float(final_score), 6)
        row["gate_status"] = gate_status(
            retrieval_target=retrieval_target,
            scoring_weather=scoring_weather,
            env_score=env_score,
            target_weather_score=target_weather_score,
            contamination_score=contamination_score,
            weather_margin=weather_margin,
            target_vs_other_weather_margin=target_vs_other_weather_margin,
        )


def gate_status(
    *,
    retrieval_target: str,
    scoring_weather: str,
    env_score: float,
    target_weather_score: float,
    contamination_score: float,
    weather_margin: float,
    target_vs_other_weather_margin: float,
) -> str:
    # Absolute CLAP scores vary by model/checkpoint; the margin is the more
    # useful MVP signal. Thresholds should be tuned after the next listening audit.
    if scoring_weather == "thunder":
        if env_score < 0.20:
            return "reject_thunder_without_storm_prior"
        if weather_margin < 0.08:
            return "reject_thunder_without_clear_audio_confirmation"
    if retrieval_target in WEATHER_PROMPTS:
        if target_vs_other_weather_margin < -0.04:
            return "reject_target_outcompeted_by_other_weather"
        if target_vs_other_weather_margin < 0.02:
            return "maybe_target_confused_with_other_weather"
    if weather_margin < -0.02:
        return "reject_contamination_dominant"
    if weather_margin < 0.05:
        return "maybe_contamination_close"
    return "candidate"


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "final_rank",
        "final_score",
        "gate_status",
        "retrieval_target",
        "target_clap_score",
        "target_vs_other_weather_margin",
        "clap_weather_label",
        "clap_weather_score",
        "clap_rain_score",
        "clap_wind_score",
        "clap_thunder_score",
        "contamination_label",
        "contamination_score",
        "weather_margin",
        "env_prior_for_clap_label",
        "audio_quality_proxy",
        "analysis_asset_role",
        "analysis_wav_sample_rate_hz",
        "analysis_wav_channels",
        "analysis_wav_duration_seconds",
        "analysis_rms_dbfs",
        "analysis_peak_dbfs",
        "analysis_clipping_ratio",
        "preview_asset_only",
        "layer_d_asset_status",
        "layer_d_target_sample_rate_hz",
        "layer_d_target_channels",
        "layer_d_recommended_format",
        "target_env_prior",
        "env_bucket",
        "preview_path",
        "wav_path",
        "source_s3_uri",
        "s3_key",
        "site_id",
        "recording_id",
        "item_id",
        "recorded_date_utc",
        "sample_bin",
        "sample_local_date",
        "coarse_clip_num",
        "coarse_size_bytes",
        "coarse_inner_start_seconds",
        "recording_start_offset_seconds",
        "duration_seconds",
        "precipitation_mm",
        "precipitation_daily_mm",
        "wind_speed_ms",
        "wind_max_ms",
        "humidity_pct",
        "temperature_c",
        "human_weather_label",
        "human_intensity_label",
        "human_accept",
        "human_reject_reason",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def select_balanced_rows(
    rows: list[dict[str, object]], target_quotas: dict[str, int]
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for target in TARGET_ORDER:
        quota = target_quotas.get(target, 0)
        if quota <= 0:
            continue
        target_rows = [
            row for row in rows if str(row.get("retrieval_target", "")) == target
        ]
        target_rows.sort(
            key=lambda row: (
                str(row.get("gate_status")) != "candidate",
                str(row.get("gate_status")).startswith("reject"),
                -float(row.get("final_score", 0.0)),
            )
        )
        selected.extend(target_rows[:quota])
    return selected


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    summary: dict[str, object] = {
        "policy_version": POLICY_VERSION,
        "total_windows": len(rows),
        "gate_counts": {},
        "retrieval_target_counts": {},
        "clap_weather_label_counts": {},
        "env_bucket_counts": {},
    }
    for row in rows:
        for field, summary_key in [
            ("gate_status", "gate_counts"),
            ("retrieval_target", "retrieval_target_counts"),
            ("clap_weather_label", "clap_weather_label_counts"),
            ("env_bucket", "env_bucket_counts"),
        ]:
            value = str(row.get(field, ""))
            bucket = summary[summary_key]  # type: ignore[index]
            bucket[value] = bucket.get(value, 0) + 1
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-csv", type=Path, required=True)
    parser.add_argument("--env-csv", type=Path, required=True)
    parser.add_argument("--s3-listing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--window-seconds", type=float, default=15.0)
    parser.add_argument("--windows-per-recording", type=int, default=3)
    parser.add_argument("--max-recordings-per-env-bucket", type=int, default=40)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--target-quotas", default="rain=30,wind=30,thunder=12")
    parser.add_argument("--max-recordings-per-target", type=int, default=30)
    parser.add_argument("--aws-region", default="ap-southeast-2")
    parser.add_argument("--clap-audio-batch-size", type=int, default=8)
    parser.add_argument("--export-mp3-previews", action="store_true")
    parser.add_argument("--skip-clap", action="store_true")
    args = parser.parse_args()

    items_rows = read_csv(args.items_csv)
    env_rows = read_csv(args.env_csv)
    listing = parse_s3_listing(args.s3_listing)
    target_quotas = parse_target_quotas(args.target_quotas)
    rows = build_window_rows(
        items_rows=items_rows,
        env_rows=env_rows,
        listing=listing,
        window_seconds=args.window_seconds,
        windows_per_recording=args.windows_per_recording,
        max_recordings_per_env_bucket=args.max_recordings_per_env_bucket,
        balanced=args.balanced,
        target_quotas=target_quotas,
        max_recordings_per_target=args.max_recordings_per_target,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_windows(
        rows,
        output_dir=args.output_dir,
        bucket=args.bucket,
        source_prefix=args.source_prefix,
        aws_region=args.aws_region,
        mp3_previews=args.export_mp3_previews,
    )

    if not args.skip_clap:
        score_with_clap(rows, audio_batch_size=args.clap_audio_batch_size)
        if args.balanced:
            rows = select_balanced_rows(rows, target_quotas)
        else:
            rows.sort(key=lambda row: float(row.get("final_score", 0.0)), reverse=True)
    else:
        for row in rows:
            row["gate_status"] = "unscored_skip_clap"
            row["final_score"] = ""
            row["clap_weather_label"] = ""

    for index, row in enumerate(rows, start=1):
        row["final_rank"] = index
        row.setdefault("human_weather_label", "")
        row.setdefault("human_intensity_label", "")
        row.setdefault("human_accept", "")
        row.setdefault("human_reject_reason", "")
        row.setdefault("notes", "")

    write_manifest(args.output_dir / "retrieval_manifest.csv", rows)
    write_summary(args.output_dir / "summary.json", rows)
    (args.output_dir / "policy_version.txt").write_text(POLICY_VERSION + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} retrieval windows to {args.output_dir}")


if __name__ == "__main__":
    main()
