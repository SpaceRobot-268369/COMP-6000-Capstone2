from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from app.schemas.audio import AudioMetadata
from app.schemas.environment import AlignedEnvironmentRecord


def _minutes_between(left: datetime, right: datetime) -> float:
    return abs((left - right).total_seconds()) / 60.0


def match_environment_record(
    audio_metadata: AudioMetadata,
    env_df: pd.DataFrame | None,
    max_time_diff_minutes: int = 60,
) -> AlignedEnvironmentRecord:
    """Match an audio record to the closest environment record."""
    if env_df is None or env_df.empty or audio_metadata.recorded_at is None:
        return AlignedEnvironmentRecord(audio_id=audio_metadata.audio_id, matched=False)

    candidates = env_df.copy()
    if audio_metadata.location_id and "location_id" in candidates.columns:
        candidates = candidates[candidates["location_id"] == audio_metadata.location_id]
    if candidates.empty or "timestamp" not in candidates.columns:
        return AlignedEnvironmentRecord(audio_id=audio_metadata.audio_id, matched=False)

    timestamps = pd.to_datetime(candidates["timestamp"], errors="coerce")
    deltas = timestamps.apply(
        lambda value: math.inf if pd.isna(value) else _minutes_between(value.to_pydatetime(), audio_metadata.recorded_at)
    )
    match_index = deltas.idxmin()
    delta = deltas.loc[match_index]
    if math.isinf(delta) or delta > max_time_diff_minutes:
        return AlignedEnvironmentRecord(audio_id=audio_metadata.audio_id, matched=False)

    row = candidates.loc[match_index]
    return AlignedEnvironmentRecord(
        audio_id=audio_metadata.audio_id,
        matched=True,
        matched_timestamp=timestamps.loc[match_index].to_pydatetime(),
        time_diff_minutes=float(delta),
        temperature=row.get("temperature"),
        humidity=row.get("humidity"),
        wind_speed=row.get("wind_speed"),
        rainfall=row.get("rainfall"),
        pressure=row.get("pressure"),
    )


def build_aligned_feature_row(
    audio_metadata: AudioMetadata,
    acoustic_features: dict,
    embedding_vector: list[float],
    environment_record: AlignedEnvironmentRecord,
) -> dict:
    """Merge audio metadata, features, embeddings, and environment data into one row."""
    row = {
        "audio_id": audio_metadata.audio_id,
        "file_path": str(audio_metadata.file_path),
        "location_id": audio_metadata.location_id,
        "recorded_at": audio_metadata.recorded_at,
        "duration": audio_metadata.duration,
        "sample_rate": audio_metadata.sample_rate,
        "environment_matched": environment_record.matched,
        "environment_timestamp": environment_record.matched_timestamp,
        "environment_time_diff_minutes": environment_record.time_diff_minutes,
        "temperature": environment_record.temperature,
        "humidity": environment_record.humidity,
        "wind_speed": environment_record.wind_speed,
        "rainfall": environment_record.rainfall,
        "pressure": environment_record.pressure,
    }
    row.update(acoustic_features)
    row["embedding"] = embedding_vector
    return row
