from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class BatchProcessRequest(BaseModel):
    audio_dir: str
    environment_file: str | None = None
    save_spectrogram: bool = False
    save_segment_embeddings: bool = True


class ProcessAudioResponse(BaseModel):
    audio_id: str
    status: str
    duration: float | None = None
    sample_rate: float | None = None
    num_local_segments: int | None = None
    num_vggish_embeddings: int | None = None
    segment_embedding_path: str | None = None
    mel_spectrogram_path: str | None = None
    error_message: str | None = None
    feature_row: dict[str, Any] = Field(default_factory=dict)


class BatchProcessResponse(BaseModel):
    audio_dir: str
    total_files: int
    success_count: int
    failure_count: int
    output_feature_table: str
    output_processing_log: str
    results: list[dict[str, Any]] = Field(default_factory=list)


class ResultLookupResponse(BaseModel):
    audio_id: str
    found: bool
    result: dict[str, Any] | None = None
