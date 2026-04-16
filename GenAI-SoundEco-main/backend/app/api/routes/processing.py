from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import get_settings
from app.pipelines.batch_pipeline import run_batch
from app.pipelines.full_pipeline import process_single_audio
from app.schemas.api import (
    BatchProcessRequest,
    BatchProcessResponse,
    HealthResponse,
    ProcessAudioResponse,
    ResultLookupResponse,
)
from app.schemas.audio import AudioMetadata


router = APIRouter(tags=["processing"])

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse()


def _lookup_latest_result(audio_id: str, log_path: Path) -> dict | None:
    if not log_path.exists():
        return None

    with log_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("audio_id") == audio_id:
            return record
    return None


@router.post("/process/file", response_model=ProcessAudioResponse)
def process_file(
    file: UploadFile = File(...),
    save_spectrogram: bool = Query(default=False),
    save_segment_embeddings: bool = Query(default=True),
) -> ProcessAudioResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {suffix or 'unknown'}")

    settings = get_settings()
    upload_dir = settings.interim_dir / "api_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    temp_path = Path(
        tempfile.mkstemp(
            prefix="upload_",
            suffix=suffix,
            dir=upload_dir,
        )[1]
    )

    try:
        with temp_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        audio_id = Path(file.filename or temp_path.name).stem
        audio_metadata = AudioMetadata(
            audio_id=audio_id,
            file_path=temp_path,
            recorded_at=datetime.utcnow(),
        )
        result = process_single_audio(
            audio_metadata=audio_metadata,
            settings=settings,
            save_spectrogram=save_spectrogram,
            save_segment_embeddings=save_segment_embeddings,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {exc}") from exc
    finally:
        file.file.close()
        if temp_path.exists():
            temp_path.unlink()

    if result.get("status") != "success":
        raise HTTPException(
            status_code=422,
            detail=result.get("error_message", "Audio processing failed."),
        )

    return ProcessAudioResponse(
        audio_id=result["audio_id"],
        status=result["status"],
        duration=result.get("duration"),
        sample_rate=result.get("sample_rate"),
        num_local_segments=result.get("num_local_segments"),
        num_vggish_embeddings=result.get("num_vggish_embeddings"),
        segment_embedding_path=result.get("segment_embedding_path"),
        mel_spectrogram_path=result.get("mel_spectrogram_path"),
        feature_row=result,
    )


@router.post("/process/batch", response_model=BatchProcessResponse)
def process_batch(request: BatchProcessRequest) -> BatchProcessResponse:
    audio_dir = Path(request.audio_dir)
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Audio directory does not exist: {audio_dir}")

    if request.environment_file:
        environment_file = Path(request.environment_file)
        if not environment_file.exists() or not environment_file.is_file():
            raise HTTPException(status_code=400, detail=f"Environment file does not exist: {environment_file}")
    else:
        environment_file = None

    settings = get_settings()

    try:
        result_df = run_batch(
            audio_dir=audio_dir,
            environment_file=environment_file,
            settings=settings,
            save_spectrogram=request.save_spectrogram,
            save_segment_embeddings=request.save_segment_embeddings,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {exc}") from exc

    records = result_df.to_dict(orient="records")
    success_count = sum(1 for record in records if record.get("status") == "success")
    failure_count = len(records) - success_count

    return BatchProcessResponse(
        audio_dir=str(audio_dir),
        total_files=len(records),
        success_count=success_count,
        failure_count=failure_count,
        output_feature_table=str(settings.features_dir / "audio_feature_table.csv"),
        output_processing_log=str(settings.metadata_dir / "processing_log.jsonl"),
        results=records,
    )


@router.get("/results/{audio_id}", response_model=ResultLookupResponse)
def get_result(audio_id: str) -> ResultLookupResponse:
    settings = get_settings()
    result = _lookup_latest_result(audio_id=audio_id, log_path=settings.metadata_dir / "processing_log.jsonl")
    return ResultLookupResponse(
        audio_id=audio_id,
        found=result is not None,
        result=result,
    )
