from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from app.pipelines.full_pipeline import process_audio_batch
from app.schemas.audio import AudioMetadata
from app.services.environment_loader import clean_environment_data, load_environment_data
from app.services.storage_service import append_processing_log, save_feature_table


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def discover_audio_files(audio_dir: str | Path) -> list[Path]:
    """Discover audio files under a directory."""
    root = Path(audio_dir)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS and path.is_file())


def build_audio_metadata_list(audio_files: list[Path]) -> list[AudioMetadata]:
    """Build minimal metadata records from discovered files."""
    metadata_list: list[AudioMetadata] = []
    for path in audio_files:
        metadata_list.append(
            AudioMetadata(
                audio_id=path.stem,
                file_path=path,
                recorded_at=datetime.fromtimestamp(path.stat().st_mtime),
            )
        )
    return metadata_list


def run_batch(
    audio_dir: str | Path,
    environment_file: str | Path | None = None,
    settings=None,
    save_spectrogram: bool = False,
    save_segment_embeddings: bool = True,
) -> pd.DataFrame:
    """Run the feature extraction pipeline over a directory of audio files."""
    settings = settings or get_settings()
    audio_files = discover_audio_files(audio_dir)
    metadata_list = build_audio_metadata_list(audio_files)

    environment_df = None
    if environment_file:
        environment_df = clean_environment_data(load_environment_data(environment_file))

    results = process_audio_batch(
        metadata_list,
        environment_df=environment_df,
        settings=settings,
        save_spectrogram=save_spectrogram,
        save_segment_embeddings=save_segment_embeddings,
    )
    result_df = pd.DataFrame(results)

    feature_output_path = settings.features_dir / "audio_feature_table.csv"
    log_output_path = settings.metadata_dir / "processing_log.jsonl"
    save_feature_table(result_df, feature_output_path)

    for record in results:
        append_processing_log(record, log_output_path)

    return result_df
