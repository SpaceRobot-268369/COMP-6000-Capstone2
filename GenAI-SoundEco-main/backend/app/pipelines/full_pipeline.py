from __future__ import annotations

from app.core.config import get_settings
from app.schemas.audio import AudioMetadata
from app.services.acoustic_features import extract_acoustic_features
from app.services.audio_preprocessor import preprocess_audio
from app.services.audio_segmenter import segment_audio
from app.services.feature_aligner import build_aligned_feature_row, match_environment_record
from app.services.spectrogram_visualizer import save_mel_spectrogram
from app.services.storage_service import save_embedding_matrix
from app.services.vggish_encoder import VGGishEncoder


def process_single_audio(
    audio_metadata: AudioMetadata,
    environment_df=None,
    settings=None,
    save_spectrogram: bool = False,
    save_segment_embeddings: bool = True,
    encoder: VGGishEncoder | None = None,
) -> dict:
    """Run the full audio processing pipeline for one audio file."""
    settings = settings or get_settings()
    processed_audio = preprocess_audio(audio_metadata=audio_metadata, settings=settings)
    if not processed_audio.is_valid:
        return {
            "audio_id": audio_metadata.audio_id,
            "status": "failed",
            "error_message": processed_audio.error_message,
        }

    acoustic_features = extract_acoustic_features(
        waveform=processed_audio.waveform,
        sample_rate=processed_audio.sample_rate,
        settings=settings,
    )
    segments = segment_audio(
        waveform=processed_audio.waveform,
        sample_rate=processed_audio.sample_rate,
        segment_duration=settings.segment_duration,
        hop_duration=settings.hop_duration,
    )
    enriched_metadata = AudioMetadata(
        audio_id=audio_metadata.audio_id,
        file_path=audio_metadata.file_path,
        location_id=audio_metadata.location_id,
        recorded_at=audio_metadata.recorded_at,
        duration=processed_audio.duration,
        sample_rate=processed_audio.sample_rate,
    )
    encoder = encoder or VGGishEncoder(
        model_path=settings.vggish_model_path,
        device=settings.device,
        embedding_dim=settings.embedding_dim,
        repo=settings.vggish_repo,
        entrypoint=settings.vggish_entrypoint,
        hub_dir=settings.torch_hub_dir,
    )
    try:
        embeddings = encoder.extract_embeddings_for_audio(enriched_metadata)
        aggregated_embedding = encoder.aggregate_embeddings(embeddings, method=settings.embedding_pooling)
    except Exception as exc:
        return {
            "audio_id": audio_metadata.audio_id,
            "status": "failed",
            "error_message": f"VGGish extraction failed: {exc}",
        }
    environment_record = match_environment_record(
        audio_metadata=enriched_metadata,
        env_df=environment_df,
        max_time_diff_minutes=settings.max_time_diff_minutes,
    )
    row = build_aligned_feature_row(
        audio_metadata=enriched_metadata,
        acoustic_features=acoustic_features,
        embedding_vector=aggregated_embedding,
        environment_record=environment_record,
    )
    row["num_local_segments"] = len(segments)
    row["num_vggish_embeddings"] = len(embeddings)
    if save_segment_embeddings:
        embedding_output_path = settings.embeddings_dir / f"{audio_metadata.audio_id}_segment_embeddings.json"
        save_embedding_matrix(embeddings, embedding_output_path)
        row["segment_embedding_path"] = str(embedding_output_path)
    if save_spectrogram:
        spectrogram_path = settings.spectrograms_dir / f"{audio_metadata.audio_id}_mel.png"
        saved_path = save_mel_spectrogram(
            waveform=processed_audio.waveform,
            sample_rate=processed_audio.sample_rate or settings.sample_rate,
            output_path=spectrogram_path,
            settings=settings,
            title=audio_metadata.audio_id,
        )
        row["mel_spectrogram_path"] = str(saved_path)
    row["status"] = "success"
    return row


def process_audio_batch(
    audio_metadata_list: list[AudioMetadata],
    environment_df=None,
    settings=None,
    save_spectrogram: bool = False,
    save_segment_embeddings: bool = True,
) -> list[dict]:
    """Process a batch of audio metadata records."""
    settings = settings or get_settings()
    encoder = VGGishEncoder(
        model_path=settings.vggish_model_path,
        device=settings.device,
        embedding_dim=settings.embedding_dim,
        repo=settings.vggish_repo,
        entrypoint=settings.vggish_entrypoint,
        hub_dir=settings.torch_hub_dir,
    )
    return [
        process_single_audio(
            audio_metadata=audio_metadata,
            environment_df=environment_df,
            settings=settings,
            save_spectrogram=save_spectrogram,
            save_segment_embeddings=save_segment_embeddings,
            encoder=encoder,
        )
        for audio_metadata in audio_metadata_list
    ]
