from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    model_dir: Path = project_root / "models"
    torch_hub_dir: Path = model_dir / "torch_hub"
    raw_audio_dir: Path = data_dir / "raw" / "audio"
    raw_weather_dir: Path = data_dir / "raw" / "weather"
    interim_dir: Path = data_dir / "interim"
    processed_dir: Path = data_dir / "processed"
    features_dir: Path = data_dir / "features"
    embeddings_dir: Path = data_dir / "embeddings"
    metadata_dir: Path = data_dir / "metadata"
    visualizations_dir: Path = data_dir / "visualizations"
    spectrograms_dir: Path = visualizations_dir / "spectrograms"

    sample_rate: int = 16000
    mono: bool = True
    segment_duration: float = 0.96
    hop_duration: float = 0.48
    min_duration_seconds: float = 0.50
    silence_threshold: float = 1e-6
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int | None = None
    n_mels: int = 64
    fmin: float = 0.0
    fmax: float | None = None
    n_mfcc: int = 13
    embedding_dim: int = 128
    embedding_pooling: str = "mean"
    max_time_diff_minutes: int = 60
    device: str = "cpu"
    vggish_model_path: Path | None = None
    vggish_repo: str = "harritaylor/torchvggish"
    vggish_entrypoint: str = "vggish"


def get_settings() -> Settings:
    return Settings()
