from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import torch

from app.schemas.audio import AudioMetadata, AudioSegment


class VGGishEncoder:
    """Thin wrapper around the future VGGish model integration."""

    def __init__(
        self,
        model_path=None,
        device: str = "cpu",
        embedding_dim: int = 128,
        repo: str = "harritaylor/torchvggish",
        entrypoint: str = "vggish",
        hub_dir: str | Path | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.embedding_dim = embedding_dim
        self.repo = repo
        self.entrypoint = entrypoint
        self.hub_dir = Path(hub_dir) if hub_dir else None
        self.model = None

    def load_model(self) -> None:
        """Load the pre-trained VGGish model from PyTorch Hub."""
        if self.model is not None:
            return
        if self.hub_dir is not None:
            self.hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(self.hub_dir))

        model = torch.hub.load(
            repo_or_dir=self.repo,
            model=self.entrypoint,
            source="github",
            trust_repo=True,
        )
        model.eval()
        if hasattr(model, "to"):
            model = model.to(self.device)
        self.model = model

    def extract_segment_embedding(self, waveform, sample_rate: int) -> list[float]:
        """Generate an embedding for a single audio segment."""
        raise NotImplementedError(
            "Waveform-level VGGish extraction is not implemented in this wrapper yet. "
            "Use extract_embeddings_from_file with an audio path."
        )

    def extract_embeddings_from_file(self, file_path: str | Path) -> list[list[float]]:
        """Generate segment-level embeddings for a full audio file."""
        self.load_model()
        assert self.model is not None

        with torch.no_grad():
            embeddings = self.model.forward(str(file_path))

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu()

        rows = embeddings.tolist()
        if rows and isinstance(rows[0], float):
            rows = [rows]
        return [[float(value) for value in row] for row in rows]

    def extract_embeddings(self, segments: Sequence[AudioSegment], sample_rate: int) -> list[list[float]]:
        """Generate embeddings for all segments from one audio file."""
        return [self.extract_segment_embedding(segment.waveform, sample_rate) for segment in segments]

    def extract_embeddings_for_audio(self, audio_metadata: AudioMetadata) -> list[list[float]]:
        """Generate embeddings using the audio file path from metadata."""
        return self.extract_embeddings_from_file(audio_metadata.file_path)

    def aggregate_embeddings(self, embeddings: Sequence[Sequence[float]], method: str = "mean") -> list[float]:
        """Aggregate segment-level embeddings into a file-level representation."""
        if not embeddings:
            return [0.0] * self.embedding_dim
        if method not in {"mean", "max"}:
            raise ValueError(f"Unsupported aggregation method: {method}")

        aggregated = []
        for index in range(len(embeddings[0])):
            column = [embedding[index] for embedding in embeddings]
            if method == "mean":
                aggregated.append(sum(column) / len(column))
            else:
                aggregated.append(max(column))
        return aggregated
