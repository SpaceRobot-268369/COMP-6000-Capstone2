"""AudioSet tagger adapters for E-B weather analysis.

PANNs CNN14 is the first wired AudioSet tagger. AST can also be used as a
conservative guard. Scorers degrade safely when packages/checkpoints are
missing so local lightweight checks do not require AI runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


AUDIOMODEL_SCORE_KEYS = (
    "rain",
    "wind",
    "thunder",
    "bio_contamination",
    "human_machine_contamination",
)


@dataclass(frozen=True)
class AudioSetScoreResult:
    scores: dict[str, float]
    available: bool
    backend: str
    warnings: list[str]
    raw: dict[str, Any]


class AudioSetScorer(Protocol):
    backend: str

    def score_window(self, samples: np.ndarray, sample_rate: int) -> AudioSetScoreResult:
        """Score one mono audio window."""


class UnavailableAudioSetScorer:
    backend = "audioset_unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def score_window(self, samples: np.ndarray, sample_rate: int) -> AudioSetScoreResult:
        return AudioSetScoreResult(
            scores={key: 0.0 for key in AUDIOMODEL_SCORE_KEYS},
            available=False,
            backend=self.backend,
            warnings=["audioset_scores_unavailable"],
            raw={"reason": self.reason},
        )


class PannsScorer:
    """PANNs CNN14 AudioSet scorer."""

    backend = "panns_cnn14"

    def __init__(self) -> None:
        try:
            from panns_inference import AudioTagging, labels
        except ModuleNotFoundError as exc:
            raise RuntimeError("panns_inference is not installed") from exc

        self.sample_rate = 32000
        self._labels = list(labels)
        self._label_to_index = {label: index for index, label in enumerate(self._labels)}
        self._model = AudioTagging(checkpoint_path=None, device=_torch_device())

    def score_window(self, samples: np.ndarray, sample_rate: int) -> AudioSetScoreResult:
        audio = _resample_linear(samples.astype(np.float32, copy=False), sample_rate, self.sample_rate)
        if len(audio) == 0:
            audio = np.zeros(1, dtype=np.float32)

        clipwise_output, _embedding = self._model.inference(audio[None, :].astype(np.float32))
        scores = clipwise_output[0]
        raw_weather = {
            label: _score_label(scores, self._label_to_index, label)
            for label in (
                "Rain",
                "Raindrop",
                "Rain on surface",
                "Wind",
                "Wind noise (microphone)",
                "Thunder",
                "Thunderstorm",
                "Bird",
                "Bird vocalization, bird call, bird song",
                "Insect",
                "Speech",
                "Vehicle",
                "Engine",
                "Machinery",
            )
        }
        grouped = {
            "rain": max(
                raw_weather["Rain"],
                raw_weather["Raindrop"],
                raw_weather["Rain on surface"],
            ),
            "wind": max(
                raw_weather["Wind"],
                raw_weather["Wind noise (microphone)"],
            ),
            "thunder": max(
                raw_weather["Thunder"],
                raw_weather["Thunderstorm"],
            ),
            "bio_contamination": max(
                raw_weather["Bird"],
                raw_weather["Bird vocalization, bird call, bird song"],
                raw_weather["Insect"],
            ),
            "human_machine_contamination": max(
                raw_weather["Speech"],
                raw_weather["Vehicle"],
                raw_weather["Engine"],
                raw_weather["Machinery"],
            ),
        }
        return AudioSetScoreResult(
            scores={key: float(grouped.get(key, 0.0)) for key in AUDIOMODEL_SCORE_KEYS},
            available=True,
            backend=self.backend,
            warnings=[],
            raw={
                "weather_labels": {
                    key: round(float(value), 6) for key, value in raw_weather.items()
                },
                "top_labels": _top_labels(scores, self._labels, limit=10),
            },
        )


class AstScorer:
    """Hugging Face AST AudioSet scorer used as a conservative guard."""

    backend = "ast_audioset"

    def __init__(self) -> None:
        try:
            import torch
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers/torch AST dependencies are not installed") from exc

        self._torch = torch
        self.sample_rate = 16000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self._extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self._model = ASTForAudioClassification.from_pretrained(model_name).to(self.device)
        self._model.eval()
        id_to_label = self._model.config.id2label
        self._labels = [id_to_label[index] for index in range(len(id_to_label))]

    def score_window(self, samples: np.ndarray, sample_rate: int) -> AudioSetScoreResult:
        audio = _resample_linear(samples.astype(np.float32, copy=False), sample_rate, self.sample_rate)
        if len(audio) == 0:
            audio = np.zeros(1, dtype=np.float32)

        inputs = self._extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0]
            probs = self._torch.sigmoid(logits).detach().cpu().numpy()

        raw_weather = {
            label: _score_label_fuzzy(probs, self._labels, label)
            for label in (
                "Rain",
                "Raindrop",
                "Rain on surface",
                "Wind",
                "Wind noise (microphone)",
                "Thunder",
                "Thunderstorm",
                "Bird",
                "Bird vocalization, bird call, bird song",
                "Insect",
                "Speech",
                "Vehicle",
                "Engine",
                "Machinery",
            )
        }
        grouped = {
            "rain": max(
                raw_weather["Rain"],
                raw_weather["Raindrop"],
                raw_weather["Rain on surface"],
            ),
            "wind": max(
                raw_weather["Wind"],
                raw_weather["Wind noise (microphone)"],
            ),
            "thunder": max(
                raw_weather["Thunder"],
                raw_weather["Thunderstorm"],
            ),
            "bio_contamination": max(
                raw_weather["Bird"],
                raw_weather["Bird vocalization, bird call, bird song"],
                raw_weather["Insect"],
            ),
            "human_machine_contamination": max(
                raw_weather["Speech"],
                raw_weather["Vehicle"],
                raw_weather["Engine"],
                raw_weather["Machinery"],
            ),
        }
        return AudioSetScoreResult(
            scores={key: float(grouped.get(key, 0.0)) for key in AUDIOMODEL_SCORE_KEYS},
            available=True,
            backend=self.backend,
            warnings=[],
            raw={
                "weather_labels": {
                    key: round(float(value), 6) for key, value in raw_weather.items()
                },
                "top_labels": _top_labels(probs, self._labels, limit=10),
            },
        )


def build_audioset_scorer(backend: str) -> AudioSetScorer:
    if backend == "none":
        return UnavailableAudioSetScorer("backend disabled")
    if backend == "panns":
        try:
            return PannsScorer()
        except Exception as exc:
            return UnavailableAudioSetScorer(f"PANNs backend unavailable: {exc}")
    if backend == "ast":
        try:
            return AstScorer()
        except Exception as exc:
            return UnavailableAudioSetScorer(f"AST backend unavailable: {exc}")
    return UnavailableAudioSetScorer(f"unknown backend: {backend}")


def _torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    if len(samples) == 0:
        return samples.astype(np.float32, copy=False)

    duration_s = len(samples) / float(source_rate)
    target_len = max(1, int(round(duration_s * target_rate)))
    source_x = np.linspace(0.0, duration_s, num=len(samples), endpoint=False)
    target_x = np.linspace(0.0, duration_s, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def _score_label(scores: np.ndarray, label_to_index: dict[str, int], label: str) -> float:
    index = label_to_index.get(label)
    if index is None:
        return 0.0
    return float(scores[index])


def _score_label_fuzzy(scores: np.ndarray, labels: list[str], label: str) -> float:
    label_lower = label.lower()
    for index, candidate in enumerate(labels):
        if candidate.lower() == label_lower:
            return float(scores[index])
    for index, candidate in enumerate(labels):
        if label_lower in candidate.lower():
            return float(scores[index])
    return 0.0


def _top_labels(scores: np.ndarray, labels: list[str], limit: int) -> list[dict[str, float | str]]:
    top_indices = np.argsort(scores)[-limit:][::-1]
    return [
        {
            "label": labels[int(index)],
            "score": round(float(scores[int(index)]), 6),
        }
        for index in top_indices
    ]
