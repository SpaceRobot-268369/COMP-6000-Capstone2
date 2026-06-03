"""Model-score adapters for E-B weather analysis.

The MVP starts with CLAP as the primary open-vocabulary analysis encoder. This
module keeps the scorer boundary explicit so the CLI can run without model
dependencies during lightweight checks, while serverB can provide real CLAP
scores when the dependencies are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


ELEMENTS = ("rain", "wind", "thunder")
CONTROL_GROUPS = ("none", "bio_contamination", "human_machine_contamination")
ALL_SCORE_KEYS = ELEMENTS + CONTROL_GROUPS


@dataclass(frozen=True)
class ModelScoreResult:
    scores: dict[str, float]
    available: bool
    backend: str
    warnings: list[str]
    raw: dict[str, Any]


class WindowScorer(Protocol):
    backend: str

    def score_window(self, samples: np.ndarray, sample_rate: int) -> ModelScoreResult:
        """Score one mono audio window."""


class UnavailableScorer:
    """Safe placeholder when CLAP/PANNs dependencies are not present."""

    backend = "unavailable"

    def __init__(self, reason: str = "model scorer not configured") -> None:
        self.reason = reason

    def score_window(self, samples: np.ndarray, sample_rate: int) -> ModelScoreResult:
        return ModelScoreResult(
            scores={key: 0.0 for key in ALL_SCORE_KEYS},
            available=False,
            backend=self.backend,
            warnings=["model_scores_unavailable"],
            raw={"reason": self.reason},
        )


def group_prompt_sets(params: dict[str, Any]) -> dict[str, list[str]]:
    prompts = params.get("prompts", {})
    contamination = prompts.get("contamination", {})
    return {
        "rain": list(prompts.get("rain", [])),
        "wind": list(prompts.get("wind", [])),
        "thunder": list(prompts.get("thunder", [])),
        "none": list(prompts.get("none", [])),
        "bio_contamination": list(contamination.get("bio", [])),
        "human_machine_contamination": list(contamination.get("human_machine", [])),
    }


def max_pool_prompt_scores(prompt_scores: dict[str, float], prompt_sets: dict[str, list[str]]) -> dict[str, float]:
    pooled: dict[str, float] = {}
    for group, prompts in prompt_sets.items():
        if not prompts:
            pooled[group] = 0.0
            continue
        pooled[group] = max(float(prompt_scores.get(prompt, 0.0)) for prompt in prompts)
    return pooled


class ClapScorer:
    """Lazy CLAP scorer boundary.

    The concrete CLAP implementation is intentionally deferred until the serverB
    dependency/runtime is selected. Calling this scorer without implementation
    returns unavailable instead of pretending feature heuristics are model
    confidence.
    """

    backend = "clap"

    def __init__(self, params: dict[str, Any]) -> None:
        self.prompt_sets = group_prompt_sets(params)
        self._delegate, unavailable_reason = _build_transformers_clap_delegate(
            self.prompt_sets
        )
        self._unavailable = UnavailableScorer(unavailable_reason)

    def score_window(self, samples: np.ndarray, sample_rate: int) -> ModelScoreResult:
        if self._delegate is None:
            result = self._unavailable.score_window(samples, sample_rate)
            return ModelScoreResult(
                scores=result.scores,
                available=False,
                backend=self.backend,
                warnings=result.warnings,
                raw={
                    "prompt_sets": self.prompt_sets,
                    "reason": result.raw["reason"],
                },
            )
        return self._delegate.score_window(samples, sample_rate)


class TransformersClapScorer:
    backend = "clap"

    def __init__(self, prompt_sets: dict[str, list[str]]) -> None:
        try:
            from .clap_backbone import CLAPBackbone
        except ImportError:
            from clap_backbone import CLAPBackbone

        self.prompt_sets = prompt_sets
        self.prompts = [prompt for prompts in prompt_sets.values() for prompt in prompts]
        self.backbone = CLAPBackbone()
        text_embeddings = self.backbone.embed_text(self.prompts)
        self.text_embeddings = text_embeddings.astype(np.float32, copy=False)

    def score_window(self, samples: np.ndarray, sample_rate: int) -> ModelScoreResult:
        audio_embedding = self.backbone.embed_audio_array(samples, sample_rate)
        prompt_scores = self.text_embeddings @ audio_embedding.astype(np.float32)
        raw_prompt_scores = {
            prompt: float(score)
            for prompt, score in zip(self.prompts, prompt_scores, strict=True)
        }
        grouped_scores = max_pool_prompt_scores(raw_prompt_scores, self.prompt_sets)
        return ModelScoreResult(
            scores={key: float(grouped_scores.get(key, 0.0)) for key in ALL_SCORE_KEYS},
            available=True,
            backend=self.backend,
            warnings=[],
            raw={"prompt_scores": raw_prompt_scores},
        )


def _build_transformers_clap_delegate(
    prompt_sets: dict[str, list[str]],
) -> tuple[WindowScorer | None, str]:
    try:
        return TransformersClapScorer(prompt_sets), ""
    except Exception as exc:
        return None, f"CLAP backend unavailable: {exc}"


def build_scorer(params: dict[str, Any], backend: str) -> WindowScorer:
    if backend == "none":
        return UnavailableScorer("backend disabled")
    if backend == "clap":
        return ClapScorer(params)
    return UnavailableScorer(f"unknown backend: {backend}")
