"""AudioSet tagger adapters for E-B weather analysis.

PANNs CNN14 is the preferred second model for weather cross-checking, but the
package/checkpoint may not be present in every environment. This module keeps
the boundary explicit and safely reports unavailable until serverB has the
runtime dependency.
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
    """PANNs scorer placeholder.

    The repo docs choose PANNs CNN14 as the strongest off-the-shelf weather
    baseline, but the current serverB venv does not include `panns_inference`.
    This class intentionally raises during construction until that dependency
    is installed or vendored.
    """

    backend = "panns_cnn14"

    def __init__(self) -> None:
        try:
            import panns_inference  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("panns_inference is not installed") from exc
        raise RuntimeError("PANNs scorer implementation is not wired yet")

    def score_window(self, samples: np.ndarray, sample_rate: int) -> AudioSetScoreResult:
        raise NotImplementedError


def build_audioset_scorer(backend: str) -> AudioSetScorer:
    if backend == "none":
        return UnavailableAudioSetScorer("backend disabled")
    if backend == "panns":
        try:
            return PannsScorer()
        except Exception as exc:
            return UnavailableAudioSetScorer(f"PANNs backend unavailable: {exc}")
    return UnavailableAudioSetScorer(f"unknown backend: {backend}")

