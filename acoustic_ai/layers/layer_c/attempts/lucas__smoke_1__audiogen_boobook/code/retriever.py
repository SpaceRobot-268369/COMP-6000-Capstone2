"""Layer C event snippet retrieval."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from event_index import EventSnippet, load_index


DIEL_NEIGHBOURS = {
    "dawn": ["morning", "night"],
    "morning": ["dawn", "afternoon"],
    "afternoon": ["morning", "dusk"],
    "dusk": ["evening", "afternoon", "night"],
    "evening": ["dusk", "night"],
    "night": ["evening", "dawn", "dusk"],
}


@dataclass(frozen=True)
class RetrievedSnippet:
    snippet: EventSnippet
    retrieval_score: float
    selection_reason: str


class EventRetriever:
    """Select real Layer C snippets for a species/time request."""

    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self.snippets = load_index(self.index_path)

    def retrieve(
        self,
        species: str,
        diel_bin: str,
        season: str,
        count: int,
        seed: int,
    ) -> list[RetrievedSnippet]:
        rng = random.Random(seed)
        species_key = species.casefold()
        species_matches = [
            snippet
            for snippet in self.snippets
            if species_key in snippet.species_common_name.casefold()
            or species_key in snippet.event_type.casefold()
        ]
        if not species_matches:
            available = sorted({s.species_common_name for s in self.snippets})
            raise ValueError(f"No snippets for species={species!r}. Available: {available}")

        exact = [s for s in species_matches if s.diel_bin == diel_bin]
        fallback_bin = ""
        if exact:
            candidates = exact
        else:
            candidates = []
            for neighbour in DIEL_NEIGHBOURS.get(diel_bin, []):
                candidates = [s for s in species_matches if s.diel_bin == neighbour]
                if candidates:
                    fallback_bin = neighbour
                    break
            if not candidates:
                candidates = species_matches
                fallback_bin = "any"

        scored = [
            RetrievedSnippet(
                snippet=snippet,
                retrieval_score=self._score(snippet, diel_bin=diel_bin, season=season),
                selection_reason=self._reason(
                    snippet,
                    requested_diel=diel_bin,
                    requested_season=season,
                    fallback_bin=fallback_bin,
                ),
            )
            for snippet in candidates
        ]

        # Keep high-scoring snippets early while allowing deterministic variety.
        scored.sort(key=lambda item: item.retrieval_score, reverse=True)
        top_pool = scored[: max(count * 3, count)]
        rng.shuffle(top_pool)

        selected: list[RetrievedSnippet] = []
        last_recording = ""
        while top_pool and len(selected) < count:
            index = next(
                (
                    i
                    for i, item in enumerate(top_pool)
                    if item.snippet.recording_id != last_recording
                ),
                0,
            )
            item = top_pool.pop(index)
            selected.append(item)
            last_recording = item.snippet.recording_id

        return selected

    @staticmethod
    def _score(snippet: EventSnippet, diel_bin: str, season: str) -> float:
        score = snippet.score
        if snippet.diel_bin == diel_bin:
            score += 0.2
        if snippet.season == season:
            score += 0.1
        # Prefer concise foreground calls over very long buffered windows.
        if 2.0 <= snippet.duration_s <= 8.0:
            score += 0.05
        return score

    @staticmethod
    def _reason(
        snippet: EventSnippet,
        requested_diel: str,
        requested_season: str,
        fallback_bin: str,
    ) -> str:
        parts = [f"BirdNET score {snippet.score:.4f}"]
        if snippet.diel_bin == requested_diel:
            parts.append(f"exact diel match {requested_diel}")
        elif fallback_bin:
            parts.append(f"diel fallback {requested_diel}->{fallback_bin}")
        else:
            parts.append(f"diel mismatch requested={requested_diel} source={snippet.diel_bin}")
        if snippet.season == requested_season:
            parts.append(f"season match {requested_season}")
        else:
            parts.append(f"season fallback requested={requested_season} source={snippet.season}")
        if snippet.verdict:
            parts.append(f"verdict {snippet.verdict}")
        return "; ".join(parts)
