"""Layer A — Ambient bed retrieval and blending.

Given a user env request (diel_bin, season, hour, month), retrieve the k most
similar cleaned ambient segments by:
  1. Hard filter: diel_bin and season must match.
  2. Soft rank: cosine similarity over [hour_sin, hour_cos, month_sin, month_cos].
  3. Blend: softmax-weighted crossfade of top-k segments, RMS-matched, tiled to
     target duration.

Output: LayerResult with audio, gain_db=-3, and metadata (retrieved_clips,
blend_weights, requested_env).
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22_050
HOP = 512
FPS = SR / HOP


@dataclass
class LayerResult:
    """Output contract for a layer (shared with mixer)."""
    audio: np.ndarray  # float32, shape (n_samples,), range [-1, 1]
    sample_rate: int   # always 22050
    gain_db: float     # recommended gain for final mix
    metadata: dict     # for explanation JSON


class AmbientRetriever:
    """Retrieve and blend ambient segments from the cleaned pool."""

    def __init__(self, index_path: Path, segments_dir: Path):
        """Load the ambient segment pool.

        Args:
            index_path: path to ambient_index.csv
            segments_dir: path to ambient_segments/ folder
        """
        self.index_path = Path(index_path)
        self.segments_dir = Path(segments_dir)
        self.rows = self._load_index()

    def _load_index(self) -> list[dict]:
        """Load and parse ambient_index.csv."""
        rows = []
        with open(self.index_path) as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows

    def retrieve(
        self,
        diel_bin: str,
        season: str,
        hour: int,
        month: int,
        k: int = 5,
        target_duration_s: float = 60.0,
    ) -> LayerResult:
        """Retrieve and blend ambient segments for the requested environment.

        Args:
            diel_bin: 'dawn', 'morning', 'afternoon', 'night'
            season: 'spring', 'summer', 'autumn', 'winter'
            hour: local hour (0–23)
            month: month (1–12)
            k: top-k segments to blend (default 5)
            target_duration_s: output duration in seconds

        Returns:
            LayerResult with blended audio, gain_db=-3, metadata.

        Fallback: if hard filter returns < k segments, relax to neighbouring
        diel_bin and set low_confidence=True in metadata.
        """
        # Hard filter: diel_bin and season must match.
        candidates = [
            r for r in self.rows
            if r["diel_bin"] == diel_bin and r["season"] == season
        ]

        low_confidence = False
        if len(candidates) < k:
            # Relax filter: try neighbouring diel_bins.
            neighbour_bins = self._diel_neighbours(diel_bin)
            for neighbour in neighbour_bins:
                candidates = [
                    r for r in self.rows
                    if r["diel_bin"] == neighbour and r["season"] == season
                ]
                if len(candidates) >= k:
                    low_confidence = True
                    diel_bin = neighbour  # use the neighbouring bin
                    break

        if len(candidates) < k:
            raise ValueError(
                f"insufficient segments for {diel_bin}/{season}. "
                f"found {len(candidates)}, need {k}."
            )

        # Soft rank: cosine similarity over cyclic time features.
        hour_sin, hour_cos = self._cyclic_encode(hour, 24)
        month_sin, month_cos = self._cyclic_encode(month, 12)
        request_vec = np.array([hour_sin, hour_cos, month_sin, month_cos])

        sims = []
        for row in candidates:
            cand_vec = np.array([
                float(row["hour_sin"]),
                float(row["hour_cos"]),
                float(row["month_sin"]),
                float(row["month_cos"]),
            ])
            sim = self._cosine_sim(request_vec, cand_vec)
            sims.append(sim)

        # Top-k by similarity.
        top_k_idx = np.argsort(sims)[-k:][::-1]  # descending
        top_k_candidates = [candidates[i] for i in top_k_idx]
        top_k_sims = [sims[i] for i in top_k_idx]

        # Load audio for top-k.
        audios = []
        for row in top_k_candidates:
            seg_id = row["segment_id"]
            seg_path = self.segments_dir / f"{seg_id}.wav"
            audio, _ = sf.read(seg_path, dtype="float32")
            audios.append(audio)

        # Blend: softmax weights, crossfade, RMS-match, tile to duration.
        blended = self._blend_segments(
            audios, top_k_sims, target_duration_s
        )

        metadata = {
            "retrieved_clips": [
                {
                    "clip_id": row["segment_id"],
                    "cosine_similarity": round(float(sims[top_k_idx[i]]), 3),
                    "env": {
                        "diel_bin": row["diel_bin"],
                        "season": row["season"],
                        "hour": hour,
                        "month": month,
                    },
                }
                for i, row in enumerate(top_k_candidates)
            ],
            "blend_weights": [round(float(w), 3) for w in self._softmax(top_k_sims)],
            "requested_env": {
                "diel_bin": diel_bin,
                "season": season,
                "hour": hour,
                "month": month,
            },
            "low_confidence": low_confidence,
        }

        return LayerResult(
            audio=blended,
            sample_rate=SR,
            gain_db=-3.0,
            metadata=metadata,
        )

    def _diel_neighbours(self, diel_bin: str) -> list[str]:
        """Return neighbouring diel_bins in cyclic order."""
        order = ["dawn", "morning", "afternoon", "night"]
        idx = order.index(diel_bin)
        return [order[(idx - 1) % 4], order[(idx + 1) % 4]]

    @staticmethod
    def _cyclic_encode(value: float, period: float) -> tuple[float, float]:
        """Encode value as sin/cos for cyclic features."""
        theta = 2 * math.pi * value / period
        return math.sin(theta), math.cos(theta)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _softmax(scores: list[float], tau: float = 0.1) -> np.ndarray:
        """Softmax with temperature."""
        x = np.array(scores) / tau
        x = x - np.max(x)  # for numerical stability
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    def _blend_segments(
        self,
        audios: list[np.ndarray],
        sims: list[float],
        target_duration_s: float,
    ) -> np.ndarray:
        """Blend k segments via sequential equal-power crossfade, RMS-matched.

        Each segment plays at full (RMS-matched) amplitude.  Transitions use a
        0.5 s equal-power fade so perceived loudness stays constant across the
        boundary.  Segments are tiled/trimmed to fill exactly target_duration_s.

        Args:
            audios: list of k float32 audio arrays
            sims: k similarity scores (used only for ordering, already top-k)
            target_duration_s: target output duration in seconds

        Returns:
            Blended audio at target duration, float32, normalised to [-1, 1].
        """
        k = len(audios)
        target_samples = int(target_duration_s * SR)

        if k == 1:
            return self._tile_to_duration(audios[0], target_samples)

        # 0.5 s equal-power crossfade.
        crossfade_samples = int(0.5 * SR)

        # Segment length so that chaining with overlaps fills target exactly:
        #   target = k * seg_samples - (k-1) * crossfade_samples
        seg_samples = (target_samples + (k - 1) * crossfade_samples) // k

        # Equal-power ramps: fade[i] = sqrt(i / n)
        fade_in  = np.sqrt(np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32))
        fade_out = fade_in[::-1].copy()

        # RMS-match all segments to the first one.
        reference_rms: float | None = None
        segments: list[np.ndarray] = []
        for i, audio in enumerate(audios):
            seg = self._tile_to_duration(audio, seg_samples).copy()
            rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-8
            if i == 0:
                reference_rms = rms
            seg = seg * (reference_rms / rms)  # type: ignore[operator]

            # Apply crossfade envelopes at segment boundaries only.
            if i > 0:                          # fade in at the start
                seg[:crossfade_samples] *= fade_in
            if i < k - 1:                      # fade out at the end
                seg[-crossfade_samples:] *= fade_out

            segments.append(seg)

        # Overlap-add into the output buffer.
        blended = np.zeros(target_samples, dtype=np.float32)
        pos = 0
        for seg in segments:
            end = min(pos + seg_samples, target_samples)
            blended[pos:end] += seg[: end - pos]
            pos = pos + seg_samples - crossfade_samples

        # Peak-normalise only if clipping.
        max_abs = float(np.abs(blended).max())
        if max_abs > 1.0:
            blended = blended / max_abs

        return blended

    @staticmethod
    def _tile_to_duration(audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Tile or trim audio to exact sample count."""
        if len(audio) == target_samples:
            return audio
        if len(audio) > target_samples:
            return audio[:target_samples]
        # Tile: repeat the audio with phase randomisation to avoid audible loops.
        reps = (target_samples // len(audio)) + 1
        tiled = np.tile(audio, reps)[:target_samples]
        return tiled.astype(np.float32)
