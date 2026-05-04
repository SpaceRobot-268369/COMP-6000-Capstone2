"""Granular synthesis blender for Layer A — Method 1A.

Extends pure retrieval (method 1) by treating retrieved segments as a grain
*source pool* rather than playing them sequentially. Each output sample is
constructed from many short overlapping windowed grains drawn from the source
segments, weighted by retrieval similarity.

Why granular for stationary ambient texture:
  - Stationary textures (cicada drone, insect wash, distant frog chorus) are
    characterised by a stable power spectrum and slow modulation envelope.
    Their identity is in the *statistics*, not in the precise time order.
    Random grain shuffling preserves both statistics and identity.
  - Output is novel — no two requests produce the same waveform — but every
    sample comes from real audio, so the machine-noise failure mode that
    cVAE+MSE produced is structurally impossible here.
  - Avoids the audible repetition of plain crossfade tiling when the requested
    duration exceeds the source pool's combined length.

Design:
  - Grain length: 200 ms default. Long enough to preserve insect/cicada cycles,
    short enough that grain boundaries are inaudible after windowing.
  - Window: Hann. Sum of squared Hann windows at 50% overlap is constant, so
    overlap-add gives constant RMS without explicit normalisation.
  - Overlap: 50% default.
  - Source pick: weighted by softmaxed retrieval similarity (top-k from
    retriever). High-similarity segments contribute more grains.
  - Within-source position: uniform random across the source segment.
  - Optional pitch jitter: ±N% per grain via fractional linear resampling.
    Off by default to keep MVP deterministic; enable for thesis-grade variety.

Notes:
  - This module does not load audio or call the retriever; it is a pure blender
    that takes (source audio arrays, similarity weights, target duration) and
    returns audio. Coupling lives in retrieval.py.
"""

from __future__ import annotations

import numpy as np

DEFAULT_GRAIN_MS = 200.0
DEFAULT_OVERLAP = 0.5
DEFAULT_PITCH_JITTER = 0.0
SR = 22_050


class GranularSynthesizer:
    """Granular synthesis OLA blender for stationary ambient textures."""

    def __init__(
        self,
        grain_ms: float = DEFAULT_GRAIN_MS,
        overlap: float = DEFAULT_OVERLAP,
        pitch_jitter: float = DEFAULT_PITCH_JITTER,
        seed: int | None = None,
    ):
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        if not 0.0 <= pitch_jitter < 0.1:
            raise ValueError(f"pitch_jitter must be in [0, 0.1), got {pitch_jitter}")
        if grain_ms <= 0:
            raise ValueError(f"grain_ms must be positive, got {grain_ms}")

        self.grain_ms = float(grain_ms)
        self.overlap = float(overlap)
        self.pitch_jitter = float(pitch_jitter)
        self.rng = np.random.default_rng(seed)
        self.last_n_grains: int = 0

    def synthesize(
        self,
        sources: list[np.ndarray],
        weights: list[float],
        target_duration_s: float,
        sample_rate: int = SR,
    ) -> np.ndarray:
        """Granular synthesis of target_duration_s seconds from a source pool.

        Args:
            sources: list of k source audio arrays (mono float32).
            weights: list of k similarity scores. Will be re-normalised so they
                sum to 1.0; treated as a categorical distribution over sources.
            target_duration_s: output duration in seconds.
            sample_rate: output sample rate (default 22050 to match SPEC_CFG).

        Returns:
            float32 audio of length target_duration_s * sample_rate.

        Raises:
            ValueError: on empty sources, length mismatch, or invalid weights.
        """
        if not sources:
            raise ValueError("sources must be non-empty")
        if len(sources) != len(weights):
            raise ValueError(
                f"sources ({len(sources)}) and weights ({len(weights)}) "
                f"must have equal length"
            )

        weights_arr = np.asarray(weights, dtype=np.float64)
        if np.any(weights_arr < 0):
            raise ValueError("weights must be non-negative")
        total = weights_arr.sum()
        if total <= 0:
            # Degenerate: fall back to uniform.
            weights_arr = np.ones_like(weights_arr) / len(weights_arr)
        else:
            weights_arr = weights_arr / total

        target_samples = int(target_duration_s * sample_rate)
        grain_samples = max(2, int(self.grain_ms * sample_rate / 1000.0))
        hop_samples = max(1, int(grain_samples * (1.0 - self.overlap)))

        # Hann window — at 50% overlap, sum of squared windows is ~constant,
        # so OLA gives constant RMS. At other overlaps, RMS drifts but stays
        # within a small range; we normalise post-hoc only if peak > 1.
        window = np.hanning(grain_samples).astype(np.float32)

        # RMS-match all sources to the first one. Without this, a loud source
        # selected even occasionally would punch above the bed level.
        ref_rms = float(np.sqrt(np.mean(sources[0].astype(np.float64) ** 2))) + 1e-8
        normed_sources: list[np.ndarray] = []
        for s in sources:
            s = np.asarray(s, dtype=np.float32)
            rms = float(np.sqrt(np.mean(s.astype(np.float64) ** 2))) + 1e-8
            normed_sources.append((s * (ref_rms / rms)).astype(np.float32))

        # Output buffer with grain-length tail so the last grain fits.
        out = np.zeros(target_samples + grain_samples, dtype=np.float32)

        pos = 0
        n_grains = 0
        while pos < target_samples:
            src_idx = int(self.rng.choice(len(normed_sources), p=weights_arr))
            src = normed_sources[src_idx]

            if len(src) <= grain_samples:
                # Source shorter than a grain — tile it (rare, sources are 20–60 s).
                grain = np.resize(src, grain_samples).astype(np.float32)
            else:
                start = int(self.rng.integers(0, len(src) - grain_samples))
                grain = src[start:start + grain_samples].copy()

            if self.pitch_jitter > 0.0:
                jitter = float(self.rng.uniform(-self.pitch_jitter, self.pitch_jitter))
                grain = self._pitch_shift(grain, 1.0 + jitter, grain_samples)

            grain *= window
            out[pos:pos + grain_samples] += grain

            pos += hop_samples
            n_grains += 1

        out = out[:target_samples]

        # Peak-normalise only if clipping. Hann-OLA at 50% overlap already
        # holds RMS roughly constant, so this is rarely triggered.
        peak = float(np.abs(out).max())
        if peak > 1.0:
            out = out / peak

        self.last_n_grains = n_grains
        return out.astype(np.float32)

    @staticmethod
    def _pitch_shift(
        grain: np.ndarray,
        rate: float,
        target_len: int,
    ) -> np.ndarray:
        """Resample a grain by linear interpolation, pad/trim to target_len.

        rate > 1 → grain plays back higher pitch and shorter; we then pad
        with zeros to keep grain_samples consistent. The trailing zero region
        is windowed to silence by the Hann envelope, so it is inaudible.
        rate < 1 → grain plays lower pitch and longer; we trim to target_len.
        """
        n = len(grain)
        new_n = max(1, int(round(n / rate)))
        idx = np.linspace(0, n - 1, new_n)
        shifted = np.interp(idx, np.arange(n), grain).astype(np.float32)
        if len(shifted) >= target_len:
            return shifted[:target_len]
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(shifted)] = shifted
        return out
