from __future__ import annotations

import numpy as np

from app.schemas.audio import AudioSegment


def pad_or_trim_segment(segment, target_length: int):
    """Pad or trim a segment to a fixed length."""
    segment_array = np.asarray(segment, dtype=np.float32)
    current_length = segment_array.shape[-1]
    if current_length == target_length:
        return segment_array
    if current_length > target_length:
        return segment_array[:target_length]

    padded = np.zeros(target_length, dtype=np.float32)
    padded[:current_length] = segment_array
    return padded


def segment_audio(waveform, sample_rate: int, segment_duration: float, hop_duration: float) -> list[AudioSegment]:
    """Split audio into fixed-size segments for embedding extraction.

    Short trailing segments are zero-padded so downstream encoders receive a fixed length.
    """
    if waveform is None:
        return []

    waveform_array = np.asarray(waveform, dtype=np.float32)
    segment_length = max(1, int(round(segment_duration * sample_rate)))
    hop_length = max(1, int(round(hop_duration * sample_rate)))

    segments: list[AudioSegment] = []
    for segment_index, start in enumerate(range(0, waveform_array.shape[-1], hop_length)):
        raw_segment = waveform_array[start : start + segment_length]
        if raw_segment.size == 0:
            break
        fixed_segment = pad_or_trim_segment(raw_segment, segment_length)
        start_sec = start / sample_rate
        end_sec = min((start + segment_length) / sample_rate, waveform_array.shape[-1] / sample_rate)
        segments.append(
            AudioSegment(
                segment_index=segment_index,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                waveform=fixed_segment,
            )
        )
        if start + segment_length >= waveform_array.shape[-1]:
            break

    return segments
