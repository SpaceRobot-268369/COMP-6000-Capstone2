"""Layer C event timeline scheduler and renderer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .retriever import RetrievedSnippet


REPO_ROOT = Path(__file__).resolve().parents[3]
SR = 22_050


@dataclass(frozen=True)
class LayerResult:
    audio: np.ndarray
    sample_rate: int
    gain_db: float
    metadata: dict


@dataclass(frozen=True)
class ScheduledEvent:
    retrieved: RetrievedSnippet
    onset_s: float
    offset_s: float
    gain_db: float
    pitch_shift_semitones: float
    time_stretch_rate: float
    fade_s: float


class EventScheduler:
    """Place retrieved snippets into a sparse Layer C event timeline."""

    def __init__(
        self,
        target_duration_s: float = 60.0,
        min_gap_s: float = 8.0,
        max_gap_s: float = 20.0,
        seed: int = 42,
        enable_variation: bool = False,
        ecological_mode: bool = False,
    ):
        self.target_duration_s = target_duration_s
        self.min_gap_s = min_gap_s
        self.max_gap_s = max_gap_s
        self.rng = random.Random(seed)
        self.enable_variation = enable_variation
        self.ecological_mode = ecological_mode

    def schedule(self, snippets: list[RetrievedSnippet]) -> list[ScheduledEvent]:
        if self.ecological_mode:
            return self._schedule_ecological(snippets)

        events: list[ScheduledEvent] = []
        cursor = self.rng.uniform(2.0, 6.0)

        for item in snippets:
            duration = max(0.1, item.snippet.duration_s)
            time_stretch_rate = self.rng.uniform(0.97, 1.03) if self.enable_variation else 1.0
            rendered_duration = duration / time_stretch_rate
            if cursor + rendered_duration > self.target_duration_s:
                break
            gain_db = self.rng.uniform(-5.0, -2.0)
            pitch_shift_semitones = (
                self.rng.uniform(-0.3, 0.3) if self.enable_variation else 0.0
            )
            fade_s = self.rng.uniform(0.02, 0.06)
            events.append(
                ScheduledEvent(
                    retrieved=item,
                    onset_s=round(cursor, 3),
                    offset_s=round(cursor + rendered_duration, 3),
                    gain_db=round(gain_db, 3),
                    pitch_shift_semitones=round(pitch_shift_semitones, 3),
                    time_stretch_rate=round(time_stretch_rate, 3),
                    fade_s=round(fade_s, 3),
                )
            )
            cursor += rendered_duration + self.rng.uniform(self.min_gap_s, self.max_gap_s)

        return events

    def _schedule_ecological(self, snippets: list[RetrievedSnippet]) -> list[ScheduledEvent]:
        if not snippets:
            return []

        profile = self._profile_for(snippets[0].snippet.event_type)
        events: list[ScheduledEvent] = []
        cursor = self.rng.uniform(*profile["start_window_s"])
        pool = list(snippets)
        last_recording = ""

        while pool and len(events) < len(snippets):
            bout_size = self.rng.randint(*profile["bout_size"])
            placed_in_bout = 0

            while pool and placed_in_bout < bout_size and len(events) < len(snippets):
                index = next(
                    (
                        i
                        for i, item in enumerate(pool)
                        if item.snippet.recording_id != last_recording
                    ),
                    0,
                )
                item = pool.pop(index)
                duration = max(0.1, item.snippet.duration_s)
                time_stretch_rate = (
                    self.rng.uniform(0.98, 1.02) if self.enable_variation else 1.0
                )
                rendered_duration = duration / time_stretch_rate
                if cursor + rendered_duration > self.target_duration_s - profile["end_margin_s"]:
                    return events

                gain_db = self.rng.uniform(*profile["gain_db"])
                pitch_shift_semitones = (
                    self.rng.uniform(-0.2, 0.2) if self.enable_variation else 0.0
                )
                fade_s = self.rng.uniform(*profile["fade_s"])
                events.append(
                    ScheduledEvent(
                        retrieved=item,
                        onset_s=round(cursor, 3),
                        offset_s=round(cursor + rendered_duration, 3),
                        gain_db=round(gain_db, 3),
                        pitch_shift_semitones=round(pitch_shift_semitones, 3),
                        time_stretch_rate=round(time_stretch_rate, 3),
                        fade_s=round(fade_s, 3),
                    )
                )
                last_recording = item.snippet.recording_id
                placed_in_bout += 1

                if placed_in_bout < bout_size:
                    cursor += rendered_duration + self.rng.uniform(*profile["within_bout_gap_s"])
                else:
                    cursor += rendered_duration + self.rng.uniform(*profile["between_bout_gap_s"])

        return events

    @staticmethod
    def _profile_for(event_type: str) -> dict[str, tuple[float, float] | tuple[int, int] | float]:
        event_key = event_type.casefold()
        if "fairywren" in event_key:
            return {
                "start_window_s": (3.0, 7.0),
                "bout_size": (2, 4),
                "within_bout_gap_s": (1.2, 3.0),
                "between_bout_gap_s": (5.0, 10.0),
                "gain_db": (-8.0, -4.5),
                "fade_s": (0.10, 0.18),
                "end_margin_s": 1.5,
            }
        if "cuckoo" in event_key:
            return {
                "start_window_s": (4.0, 9.0),
                "bout_size": (2, 4),
                "within_bout_gap_s": (2.5, 5.0),
                "between_bout_gap_s": (7.0, 14.0),
                "gain_db": (-6.5, -3.0),
                "fade_s": (0.08, 0.16),
                "end_margin_s": 2.0,
            }
        return {
            "start_window_s": (3.0, 8.0),
            "bout_size": (1, 2),
            "within_bout_gap_s": (4.0, 8.0),
            "between_bout_gap_s": (10.0, 20.0),
            "gain_db": (-7.0, -3.0),
            "fade_s": (0.08, 0.18),
            "end_margin_s": 2.0,
        }

    def render(self, events: list[ScheduledEvent]) -> LayerResult:
        total_samples = int(round(self.target_duration_s * SR))
        layer = np.zeros(total_samples, dtype=np.float32)
        metadata_events = []

        for event in events:
            audio = self._load_audio(Path(event.retrieved.snippet.audio_path))
            audio = self._apply_variation(
                audio,
                pitch_shift_semitones=event.pitch_shift_semitones,
                time_stretch_rate=event.time_stretch_rate,
            )
            audio = self._fade(audio, fade_s=event.fade_s)
            audio *= 10 ** (event.gain_db / 20.0)

            start = int(round(event.onset_s * SR))
            end = min(start + len(audio), total_samples)
            if end <= start:
                continue
            layer[start:end] += audio[: end - start]

            metadata_events.append(
                {
                    "snippet_id": event.retrieved.snippet.snippet_id,
                    "label": event.retrieved.snippet.species_common_name,
                    "event_type": event.retrieved.snippet.event_type,
                    "audio_event_id": event.retrieved.snippet.audio_event_id,
                    "snippet_path": event.retrieved.snippet.audio_path,
                    "source_recording": event.retrieved.snippet.recording_id,
                    "onset_s": event.onset_s,
                    "offset_s": event.offset_s,
                    "gain_db": event.gain_db,
                    "pitch_shift_semitones": event.pitch_shift_semitones,
                    "time_stretch_rate": event.time_stretch_rate,
                    "fade_s": event.fade_s,
                    "confidence": round(event.retrieved.snippet.score, 4),
                    "retrieval_score": round(event.retrieved.retrieval_score, 4),
                    "diel_bin": event.retrieved.snippet.diel_bin,
                    "season": event.retrieved.snippet.season,
                    "selection_reason": event.retrieved.selection_reason,
                }
            )

        peak = float(np.max(np.abs(layer))) if layer.size else 0.0
        if peak > 0.98:
            layer = layer / peak * 0.98

        return LayerResult(
            audio=layer.astype(np.float32),
            sample_rate=SR,
            gain_db=-9.0,
            metadata={"events": metadata_events},
        )

    @staticmethod
    def _load_audio(path: Path) -> np.ndarray:
        if not path.is_absolute():
            path = REPO_ROOT / path
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def _fade(audio: np.ndarray, fade_s: float = 0.03) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).copy()
        fade_n = min(int(round(fade_s * SR)), len(audio) // 2)
        if fade_n <= 1:
            return audio
        fade_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        audio[:fade_n] *= fade_in
        audio[-fade_n:] *= fade_out
        return audio

    @staticmethod
    def _apply_variation(
        audio: np.ndarray,
        pitch_shift_semitones: float,
        time_stretch_rate: float,
    ) -> np.ndarray:
        varied = np.asarray(audio, dtype=np.float32)
        if abs(time_stretch_rate - 1.0) > 0.001 and len(varied) > 1024:
            varied = librosa.effects.time_stretch(varied, rate=time_stretch_rate)
        if abs(pitch_shift_semitones) > 0.001 and len(varied) > 1024:
            varied = librosa.effects.pitch_shift(varied, sr=SR, n_steps=pitch_shift_semitones)
        return np.asarray(varied, dtype=np.float32)
