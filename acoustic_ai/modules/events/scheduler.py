"""Layer C event timeline scheduler and renderer."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from retriever import RetrievedSnippet


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
    ):
        self.target_duration_s = target_duration_s
        self.min_gap_s = min_gap_s
        self.max_gap_s = max_gap_s
        self.rng = random.Random(seed)
        self.enable_variation = enable_variation

    def schedule(self, snippets: list[RetrievedSnippet]) -> list[ScheduledEvent]:
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

    def write_debug_bundle(
        self,
        result: LayerResult,
        out_dir: Path,
        request: dict,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / "layer_c_events.wav"
        json_path = out_dir / "layer_c_timeline.json"
        png_path = out_dir / "layer_c_timeline.png"
        spectrogram_path = out_dir / "layer_c_spectrogram.png"

        sf.write(audio_path, result.audio, result.sample_rate)
        payload = {
            "layer": "events",
            "audio_path": str(audio_path),
            "timeline_path": str(png_path),
            "spectrogram_path": str(spectrogram_path),
            "events": result.metadata["events"],
            "scheduling_params": {
                **request,
                "target_duration_s": self.target_duration_s,
                "sample_rate": result.sample_rate,
                "gain_db": result.gain_db,
                "min_gap_s": self.min_gap_s,
                "max_gap_s": self.max_gap_s,
            },
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._write_timeline_png(payload["events"], png_path)
        self._write_spectrogram_png(result.audio, result.sample_rate, spectrogram_path)

    @staticmethod
    def _load_audio(path: Path) -> np.ndarray:
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
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
        """Apply conservative sample-level variation to real retrieval snippets."""

        varied = np.asarray(audio, dtype=np.float32)
        if abs(time_stretch_rate - 1.0) > 0.001 and len(varied) > 1024:
            varied = librosa.effects.time_stretch(varied, rate=time_stretch_rate)
        if abs(pitch_shift_semitones) > 0.001 and len(varied) > 1024:
            varied = librosa.effects.pitch_shift(
                varied,
                sr=SR,
                n_steps=pitch_shift_semitones,
            )
        return np.asarray(varied, dtype=np.float32)

    @staticmethod
    def _write_timeline_png(events: list[dict], path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 2.4))
        for idx, event in enumerate(events):
            onset = float(event["onset_s"])
            duration = float(event["offset_s"]) - onset
            ax.broken_barh([(onset, duration)], (idx - 0.35, 0.7), facecolors="#3b82f6")
            ax.text(
                onset,
                idx,
                event["audio_event_id"],
                va="center",
                ha="left",
                fontsize=8,
                color="black",
            )
        ax.set_xlabel("time (s)")
        ax.set_yticks([])
        ax.set_title("Layer C Retrieval Timeline")
        ax.set_xlim(0, max([60.0, *[float(e["offset_s"]) for e in events]]))
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _write_spectrogram_png(audio: np.ndarray, sr: int, path: Path) -> None:
        mel = librosa.feature.melspectrogram(
            y=np.asarray(audio, dtype=np.float32),
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
        fig, ax = plt.subplots(figsize=(11, 3.2))
        image = ax.imshow(
            mel_db,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(0, len(audio) / sr, 0, mel_db.shape[0]),
            vmin=-80,
            vmax=0,
            cmap="magma",
        )
        ax.set_title("Layer C Retrieval Event Spectrogram")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mel bin")
        fig.colorbar(image, ax=ax, label="dB")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
