"""Registry handler implementation for the Layer C retrieval baseline.

This handler exposes the audited real-snippet retrieval path through the same
FastAPI registry contract used by the frontend. It intentionally avoids any
LoRA/model load: Layer C smoke/MVP demo output is selected from the audited
event library, scheduled into a 60-second event timeline, and mixed over a quiet
procedural bed so the displayed mel-spectrogram is presentation-complete.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .retriever import EventRetriever
from .scheduler import EventScheduler, SR

REPO_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_INDEX = (
    REPO_ROOT
    / "resources"
    / "site_257_bowra-dry-a"
    / "layer_c_retrieval_event_library_split_v1"
    / "final_pass_library_v1"
    / "layer_c_retrieval_final_pass_event_index.csv"
)


@dataclass(frozen=True)
class RetrievalState:
    params: dict
    index_path: Path


def load(checkpoint_dir: Path | None, params: dict, extra: dict | None = None) -> RetrievalState:
    """Load the audited snippet index. No checkpoint is needed."""

    del checkpoint_dir, extra
    index_path = Path(params.get("retrieval_index", DEFAULT_INDEX))
    if not index_path.is_absolute():
        index_path = REPO_ROOT / index_path
    if not index_path.exists():
        raise FileNotFoundError(f"Layer C retrieval index not found: {index_path}")
    return RetrievalState(params=dict(params), index_path=index_path)


def generate(
    state: RetrievalState,
    seed: int | None = None,
    season: str | None = None,
    diel: str | None = None,
    **_: object,
) -> dict:
    """Generate a frontend-ready 60s Layer C retrieval demo."""

    params = state.params
    run_seed = int(seed if seed is not None else params.get("seed", 42))
    species = str(params["species_common_name"])
    duration_s = float(params.get("duration_s", 60.0))
    count = int(params.get("count", 10))
    season = season or params.get("default_season", "summer")
    diel = diel or params.get("default_diel", "morning")

    retriever = EventRetriever(state.index_path)
    selected = retriever.retrieve(
        species=species,
        diel_bin=diel,
        season=season,
        count=count,
        seed=run_seed,
    )
    scheduler = EventScheduler(
        target_duration_s=duration_s,
        seed=run_seed,
        ecological_mode=bool(params.get("ecological_mode", True)),
        enable_variation=bool(params.get("enable_variation", False)),
    )
    scheduled_events = scheduler.schedule(selected)
    layer_c = scheduler.render(scheduled_events)

    ambient = _procedural_ambient(SR, duration_s, seed=run_seed + 10_000)
    mix = (
        ambient * _gain_to_amp(float(params.get("ambient_gain_db", -26.0)))
        + layer_c.audio * _gain_to_amp(float(params.get("events_gain_db", -1.0)))
    )
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix = mix * (0.98 / peak)
        peak = 0.98

    metadata = {
        "layer": "layer_c",
        "method": "retrieval_baseline",
        "species": species,
        "request": {
            "seed": run_seed,
            "season": season,
            "diel": diel,
            "duration_s": duration_s,
            "count_requested": count,
        },
        "audio": {
            "sample_rate": SR,
            "duration_s": duration_s,
            "peak": peak,
        },
        "retrieval": {
            "index_path": str(state.index_path.relative_to(REPO_ROOT)),
            "selected_count": len(scheduled_events),
            "library_source": "audited real bird-call snippets only",
        },
        "mix": {
            "ambient_kind": "procedural_debug_bed",
            "ambient_gain_db": float(params.get("ambient_gain_db", -26.0)),
            "events_gain_db": float(params.get("events_gain_db", -1.0)),
            "variation_enabled": bool(params.get("enable_variation", False)),
            "limitations": [
                "This frontend demo is an A+C retrieval presentation mix, not full Layer D.",
                "Layer C events are real audited retrieval snippets, not from-scratch generated calls.",
            ],
        },
        "events": layer_c.metadata["events"],
    }

    return {
        "wav_bytes": _wav_bytes(mix, SR),
        "mel_db": _mel_db(mix, SR),
        "metadata": metadata,
    }


def _gain_to_amp(gain_db: float) -> float:
    return float(10 ** (gain_db / 20.0))


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _mel_db(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=80)


def _procedural_ambient(sample_rate: int, duration_s: float, seed: int) -> np.ndarray:
    """Quiet non-semantic bed used only to make the 60s demo mel complete."""

    rng = np.random.default_rng(seed)
    n = int(round(sample_rate * duration_s))
    white = rng.normal(0.0, 1.0, n).astype(np.float32)
    brown = np.cumsum(white)
    brown = brown / (np.max(np.abs(brown)) + 1e-8)
    shimmer = rng.normal(0.0, 0.15, n).astype(np.float32)
    shimmer = librosa.effects.preemphasis(shimmer)
    bed = 0.85 * brown + 0.15 * shimmer
    bed = bed / (np.max(np.abs(bed)) + 1e-8)
    fade_n = min(int(sample_rate * 2.0), n // 2)
    if fade_n > 0:
        fade = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        bed[:fade_n] *= fade
        bed[-fade_n:] *= fade[::-1]
    return bed.astype(np.float32)
