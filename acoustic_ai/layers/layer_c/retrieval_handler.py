"""Registry handler for the Layer C audited retrieval baseline.

This path is intentionally retrieval-only: it selects audited real bird-call
snippets, schedules them into a 60-second Layer C timeline, and returns a WAV,
mel array, and metadata through the same registry interface as model-backed
attempts.
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[3]
SR = 22_050

DEFAULT_INDEX = (
    REPO_ROOT
    / "acoustic_ai"
    / "layers"
    / "layer_c"
    / "attempts"
    / "burger__mvp_2__retrieval_v2_library"
    / "data"
    / "media_asset_bank"
    / "layer_c_retrieval_v2_event_index.csv"
)

DIEL_NEIGHBOURS = {
    "dawn": ["morning", "night"],
    "morning": ["dawn", "afternoon"],
    "afternoon": ["morning", "dusk"],
    "dusk": ["evening", "afternoon", "night"],
    "evening": ["dusk", "night"],
    "night": ["evening", "dawn", "dusk"],
}


@dataclass(frozen=True)
class EventSnippet:
    snippet_id: str
    event_type: str
    species_common_name: str
    species_scientific_name: str
    audio_event_id: str
    audio_path: str
    score: float
    quality_score: float | None
    diel_bin: str
    season: str
    duration_s: float
    recording_id: str
    event_start_seconds: float | None
    event_end_seconds: float | None
    source_manifest: str
    verdict: str
    notes: str


@dataclass(frozen=True)
class RetrievedSnippet:
    snippet: EventSnippet
    retrieval_score: float
    selection_reason: str


@dataclass(frozen=True)
class ScheduledEvent:
    retrieved: RetrievedSnippet
    onset_s: float
    offset_s: float
    gain_db: float
    pitch_shift_semitones: float
    time_stretch_rate: float
    fade_s: float


@dataclass(frozen=True)
class LayerResult:
    audio: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class RetrievalState:
    params: dict
    index_path: Path
    snippets: list[EventSnippet]


def load(checkpoint_dir: Path | None, params: dict, extra: dict | None = None) -> RetrievalState:
    """Load the audited snippet index. No checkpoint is needed."""

    del checkpoint_dir, extra
    index_path = Path(params.get("retrieval_index", DEFAULT_INDEX))
    if not index_path.is_absolute():
        index_path = REPO_ROOT / index_path
    if not index_path.exists():
        raise FileNotFoundError(f"Layer C retrieval index not found: {index_path}")
    return RetrievalState(
        params=dict(params),
        index_path=index_path,
        snippets=_load_index(index_path),
    )


def generate(
    state: RetrievalState,
    seed: int | None = None,
    season: str | None = None,
    diel: str | None = None,
    species_common_name: str | None = None,
    duration_s: float | None = None,
    **_: object,
) -> dict:
    """Generate a frontend-ready 60s Layer C retrieval demo."""

    params = state.params
    run_seed = int(seed if seed is not None else params.get("seed", 42))
    species = str(species_common_name or params["species_common_name"])
    duration_s = float(
        duration_s if duration_s is not None else params.get("duration_s", 60.0)
    )
    count = int(params.get("count", 10))
    season = season or params.get("default_season", "summer")
    diel = diel or params.get("default_diel", "morning")

    selected = _retrieve(
        snippets=state.snippets,
        species=species,
        diel_bin=diel,
        season=season,
        count=count,
        seed=run_seed,
    )
    layer_c = _render_events(
        _schedule(
            selected,
            target_duration_s=duration_s,
            seed=run_seed,
            enable_variation=bool(params.get("enable_variation", False)),
            ecological_mode=bool(params.get("ecological_mode", True)),
        ),
        target_duration_s=duration_s,
    )

    events_gain_db = float(params.get("events_gain_db", -1.0))
    layer_c_only = bool(params.get("layer_c_only", False))
    if layer_c_only:
        mix = layer_c.audio * _gain_to_amp(events_gain_db)
    else:
        ambient = _procedural_ambient(SR, duration_s, seed=run_seed + 10_000)
        mix = (
            ambient * _gain_to_amp(float(params.get("ambient_gain_db", -26.0)))
            + layer_c.audio * _gain_to_amp(events_gain_db)
        )
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix = mix * (0.98 / peak)
        peak = 0.98

    metadata = {
        "layer": "layer_c",
        "method": "retrieval_baseline_layer_c_only" if layer_c_only else "retrieval_baseline",
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
            "rms": float(np.sqrt(np.mean(np.square(mix)))) if mix.size else 0.0,
            "contains_ambient_bed": not layer_c_only,
        },
        "retrieval": {
            "index_path": str(state.index_path.relative_to(REPO_ROOT)),
            "selected_count": len(layer_c.metadata["events"]),
            "library_source": "audited real bird-call snippets only",
        },
        "mix": {
            "ambient_kind": "none" if layer_c_only else "procedural_debug_bed",
            "ambient_gain_db": float(params.get("ambient_gain_db", -26.0)),
            "events_gain_db": events_gain_db,
            "variation_enabled": bool(params.get("enable_variation", False)),
            "limitations": [
                (
                    "This frontend demo is Layer C only; Layer D mixing is separate."
                    if layer_c_only
                    else "This frontend demo is an A+C retrieval presentation mix, not full Layer D."
                ),
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


def _load_index(path: Path) -> list[EventSnippet]:
    snippets: list[EventSnippet] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            snippets.append(
                EventSnippet(
                    snippet_id=row["snippet_id"],
                    event_type=row["event_type"],
                    species_common_name=row["species_common_name"],
                    species_scientific_name=row.get("species_scientific_name", ""),
                    audio_event_id=row.get("audio_event_id", ""),
                    audio_path=row["audio_path"],
                    score=_parse_float(row.get("score")) or 0.0,
                    quality_score=_parse_float(row.get("quality_score")),
                    diel_bin=row.get("diel_bin", ""),
                    season=row.get("season", "unknown"),
                    duration_s=_parse_float(row.get("duration_s")) or 0.0,
                    recording_id=row.get("recording_id", ""),
                    event_start_seconds=_parse_float(row.get("event_start_seconds")),
                    event_end_seconds=_parse_float(row.get("event_end_seconds")),
                    source_manifest=row.get("source_manifest", ""),
                    verdict=row.get("verdict", ""),
                    notes=row.get("notes", ""),
                )
            )
    return snippets


def _retrieve(
    *,
    snippets: list[EventSnippet],
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
        for snippet in snippets
        if species_key in snippet.species_common_name.casefold()
        or species_key in snippet.event_type.casefold()
    ]
    if not species_matches:
        available = sorted({s.species_common_name for s in snippets})
        raise ValueError(f"No snippets for species={species!r}. Available: {available}")

    exact = [s for s in species_matches if s.diel_bin == diel_bin]
    fallback_bin = ""
    if exact:
        candidates = list(exact)
        seen_ids = {s.snippet_id for s in candidates}
        for neighbour in DIEL_NEIGHBOURS.get(diel_bin, []):
            for snippet in species_matches:
                if snippet.diel_bin == neighbour and snippet.snippet_id not in seen_ids:
                    candidates.append(snippet)
                    seen_ids.add(snippet.snippet_id)
            if len(candidates) >= count:
                fallback_bin = f"expanded_with_{neighbour}"
                break
        if len(candidates) < count:
            for snippet in species_matches:
                if snippet.snippet_id not in seen_ids:
                    candidates.append(snippet)
                    seen_ids.add(snippet.snippet_id)
            if len(candidates) > len(exact):
                fallback_bin = fallback_bin or "expanded_with_any"
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
            retrieval_score=_score(snippet, diel_bin=diel_bin, season=season),
            selection_reason=_reason(
                snippet,
                requested_diel=diel_bin,
                requested_season=season,
                fallback_bin=fallback_bin,
            ),
        )
        for snippet in candidates
    ]
    scored.sort(key=lambda item: item.retrieval_score, reverse=True)
    top_pool = scored[: max(count * 3, count)]
    rng.shuffle(top_pool)

    selected: list[RetrievedSnippet] = []
    last_recording = ""
    while top_pool and len(selected) < count:
        index = next(
            (i for i, item in enumerate(top_pool) if item.snippet.recording_id != last_recording),
            0,
        )
        item = top_pool.pop(index)
        selected.append(item)
        last_recording = item.snippet.recording_id
    return selected


def _schedule(
    snippets: list[RetrievedSnippet],
    *,
    target_duration_s: float,
    seed: int,
    enable_variation: bool,
    ecological_mode: bool,
) -> list[ScheduledEvent]:
    if not snippets:
        return []
    rng = random.Random(seed)
    profile = _profile_for(snippets[0].snippet.event_type) if ecological_mode else _default_profile()
    events: list[ScheduledEvent] = []
    cursor = rng.uniform(*profile["start_window_s"])
    pool = list(snippets)
    last_recording = ""

    while pool and len(events) < len(snippets):
        bout_size = rng.randint(*profile["bout_size"])
        placed_in_bout = 0

        while pool and placed_in_bout < bout_size and len(events) < len(snippets):
            index = next(
                (i for i, item in enumerate(pool) if item.snippet.recording_id != last_recording),
                0,
            )
            item = pool.pop(index)
            duration = max(0.1, item.snippet.duration_s)
            time_stretch_rate = rng.uniform(0.98, 1.02) if enable_variation else 1.0
            rendered_duration = duration / time_stretch_rate
            if cursor + rendered_duration > target_duration_s - profile["end_margin_s"]:
                return events

            event = ScheduledEvent(
                retrieved=item,
                onset_s=round(cursor, 3),
                offset_s=round(cursor + rendered_duration, 3),
                gain_db=round(rng.uniform(*profile["gain_db"]), 3),
                pitch_shift_semitones=round(rng.uniform(-0.2, 0.2), 3) if enable_variation else 0.0,
                time_stretch_rate=round(time_stretch_rate, 3),
                fade_s=round(rng.uniform(*profile["fade_s"]), 3),
            )
            events.append(event)
            last_recording = item.snippet.recording_id
            placed_in_bout += 1
            gap_key = "within_bout_gap_s" if placed_in_bout < bout_size else "between_bout_gap_s"
            cursor += rendered_duration + rng.uniform(*profile[gap_key])
    return events


def _render_events(events: list[ScheduledEvent], *, target_duration_s: float) -> LayerResult:
    total_samples = int(round(target_duration_s * SR))
    layer = np.zeros(total_samples, dtype=np.float32)
    metadata_events = []

    for event in events:
        audio = _load_audio(Path(event.retrieved.snippet.audio_path))
        audio = _apply_variation(
            audio,
            pitch_shift_semitones=event.pitch_shift_semitones,
            time_stretch_rate=event.time_stretch_rate,
        )
        audio = _fade(audio, fade_s=event.fade_s)
        audio *= _gain_to_amp(event.gain_db)

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
    return LayerResult(audio=layer.astype(np.float32), metadata={"events": metadata_events})


def _profile_for(event_type: str) -> dict:
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
    return _default_profile()


def _default_profile() -> dict:
    return {
        "start_window_s": (3.0, 8.0),
        "bout_size": (1, 2),
        "within_bout_gap_s": (4.0, 8.0),
        "between_bout_gap_s": (10.0, 20.0),
        "gain_db": (-7.0, -3.0),
        "fade_s": (0.08, 0.18),
        "end_margin_s": 2.0,
    }


def _score(snippet: EventSnippet, diel_bin: str, season: str) -> float:
    score = snippet.score
    if snippet.diel_bin == diel_bin:
        score += 0.2
    if snippet.season == season:
        score += 0.1
    if 2.0 <= snippet.duration_s <= 8.0:
        score += 0.05
    return score


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


def _load_audio(path: Path) -> np.ndarray:
    if not path.is_absolute():
        path = REPO_ROOT / path
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return np.asarray(audio, dtype=np.float32)


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


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
