"""Registry handler for the Layer C audited retrieval baseline.

This path is intentionally retrieval-only: it selects audited real bird-call
snippets, schedules them into a 60-second Layer C timeline, and returns a WAV,
mel array, and metadata through the same registry interface as model-backed
attempts.
"""

from __future__ import annotations

import csv
import io
import json
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

DEFAULT_BANK = (
    REPO_ROOT / "model" / "candidates" / "burger" / "mvp_2__retrieval_v2_library"
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
    bank_root: Path | None
    snippets: list[EventSnippet]


def load(checkpoint_dir: Path | None, params: dict, extra: dict | None = None) -> RetrievalState:
    """Load the audited snippet index from the retrieval asset bank."""

    del extra
    bank_root: Path | None = None
    if checkpoint_dir is not None:
        bank_root = Path(checkpoint_dir)
        index_path = bank_root / "index.json"
    else:
        index_path = Path(params.get("retrieval_index", DEFAULT_BANK / "index.json"))
        if not index_path.is_absolute():
            index_path = REPO_ROOT / index_path
        if index_path.name == "index.json":
            bank_root = index_path.parent
    if not index_path.exists():
        raise FileNotFoundError(f"Layer C retrieval index not found: {index_path}")
    return RetrievalState(
        params=dict(params),
        index_path=index_path,
        bank_root=bank_root,
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
    """Generate a frontend-ready Layer C retrieval clip."""

    params = state.params
    run_seed = int(seed if seed is not None else params.get("seed", 42))
    species = str(species_common_name or params["species_common_name"])
    requested_duration_s = duration_s
    season = season or params.get("default_season", "summer")
    diel = diel or params.get("default_diel", "morning")

    selected = _retrieve(
        snippets=state.snippets,
        species=species,
        diel_bin=diel,
        season=season,
        count=1,
        seed=run_seed,
    )
    chosen = selected[0]
    snippet = chosen.snippet
    clip = _load_audio(Path(snippet.audio_path), base_root=state.bank_root)

    events_gain_db = float(params.get("events_gain_db", -1.0))
    mix = clip * _gain_to_amp(events_gain_db)
    output_duration_s = len(mix) / SR if len(mix) else 0.0
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix = mix * (0.98 / peak)
        peak = 0.98
    event = {
        "snippet_id": snippet.snippet_id,
        "label": snippet.species_common_name,
        "event_type": snippet.event_type,
        "audio_event_id": snippet.audio_event_id,
        "snippet_path": snippet.audio_path,
        "source_recording": snippet.recording_id,
        "onset_s": 0.0,
        "offset_s": round(output_duration_s, 3),
        "gain_db": events_gain_db,
        "pitch_shift_semitones": 0.0,
        "time_stretch_rate": 1.0,
        "fade_s": 0.0,
        "confidence": round(snippet.score, 4),
        "retrieval_score": round(chosen.retrieval_score, 4),
        "diel_bin": snippet.diel_bin,
        "season": snippet.season,
        "selection_reason": chosen.selection_reason,
    }

    metadata = {
        "layer": "layer_c",
        "method": "retrieval_clip",
        "species": species,
        "request": {
            "seed": run_seed,
            "season": season,
            "diel": diel,
            "duration_s": requested_duration_s,
            "duration_s_ignored": True,
            "count_requested": 1,
        },
        "audio": {
            "sample_rate": SR,
            "duration_s": output_duration_s,
            "peak": peak,
            "rms": float(np.sqrt(np.mean(np.square(mix)))) if mix.size else 0.0,
            "contains_ambient_bed": False,
        },
        "retrieval": {
            "index_path": str(state.index_path.relative_to(REPO_ROOT)),
            "selected_count": 1,
            "library_source": "audited real bird-call snippets only",
            "selected_snippet": event,
        },
        "mix": {
            "ambient_kind": "none",
            "ambient_gain_db": None,
            "events_gain_db": events_gain_db,
            "variation_enabled": False,
            "limitations": [
                "This frontend demo is a single retrieved Layer C clip; Layer D mixing is separate.",
                "Layer C events are real audited retrieval snippets, not from-scratch generated calls.",
            ],
        },
        "events": [event],
    }

    return {
        "wav_bytes": _wav_bytes(mix, SR),
        "mel_db": _mel_db(mix, SR),
        "metadata": metadata,
    }


def _load_index(path: Path) -> list[EventSnippet]:
    if path.suffix.lower() == ".json":
        return _load_json_index(path)
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


def _load_json_index(path: Path) -> list[EventSnippet]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    snippets: list[EventSnippet] = []
    for asset in doc.get("assets", []):
        attrs = asset.get("attributes") or {}
        snippets.append(
            EventSnippet(
                snippet_id=str(asset["id"]),
                event_type=str(attrs.get("event_type", "")),
                species_common_name=str(attrs.get("species_common_name", "")),
                species_scientific_name=str(attrs.get("species_scientific_name", "")),
                audio_event_id=str(attrs.get("audio_event_id", "")),
                audio_path=str(asset["audio_path"]),
                score=float(attrs.get("score") or 0.0),
                quality_score=_parse_float(attrs.get("quality_score")),
                diel_bin=str(attrs.get("diel_bin", "")),
                season=str(attrs.get("season", "unknown")),
                duration_s=float(attrs.get("duration_s") or 0.0),
                recording_id=str(attrs.get("recording_id", "")),
                event_start_seconds=_parse_float(attrs.get("event_start_seconds")),
                event_end_seconds=_parse_float(attrs.get("event_end_seconds")),
                source_manifest=str(attrs.get("source_manifest", "")),
                verdict=str(attrs.get("verdict", "")),
                notes=str(attrs.get("notes", "")),
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

    # Diversify by source recording before sampling. A flat _score gives every
    # snippet in the same (diel, season) bin near-identical points, so a plain
    # top-k pool collapses onto whatever single recording happens to have the
    # most high-confidence cuts — every seed then returns the same call. Round-
    # robin across recordings (best cut of each first, then second-best, ...)
    # so the pool spans many distinct sources while still front-loading quality.
    diversified = _diversify_by_recording(scored)
    pool_size = max(count * 8, 12)
    top_pool = diversified[:pool_size]
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


def _diversify_by_recording(scored: list[RetrievedSnippet]) -> list[RetrievedSnippet]:
    """Reorder a score-sorted list so source recordings are interleaved.

    Groups by ``recording_id`` (insertion order preserves descending score),
    then emits the best cut of each recording, then the second-best of each,
    and so on. The result keeps quality near the front while guaranteeing the
    leading entries span as many distinct recordings as exist.
    """

    groups: dict[str, list[RetrievedSnippet]] = {}
    for item in scored:
        groups.setdefault(item.snippet.recording_id, []).append(item)

    diversified: list[RetrievedSnippet] = []
    rank = 0
    while len(diversified) < len(scored):
        added = False
        for cuts in groups.values():
            if rank < len(cuts):
                diversified.append(cuts[rank])
                added = True
        if not added:
            break
        rank += 1
    return diversified


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


def _render_events(
    events: list[ScheduledEvent],
    *,
    target_duration_s: float,
    bank_root: Path | None = None,
) -> LayerResult:
    total_samples = int(round(target_duration_s * SR))
    layer = np.zeros(total_samples, dtype=np.float32)
    metadata_events = []

    for event in events:
        audio = _load_audio(Path(event.retrieved.snippet.audio_path), base_root=bank_root)
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


def _load_audio(path: Path, *, base_root: Path | None = None) -> np.ndarray:
    if not path.is_absolute():
        path = (base_root or REPO_ROOT) / path
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
