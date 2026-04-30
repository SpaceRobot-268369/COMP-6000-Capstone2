"""Layer A ambient bed selector for revised generation mode.

Layer A is retrieval-first: choose a real Site 257 clip whose metadata is close
to the requested environmental conditions, then use it as background texture.
"""

from __future__ import annotations

import base64
import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE_DIR = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a"
MANIFEST_PATH = RESOURCE_DIR / "site_257_training_manifest.csv"

NUMERIC_WEIGHTS = {
    "temperature_c": 1.2,
    "humidity_pct": 0.8,
    "wind_speed_ms": 1.0,
    "precipitation_mm": 1.2,
    "solar_radiation_wm2": 0.6,
    "surface_pressure_kpa": 0.4,
    "temp_max_c": 0.6,
    "temp_min_c": 0.6,
    "precipitation_daily_mm": 0.8,
    "wind_max_ms": 0.7,
    "days_since_rain": 0.7,
    "daylight_hours": 0.7,
    "hour_local": 1.0,
}

CATEGORICAL_PENALTIES = {
    "season": 3.0,
    "sample_bin": 4.0,
}

MONTH_WEIGHT = 1.8
_QUALITY_WARNING_SHOWN = False
TIME_BINS = ["dawn", "morning", "afternoon", "night"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass(frozen=True)
class AmbientCandidate:
    path: Path
    row: dict
    score: float = 0.0
    env_score: float = 0.0
    quality_penalty: float = 0.0
    peak_ratio_db: Optional[float] = None
    transient_rate: Optional[float] = None


def _to_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip_path_from_row(row: dict) -> Path:
    return PROJECT_ROOT / row["clip_path"]


def load_candidates(manifest_path: Path = MANIFEST_PATH) -> list[AmbientCandidate]:
    """Load manifest rows whose referenced .webm clip exists locally."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    candidates: list[AmbientCandidate] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            path = _clip_path_from_row(row)
            if path.exists() and path.stat().st_size > 0:
                candidates.append(AmbientCandidate(path=path, row=row))

    if not candidates:
        raise FileNotFoundError(
            "No downloaded Layer A clips found. Run script/download_site_257_clips.py first."
        )

    return candidates


def _stats(candidates: Iterable[AmbientCandidate]) -> dict[str, tuple[float, float]]:
    values: dict[str, list[float]] = {key: [] for key in NUMERIC_WEIGHTS}
    for candidate in candidates:
        for key in NUMERIC_WEIGHTS:
            value = _to_float(candidate.row.get(key))
            if value is not None:
                values[key].append(value)

    out: dict[str, tuple[float, float]] = {}
    for key, vals in values.items():
        if not vals:
            out[key] = (0.0, 1.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
        out[key] = (mean, math.sqrt(var) or 1.0)
    return out


def _distance(env: dict, candidate: AmbientCandidate, stats: dict[str, tuple[float, float]]) -> float:
    row = candidate.row
    score = 0.0

    for key, weight in NUMERIC_WEIGHTS.items():
        if key not in env:
            continue
        query = _to_float(env.get(key))
        actual = _to_float(row.get(key))
        if query is None or actual is None:
            continue
        _, std = stats[key]
        score += weight * ((query - actual) / std) ** 2

    for key, penalty in CATEGORICAL_PENALTIES.items():
        query = str(env.get(key, "")).strip().lower()
        actual = str(row.get(key, "")).strip().lower()
        if query and actual and query != actual:
            score += penalty

    query_month = _to_float(env.get("month"))
    actual_month = _to_float(row.get("month"))
    if query_month is not None and actual_month is not None:
        diff = abs(query_month - actual_month) % 12
        circular_diff = min(diff, 12 - diff)
        score += MONTH_WEIGHT * (circular_diff / 3.0) ** 2

    return score


def _month_distance(query_month: float, actual_month: float) -> float:
    diff = abs(query_month - actual_month) % 12
    return min(diff, 12 - diff)


def _prefilter_candidates(env: dict, candidates: list[AmbientCandidate]) -> list[AmbientCandidate]:
    """Prefer the requested diel bin and month before numeric env similarity.

    The UI exposes month as a primary control, so retrieval should not choose a
    different month merely because temperature or humidity happens to be closer.
    If exact coverage is missing, fall back to the nearest available month.
    """
    pool = candidates

    query_bin = str(env.get("sample_bin", "")).strip().lower()
    if query_bin:
        bin_matches = [
            candidate for candidate in pool
            if str(candidate.row.get("sample_bin", "")).strip().lower() == query_bin
        ]
        if bin_matches:
            pool = bin_matches

    query_month = _to_float(env.get("month"))
    if query_month is not None:
        month_matches = [
            candidate for candidate in pool
            if _to_float(candidate.row.get("month")) == query_month
        ]
        if month_matches:
            return month_matches

        month_distances = [
            (
                _month_distance(query_month, actual_month),
                candidate,
            )
            for candidate in pool
            if (actual_month := _to_float(candidate.row.get("month"))) is not None
        ]
        if month_distances:
            nearest = min(distance for distance, _ in month_distances)
            return [candidate for distance, candidate in month_distances if distance == nearest]

    return pool


def _summarise(candidate: AmbientCandidate) -> dict:
    row = candidate.row
    return {
        "score": round(float(candidate.score), 4),
        "env_score": round(float(candidate.env_score), 4),
        "quality_penalty": round(float(candidate.quality_penalty), 4),
        "peak_ratio_db": (
            round(float(candidate.peak_ratio_db), 2)
            if candidate.peak_ratio_db is not None else None
        ),
        "transient_rate": (
            round(float(candidate.transient_rate), 4)
            if candidate.transient_rate is not None else None
        ),
        "clip_path": str(candidate.path.relative_to(PROJECT_ROOT)),
        "recording_id": row.get("recording_id"),
        "clip_index": int(_to_float(row.get("clip_index"), 0) or 0),
        "season": row.get("season"),
        "month": int(_to_float(row.get("month"), 0) or 0),
        "sample_bin": row.get("sample_bin"),
        "temperature_c": _to_float(row.get("temperature_c")),
        "humidity_pct": _to_float(row.get("humidity_pct")),
        "wind_speed_ms": _to_float(row.get("wind_speed_ms")),
        "precipitation_mm": _to_float(row.get("precipitation_mm")),
    }


def _ambient_quality(path: Path) -> dict:
    """Estimate how suitable a clip is as a continuous background bed.

    Higher penalty means the clip likely contains foreground events such as
    sharp bird calls, knocks, or other sudden transients. This is deliberately
    lightweight and best-effort: if audio analysis dependencies are unavailable,
    return a neutral score so retrieval still works.
    """
    try:
        import numpy as np
        import librosa

        waveform, _ = librosa.load(str(path), sr=22_050, mono=True, duration=45.0)
        if waveform.size == 0:
            return {"penalty": 2.0, "peak_ratio_db": None, "transient_rate": None}

        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        rms = np.maximum(rms, 1e-8)
        db = librosa.amplitude_to_db(rms, ref=np.median)

        peak_ratio_db = float(np.percentile(db, 95) - np.percentile(db, 50))
        delta = np.diff(db, prepend=db[0])
        transient_rate = float(np.mean(delta > 6.0))

        penalty = 0.0
        if peak_ratio_db > 14:
            penalty += min((peak_ratio_db - 14) / 8, 2.0)
        if transient_rate > 0.08:
            penalty += min((transient_rate - 0.08) * 8, 2.0)

        return {
            "penalty": float(penalty),
            "peak_ratio_db": peak_ratio_db,
            "transient_rate": transient_rate,
        }
    except Exception as exc:
        global _QUALITY_WARNING_SHOWN
        if not _QUALITY_WARNING_SHOWN:
            print(f"[WARN] Layer A ambient quality analysis unavailable; using neutral quality scores ({exc}).")
            _QUALITY_WARNING_SHOWN = True
        return {"penalty": 0.0, "peak_ratio_db": None, "transient_rate": None}


def _spectrogram_png_b64(path: Path) -> str:
    """Build a compact mel-spectrogram preview for the selected ambient bed."""
    try:
        import io

        import librosa
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        waveform, sr = librosa.load(str(path), sr=22_050, mono=True, duration=30.0)
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            fmin=50,
            fmax=11_000,
        )
        mel_db = librosa.power_to_db(mel, ref=max, top_db=80)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
        ax.set_title("Layer A Ambient Bed Preview")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as exc:
        print(f"[WARN] Layer A spectrogram preview failed: {exc}")
        return ""


def select_ambient_bed(
    env: dict,
    manifest_path: Path = MANIFEST_PATH,
    top_k: int = 8,
    seed: Optional[int] = None,
) -> tuple[AmbientCandidate, list[dict]]:
    """Select a real ambient clip closest to requested conditions."""
    all_candidates = load_candidates(manifest_path)
    candidates = _prefilter_candidates(env, all_candidates)
    stats = _stats(all_candidates)
    env_ranked = sorted(
        (AmbientCandidate(path=c.path, row=c.row, env_score=_distance(env, c, stats)) for c in candidates),
        key=lambda candidate: candidate.env_score,
    )

    # Analyse only the best environmental matches so quality filtering stays
    # cheap. This rejects spiky foreground-event clips without scanning all data.
    env_pool = env_ranked[: max(top_k, min(40, len(env_ranked)))]
    ranked: list[AmbientCandidate] = []
    for candidate in env_pool:
        quality = _ambient_quality(candidate.path)
        quality_penalty = float(quality["penalty"])
        ranked.append(AmbientCandidate(
            path=candidate.path,
            row=candidate.row,
            score=candidate.env_score + quality_penalty,
            env_score=candidate.env_score,
            quality_penalty=quality_penalty,
            peak_ratio_db=quality["peak_ratio_db"],
            transient_rate=quality["transient_rate"],
        ))
    ranked.sort(key=lambda candidate: candidate.score)

    shortlist = ranked[: max(1, min(top_k, len(ranked)))]
    chosen = random.Random(seed).choice(shortlist)
    matches = [_summarise(candidate) for candidate in shortlist[:5]]
    return chosen, matches


def coverage_report(candidates: Optional[list[AmbientCandidate]] = None) -> dict:
    """Summarise downloaded Layer A coverage by month and diel bin."""
    candidates = candidates or load_candidates()
    counts = Counter()
    recording_ids: set[str] = set()

    for candidate in candidates:
        row = candidate.row
        month = int(_to_float(row.get("month"), 0) or 0)
        sample_bin = str(row.get("sample_bin", "")).strip().lower()
        if 1 <= month <= 12 and sample_bin:
            counts[(month, sample_bin)] += 1
        recording_id = str(row.get("recording_id", "")).strip()
        if recording_id:
            recording_ids.add(recording_id)

    by_month = {}
    missing = []
    for month in range(1, 13):
        by_month[str(month)] = {}
        for sample_bin in TIME_BINS:
            n = counts[(month, sample_bin)]
            by_month[str(month)][sample_bin] = n
            if n == 0:
                missing.append({"month": month, "sample_bin": sample_bin})

    return {
        "candidate_clips": len(candidates),
        "recordings": len(recording_ids),
        "by_month": by_month,
        "missing": missing,
    }


def recommend_download_counts(
    filtered_csv: Path = RESOURCE_DIR / "site_257_filtered_items.csv",
    max_recommendations: int = 48,
) -> list[dict]:
    """Recommend filtered CSV counts that fill currently missing month/bin cells."""
    coverage = coverage_report()
    missing = {(item["month"], item["sample_bin"]) for item in coverage["missing"]}
    if not filtered_csv.exists() or not missing:
        return []

    recommendations = []
    seen: set[tuple[int, str]] = set()
    with filtered_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            local_date = str(row.get("sample_local_date", ""))
            try:
                month = int(local_date[5:7])
            except ValueError:
                continue
            sample_bin = str(row.get("sample_bin", "")).strip().lower()
            key = (month, sample_bin)
            if key not in missing or key in seen:
                continue
            recommendations.append({
                "count": int(row["count"]),
                "recording_id": row["id"],
                "month": month,
                "month_label": MONTH_LABELS[month - 1],
                "sample_bin": sample_bin,
                "sample_local_date": local_date,
            })
            seen.add(key)
            if len(recommendations) >= max_recommendations:
                break

    return recommendations


def generate_layer_a_response(env: dict, seed: Optional[int] = None) -> dict:
    """Return a JSON-ready Layer A response with base64 webm audio."""
    candidate, matches = select_ambient_bed(env, seed=seed)
    row = candidate.row
    audio_b64 = base64.b64encode(candidate.path.read_bytes()).decode("utf-8")
    image_b64 = _spectrogram_png_b64(candidate.path)
    coverage = coverage_report()

    selected = {
        "clip_path": str(candidate.path.relative_to(PROJECT_ROOT)),
        "recording_id": row.get("recording_id"),
        "clip_index": int(_to_float(row.get("clip_index"), 0) or 0),
        "clip_start_seconds": _to_float(row.get("clip_start_seconds")),
        "clip_end_seconds": _to_float(row.get("clip_end_seconds")),
        "clip_duration_seconds": _to_float(row.get("clip_duration_seconds")),
        "sample_local_date": row.get("sample_local_date"),
        "season": row.get("season"),
        "month": int(_to_float(row.get("month"), 0) or 0),
        "sample_bin": row.get("sample_bin"),
        "score": round(float(candidate.score), 4),
        "env_score": round(float(candidate.env_score), 4),
        "quality_penalty": round(float(candidate.quality_penalty), 4),
        "peak_ratio_db": (
            round(float(candidate.peak_ratio_db), 2)
            if candidate.peak_ratio_db is not None else None
        ),
        "transient_rate": (
            round(float(candidate.transient_rate), 4)
            if candidate.transient_rate is not None else None
        ),
        "env": {
            "temperature_c": _to_float(row.get("temperature_c")),
            "humidity_pct": _to_float(row.get("humidity_pct")),
            "wind_speed_ms": _to_float(row.get("wind_speed_ms")),
            "precipitation_mm": _to_float(row.get("precipitation_mm")),
            "days_since_rain": _to_float(row.get("days_since_rain")),
            "daylight_hours": _to_float(row.get("daylight_hours")),
        },
    }

    return {
        "ok": True,
        "mode": "layer_a_ambient_bed",
        "audio_b64": audio_b64,
        "audio_mime": "audio/webm",
        "audio_ext": "webm",
        "image_b64": image_b64,
        "selected": selected,
        "matches": matches,
        "coverage": {
            "candidate_clips": coverage["candidate_clips"],
            "recordings": coverage["recordings"],
        },
        "layer_status": {
            "ambient_bed": "retrieved",
            "weather_layer": "pending",
            "species_event_layer": "pending",
            "mixer": "pending",
        },
        "explanation": (
            "Layer A selected a real Site 257 clip as the ambient bed using "
            "month, diel bin, weighted environmental similarity, and a "
            "background-bed quality check that penalises sharp transient peaks."
        ),
    }


if __name__ == "__main__":
    report = coverage_report()
    print(f"Layer A clips: {report['candidate_clips']} | recordings: {report['recordings']}")
    for month in range(1, 13):
        cells = report["by_month"][str(month)]
        values = " ".join(f"{bin_name}:{cells[bin_name]:3d}" for bin_name in TIME_BINS)
        print(f"{MONTH_LABELS[month - 1]:>3}  {values}")

    recs = recommend_download_counts()
    if recs:
        print("\nRecommended next CSV counts:")
        for rec in recs[:24]:
            print(
                f"  count {rec['count']:>3}  id {rec['recording_id']:<8} "
                f"{rec['month_label']} {rec['sample_bin']:<10} {rec['sample_local_date']}"
            )
