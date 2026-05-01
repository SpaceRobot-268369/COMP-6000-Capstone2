"""Build the cleaned ambient-only segment pool for Layer A.

For each clip in the training manifest, run an audio-only stationarity
gate: per-frame features (mel-mean, RMS, spectral centroid, spectral
flatness, spectral flux, ZCR) → per-clip rolling median + MAD baseline →
mark any frame deviating > 3·MAD as anomalous → dilate ± 0.5 s → take
contiguous unmasked spans ≥ 20 s → slice into 20–60 s segments.

Annotations and BirdNET are deliberately not used here — they run as
post-hoc audits over the retained segments, not as gates. Rationale and
full design in .claude/context/dev/layer_a_change_log.md.

Usage (from project root):
  python3 acoustic_ai/precompute/build_ambient_index.py
  python3 acoustic_ai/precompute/build_ambient_index.py --workers 6
  python3 acoustic_ai/precompute/build_ambient_index.py --limit 20  # smoke test
  python3 acoustic_ai/precompute/build_ambient_index.py --overwrite
"""

import argparse
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, median_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.ambient.preprocess import SPEC_CFG, load_audio  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "site_257_training_manifest.csv"
OUT_DIR = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "ambient_segments"
INDEX_CSV = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "ambient_index.csv"

SR = SPEC_CFG["sample_rate"]
HOP = SPEC_CFG["hop_length"]
FPS = SR / HOP  # ~43.07 frames per second

# Tunable thresholds — single source of truth.
CFG = {
    "rolling_window_s":   30.0,   # baseline window for per-clip median + MAD
    "mad_threshold":      3.0,    # frame anomalous if any feature > 3·MAD
    "dilation_s":         0.5,    # extend each anomalous frame by ±0.5 s
    "min_span_s":         20.0,   # discard contiguous spans shorter than this
    "target_seg_s":       30.0,   # preferred segment length when slicing a span
    "max_seg_s":          60.0,   # cap segment length
    "rms_low_pct":        20,     # span RMS must lie within [p20, p80] of clip
    "rms_high_pct":       80,
    "max_frame_diff_z":   2.0,    # span frame-to-frame mel-diff must be <= this·median(clip)
}


@dataclass
class ClipMeta:
    clip_path: str
    recording_id: str
    clip_index: int
    sample_bin: str
    season: str
    hour_local: int
    month: int
    day_of_year: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--index-csv", type=Path, default=INDEX_CSV)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Process only N clips (smoke test).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_manifest(path: Path, limit: int | None) -> list[ClipMeta]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(ClipMeta(
                clip_path=row["clip_path"],
                recording_id=row["recording_id"],
                clip_index=int(row["clip_index"]),
                sample_bin=row["sample_bin"],
                season=row["season"],
                hour_local=int(row["hour_local"]),
                month=int(row["month"]),
                day_of_year=int(row["day_of_year"]),
            ))
            if limit and len(rows) >= limit:
                break
    return rows


def cyclic_encode(value: float, period: float) -> tuple[float, float]:
    theta = 2 * math.pi * value / period
    return math.sin(theta), math.cos(theta)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_frame_features(waveform: np.ndarray) -> dict[str, np.ndarray]:
    """Return dict of 1-D arrays, one entry per analysis frame."""
    import librosa

    n_fft = SPEC_CFG["n_fft"]
    hop = HOP

    # STFT magnitude — reused across features.
    S = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop))

    mel = librosa.feature.melspectrogram(
        S=S ** 2, sr=SR, n_mels=SPEC_CFG["n_mels"],
        fmin=SPEC_CFG["fmin"], fmax=SPEC_CFG["fmax"],
    )
    mel_db = librosa.power_to_db(mel, top_db=SPEC_CFG["top_db"])

    rms = librosa.feature.rms(S=S, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    zcr = librosa.feature.zero_crossing_rate(waveform, hop_length=hop)[0]

    # Spectral flux: sum of positive frame-to-frame magnitude diffs.
    diff = np.diff(S, axis=1, prepend=S[:, :1])
    flux = np.maximum(diff, 0).sum(axis=0)

    # Align lengths defensively.
    T = min(mel_db.shape[1], rms.size, centroid.size, flatness.size, zcr.size, flux.size)

    return {
        "mel_mean": mel_db[:, :T].mean(axis=0),
        "rms": rms[:T],
        "centroid": centroid[:T],
        "flatness": flatness[:T],
        "flux": flux[:T],
        "zcr": zcr[:T],
        "mel_db": mel_db[:, :T],  # full mel kept for stationarity check below
    }


def rolling_median_mad(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-element rolling median and MAD (median-absolute-deviation)."""
    if window % 2 == 0:
        window += 1
    med = median_filter(x, size=window, mode="reflect")
    mad = median_filter(np.abs(x - med), size=window, mode="reflect")
    return med, mad


def anomaly_mask(features: dict[str, np.ndarray]) -> np.ndarray:
    """Return a bool array — True where the frame is anomalous on any feature."""
    window = max(3, int(round(CFG["rolling_window_s"] * FPS)))
    threshold = CFG["mad_threshold"]
    eps = 1e-6

    feature_names = ["mel_mean", "rms", "centroid", "flatness", "flux", "zcr"]
    T = features[feature_names[0]].size
    mask = np.zeros(T, dtype=bool)

    for name in feature_names:
        x = features[name]
        med, mad = rolling_median_mad(x, window)
        z = np.abs(x - med) / (mad + eps)
        mask |= (z > threshold)

    return mask


def dilate_mask(mask: np.ndarray, radius_frames: int) -> np.ndarray:
    if radius_frames <= 0:
        return mask
    size = 2 * radius_frames + 1
    return maximum_filter1d(mask.astype(np.uint8), size=size).astype(bool)


def find_clean_spans(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    """Return list of (start_frame, end_frame_exclusive) where mask is False
    and the span is at least `min_frames` long."""
    spans = []
    i = 0
    T = mask.size
    while i < T:
        if mask[i]:
            i += 1
            continue
        j = i
        while j < T and not mask[j]:
            j += 1
        if j - i >= min_frames:
            spans.append((i, j))
        i = j
    return spans


def verify_span(
    span: tuple[int, int],
    features: dict[str, np.ndarray],
    clip_rms_lo: float,
    clip_rms_hi: float,
    clip_frame_diff_median: float,
) -> bool:
    s, e = span
    rms_span = features["rms"][s:e].mean()
    if not (clip_rms_lo <= rms_span <= clip_rms_hi):
        return False

    mel_db = features["mel_db"][:, s:e]
    if mel_db.shape[1] < 2:
        return False
    frame_diff = np.linalg.norm(np.diff(mel_db, axis=1), axis=0).mean()
    if frame_diff > CFG["max_frame_diff_z"] * clip_frame_diff_median:
        return False
    return True


def slice_span_into_segments(s: int, e: int) -> list[tuple[int, int]]:
    """Carve a span (frame indices) into segments of target_seg_s, capped at max_seg_s.

    Last segment absorbs any remainder up to max_seg_s; otherwise it's split."""
    target = int(round(CFG["target_seg_s"] * FPS))
    cap = int(round(CFG["max_seg_s"] * FPS))
    out = []
    i = s
    while i < e:
        remaining = e - i
        if remaining <= cap:
            out.append((i, e))
            break
        out.append((i, i + target))
        i += target
    return out


# ---------------------------------------------------------------------------
# Per-clip worker
# ---------------------------------------------------------------------------

def process_clip(meta: ClipMeta, out_dir: Path, overwrite: bool) -> dict:
    clip_path = PROJECT_ROOT / meta.clip_path
    wav_path = clip_path.with_suffix(clip_path.suffix + ".wav")
    src = wav_path if wav_path.exists() else clip_path

    try:
        waveform = load_audio(str(src))
    except Exception as exc:
        return {"clip": meta.clip_path, "status": "load_error", "error": str(exc), "segments": []}

    if waveform.size < int(CFG["min_span_s"] * SR):
        return {"clip": meta.clip_path, "status": "too_short", "segments": []}

    features = extract_frame_features(waveform)

    mask = anomaly_mask(features)
    mask = dilate_mask(mask, radius_frames=int(round(CFG["dilation_s"] * FPS)))

    rms = features["rms"]
    clip_rms_lo = float(np.percentile(rms, CFG["rms_low_pct"]))
    clip_rms_hi = float(np.percentile(rms, CFG["rms_high_pct"]))

    mel_db = features["mel_db"]
    frame_diffs = np.linalg.norm(np.diff(mel_db, axis=1), axis=0)
    clip_frame_diff_median = float(np.median(frame_diffs)) if frame_diffs.size else 1.0

    min_span_frames = int(round(CFG["min_span_s"] * FPS))
    spans = find_clean_spans(mask, min_span_frames)

    out_rows = []
    seg_count = 0
    for span in spans:
        if not verify_span(span, features, clip_rms_lo, clip_rms_hi, clip_frame_diff_median):
            continue
        for s, e in slice_span_into_segments(*span):
            if e - s < min_span_frames:
                continue
            t_start = s * HOP / SR
            t_end = e * HOP / SR
            sample_start = s * HOP
            sample_end = min(e * HOP, waveform.size)
            seg_audio = waveform[sample_start:sample_end]

            seg_id = f"{meta.recording_id}_clip{meta.clip_index:03d}_s{seg_count:03d}"
            seg_path = out_dir / f"{seg_id}.wav"
            if overwrite or not seg_path.exists():
                sf.write(seg_path, seg_audio, SR, subtype="PCM_16")

            hour_sin, hour_cos = cyclic_encode(meta.hour_local, 24)
            month_sin, month_cos = cyclic_encode(meta.month, 12)
            out_rows.append({
                "segment_id": seg_id,
                "source_clip": meta.clip_path,
                "t_start": round(t_start, 3),
                "t_end": round(t_end, 3),
                "duration_s": round(t_end - t_start, 3),
                "diel_bin": meta.sample_bin,
                "season": meta.season,
                "hour_sin": round(hour_sin, 6),
                "hour_cos": round(hour_cos, 6),
                "month_sin": round(month_sin, 6),
                "month_cos": round(month_cos, 6),
            })
            seg_count += 1

    return {
        "clip": meta.clip_path,
        "status": "ok",
        "n_anomalous_frames": int(mask.sum()),
        "n_total_frames": int(mask.size),
        "n_clean_spans": len(spans),
        "segments": out_rows,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

INDEX_FIELDS = [
    "segment_id", "source_clip", "t_start", "t_end", "duration_s",
    "diel_bin", "season", "hour_sin", "hour_cos", "month_sin", "month_cos",
]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.index_csv.parent.mkdir(parents=True, exist_ok=True)

    metas = load_manifest(args.manifest, args.limit)
    print(f"loaded {len(metas)} clips from {args.manifest.name}")

    all_rows: list[dict] = []
    stats = {"ok": 0, "load_error": 0, "too_short": 0, "total_segments": 0}

    if args.workers <= 1:
        for meta in tqdm(metas, desc="cleaning"):
            res = process_clip(meta, args.out_dir, args.overwrite)
            stats[res["status"]] = stats.get(res["status"], 0) + 1
            stats["total_segments"] += len(res["segments"])
            all_rows.extend(res["segments"])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_clip, m, args.out_dir, args.overwrite) for m in metas]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="cleaning"):
                res = fut.result()
                stats[res["status"]] = stats.get(res["status"], 0) + 1
                stats["total_segments"] += len(res["segments"])
                all_rows.extend(res["segments"])

    all_rows.sort(key=lambda r: r["segment_id"])
    with open(args.index_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nclips ok:          {stats['ok']}")
    print(f"clips load_error:  {stats['load_error']}")
    print(f"clips too_short:   {stats['too_short']}")
    print(f"total segments:    {stats['total_segments']}")
    print(f"index written to:  {args.index_csv}")
    print(f"segments dir:      {args.out_dir}")

    # Per-cell coverage report (diel_bin × season).
    from collections import Counter
    cells = Counter((r["diel_bin"], r["season"]) for r in all_rows)
    print("\nper-cell segment counts (diel_bin × season):")
    for (bin_, season), n in sorted(cells.items()):
        flag = "  ⚠ low" if n < 20 else ""
        print(f"  {bin_:<10} {season:<8} {n:>5}{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
