"""Run sliding-window species prediction over a longer audio file."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from clap_backbone import CLAPBackbone, TARGET_SR
from common import build_probe, device, load_config, load_species_phenology, project_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="Path to a longer audio file.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--hop-s", type=float, default=1.0)
    parser.add_argument("--merge-gap-s", type=float, default=1.0)
    parser.add_argument("--min-event-windows", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    result = detect_windows(
        args.audio,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        window_s=args.window_s,
        hop_s=args.hop_s,
        merge_gap_s=args.merge_gap_s,
        min_event_windows=args.min_event_windows,
    )
    printed = summarize_result(result) if args.summary_only else result
    text = json.dumps(printed, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(text)
    return 0


def detect_windows(
    audio_path: Path,
    *,
    checkpoint: Path | None = None,
    threshold: float = 0.55,
    window_s: float = 5.0,
    hop_s: float = 1.0,
    merge_gap_s: float = 1.0,
    min_event_windows: int = 7,
    backbone: CLAPBackbone | None = None,
) -> dict:
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if hop_s <= 0:
        raise ValueError("hop_s must be positive")
    if merge_gap_s < 0:
        raise ValueError("merge_gap_s must be non-negative")
    if min_event_windows < 1:
        raise ValueError("min_event_windows must be at least 1")

    cfg = load_config()
    labels = list(cfg["data"]["labels"])
    phenology_by_label = load_species_phenology()
    model_dir = project_path(cfg["output"]["model_dir"])
    checkpoint_path = checkpoint or model_dir / "best_probe.pt"
    run_device = device()

    saved = torch.load(checkpoint_path, map_location=run_device)
    probe = build_probe(
        int(saved["in_dim"]),
        len(labels),
        str(saved["arch"]),
        int(saved["hidden"]),
    ).to(run_device)
    probe.load_state_dict(saved["state_dict"])
    probe.eval()

    audio = load_audio_48k(audio_path)
    windows = build_windows(audio, window_s=window_s, hop_s=hop_s)
    clap = backbone or CLAPBackbone(device=str(run_device))

    with tempfile.TemporaryDirectory(prefix="ec_detect_windows_") as tmp:
        tmp_dir = Path(tmp)
        window_paths = []
        for idx, item in enumerate(windows):
            path = tmp_dir / f"window_{idx:05d}.wav"
            sf.write(path, item["audio"], TARGET_SR, subtype="PCM_16")
            window_paths.append(path)

        embeddings = clap.embed_audio(window_paths, verbose=True)

    features = torch.from_numpy(embeddings).float().to(run_device)
    with torch.no_grad():
        probs = torch.softmax(probe(features), dim=1).cpu().numpy()

    window_results = []
    detected_windows = []
    for idx, item in enumerate(windows):
        row = probs[idx]
        top_idx = int(np.argmax(row))
        top_label = labels[top_idx]
        confidence = float(row[top_idx])
        scores = {label: float(row[label_idx]) for label_idx, label in enumerate(labels)}
        result = {
            "window_index": idx,
            "start_s": item["start_s"],
            "end_s": item["end_s"],
            "top_label": top_label,
            "confidence": confidence,
            "detected": confidence >= threshold,
            "scores": scores,
        }
        window_results.append(result)
        if result["detected"]:
            detected_windows.append(result)

    effective_min_event_windows = min(min_event_windows, len(window_results)) if window_results else min_event_windows
    events = merge_detected_windows(
        detected_windows,
        merge_gap_s=merge_gap_s,
        min_event_windows=effective_min_event_windows,
        phenology_by_label=phenology_by_label,
    )

    return {
        "audio_path": str(audio_path),
        "duration_s": round(float(audio.size / TARGET_SR), 3),
        "window_s": window_s,
        "hop_s": hop_s,
        "threshold": threshold,
        "merge_gap_s": merge_gap_s,
        "min_event_windows": min_event_windows,
        "effective_min_event_windows": effective_min_event_windows,
        "trained_labels": labels,
        "num_windows": len(window_results),
        "num_detected_windows": len(detected_windows),
        "num_events": len(events),
        "events": events,
        "detected_windows": detected_windows,
        "windows": window_results,
    }


def load_audio_48k(path: Path) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32, copy=False)


def summarize_result(result: dict) -> dict:
    return {
        "audio_path": result["audio_path"],
        "duration_s": result["duration_s"],
        "window_s": result["window_s"],
        "hop_s": result["hop_s"],
        "threshold": result["threshold"],
        "merge_gap_s": result["merge_gap_s"],
        "min_event_windows": result["min_event_windows"],
        "effective_min_event_windows": result["effective_min_event_windows"],
        "trained_labels": result["trained_labels"],
        "num_windows": result["num_windows"],
        "num_detected_windows": result["num_detected_windows"],
        "num_events": result["num_events"],
        "events": result["events"],
    }


def merge_detected_windows(
    windows: list[dict],
    *,
    merge_gap_s: float,
    min_event_windows: int = 1,
    phenology_by_label: dict[str, dict] | None = None,
) -> list[dict]:
    by_label: dict[str, list[dict]] = {}
    for window in windows:
        by_label.setdefault(str(window["top_label"]), []).append(window)

    events = []
    for label, label_windows in by_label.items():
        label_windows = sorted(label_windows, key=lambda row: (row["start_s"], row["end_s"]))
        current = None
        for window in label_windows:
            if current is None:
                current = start_event(label, window)
                continue

            gap_s = float(window["start_s"]) - float(current["offset_s"])
            if gap_s <= merge_gap_s:
                extend_event(current, window)
            else:
                append_event(
                    events,
                    current,
                    min_event_windows=min_event_windows,
                    phenology_by_label=phenology_by_label,
                )
                current = start_event(label, window)

        if current is not None:
            append_event(
                events,
                current,
                min_event_windows=min_event_windows,
                phenology_by_label=phenology_by_label,
            )

    return sorted(events, key=lambda row: (row["onset_s"], row["offset_s"], row["label"]))


def append_event(
    events: list[dict],
    event: dict,
    *,
    min_event_windows: int,
    phenology_by_label: dict[str, dict] | None = None,
) -> None:
    if int(event["window_count"]) >= min_event_windows:
        events.append(finalize_event(event, phenology_by_label=phenology_by_label))


def start_event(label: str, window: dict) -> dict:
    confidence = float(window["confidence"])
    scores = {str(k): float(v) for k, v in dict(window.get("scores", {})).items()}
    return {
        "label": label,
        "onset_s": float(window["start_s"]),
        "offset_s": float(window["end_s"]),
        "confidence_sum": confidence,
        "confidence_max": confidence,
        "window_count": 1,
        "window_indices": [int(window["window_index"])],
        "score_sums": scores,
    }


def extend_event(event: dict, window: dict) -> None:
    confidence = float(window["confidence"])
    event["offset_s"] = max(float(event["offset_s"]), float(window["end_s"]))
    event["confidence_sum"] += confidence
    event["confidence_max"] = max(float(event["confidence_max"]), confidence)
    event["window_count"] += 1
    event["window_indices"].append(int(window["window_index"]))
    score_sums = event.setdefault("score_sums", {})
    for label, score in dict(window.get("scores", {})).items():
        key = str(label)
        score_sums[key] = float(score_sums.get(key, 0.0)) + float(score)


def finalize_event(event: dict, *, phenology_by_label: dict[str, dict] | None = None) -> dict:
    window_count = int(event["window_count"])
    confidence_mean = float(event["confidence_sum"]) / window_count
    score_sums = dict(event.get("score_sums", {}))
    species_matches = [
        {
            "label": label,
            "score": round(float(score_sum) / window_count, 6),
        }
        for label, score_sum in score_sums.items()
    ]
    species_matches.sort(key=lambda row: (-float(row["score"]), str(row["label"])))
    finalized = {
        "label": event["label"],
        "onset_s": round(float(event["onset_s"]), 3),
        "offset_s": round(float(event["offset_s"]), 3),
        "confidence_mean": round(confidence_mean, 6),
        "confidence_max": round(float(event["confidence_max"]), 6),
        "window_count": window_count,
        "window_indices": event["window_indices"],
        "species_matches": species_matches,
    }
    phenology = (phenology_by_label or {}).get(str(event["label"]))
    if phenology:
        finalized["phenology"] = phenology
    return finalized


def build_windows(audio: np.ndarray, *, window_s: float, hop_s: float) -> list[dict]:
    window_samples = int(round(window_s * TARGET_SR))
    hop_samples = int(round(hop_s * TARGET_SR))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_s and hop_s must produce at least one sample")

    if audio.size == 0:
        audio = np.zeros(window_samples, dtype=np.float32)

    starts = list(range(0, max(audio.size - window_samples + 1, 1), hop_samples))
    if starts[-1] + window_samples < audio.size:
        starts.append(max(audio.size - window_samples, 0))

    windows = []
    for start in starts:
        end = start + window_samples
        chunk = audio[start:end]
        if chunk.size < window_samples:
            chunk = np.pad(chunk, (0, window_samples - chunk.size))
        windows.append({
            "start_s": round(float(start / TARGET_SR), 3),
            "end_s": round(float(min(end, audio.size) / TARGET_SR), 3),
            "audio": chunk,
        })
    return windows


if __name__ == "__main__":
    raise SystemExit(main())
