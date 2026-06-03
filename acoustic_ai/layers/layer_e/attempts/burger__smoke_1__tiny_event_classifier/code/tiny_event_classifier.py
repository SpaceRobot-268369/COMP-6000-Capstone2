"""Tiny Layer E-C event classifier.

Smoke-oriented model for classifying audited event snippets with hand-crafted
log-mel/acoustic features. It is intentionally lightweight so it can be trained
locally and used as a fallback or sanity-check alongside BirdNET.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import scipy.signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_MANIFESTS = [
    REPO_ROOT / "resources/site_257_bowra-dry-a/layer_c_retrieval_event_library_split_v1/final_pass_library_v1/layer_c_retrieval_final_pass_event_index.csv",
    REPO_ROOT / "resources/site_257_bowra-dry-a/layer_c_retrieval_event_library_split_v1/boobook_audit_v1/refined_call_units_v1/boobook_pass24_refined_call_units_index.csv",
]
DEFAULT_OUT_DIR = REPO_ROOT / "model/candidates/burger/smoke_1__tiny_event_classifier"
DEFAULT_SAMPLE_RATE = 22050
EVENT_BANDS_HZ = [
    ("low_background", 100.0, 450.0),
    ("boobook_band", 450.0, 1200.0),
    ("cuckoo_band", 2100.0, 4100.0),
    ("fairywren_band", 3500.0, 7600.0),
    ("high_texture", 7600.0, 10000.0),
]
CLASS_BANDS_HZ = {
    "southern_boobook": (450.0, 1200.0),
    "horsfields_bronze_cuckoo": (2100.0, 4100.0),
    "splendid_fairywren": (3500.0, 7600.0),
}


@dataclass(frozen=True)
class EventExample:
    snippet_id: str
    label: str
    audio_path: Path
    duration_s: float


@dataclass(frozen=True)
class ComplexExample:
    snippet_id: str
    label: str
    clean_audio_path: Path
    complex_audio_path: Path
    source_kind: str


def read_examples(manifests: list[Path]) -> list[EventExample]:
    examples: list[EventExample] = []
    seen: set[Path] = set()
    for manifest in manifests:
        with manifest.open(newline="") as f:
            for row in csv.DictReader(f):
                verdict = (row.get("verdict") or "Pass").strip().lower()
                if verdict and verdict != "pass":
                    continue
                rel_audio = row.get("audio_path") or ""
                audio_path = (REPO_ROOT / rel_audio).resolve()
                if not audio_path.exists() or audio_path in seen:
                    continue
                seen.add(audio_path)
                examples.append(
                    EventExample(
                        snippet_id=row.get("snippet_id") or audio_path.stem,
                        label=row.get("event_type") or row.get("species_common_name") or "unknown",
                        audio_path=audio_path,
                        duration_s=float(row.get("duration_s") or 0.0),
                    )
                )
    examples.sort(key=lambda ex: (ex.label, ex.snippet_id))
    return examples


def _stats(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values = values[None, :]
    return np.concatenate(
        [
            np.mean(values, axis=1),
            np.std(values, axis=1),
            np.min(values, axis=1),
            np.max(values, axis=1),
            np.percentile(values, 10, axis=1),
            np.percentile(values, 50, axis=1),
            np.percentile(values, 90, axis=1),
        ]
    )


def extract_features(
    audio_path: Path,
    *,
    sample_rate: int = 22050,
    n_mels: int = 64,
    fmin_hz: float = 300.0,
    fmax_hz: float = 10000.0,
) -> np.ndarray:
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return extract_features_from_waveform(
        y,
        sr,
        n_mels=n_mels,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
    )


def extract_features_from_waveform(
    y: np.ndarray,
    sr: int,
    *,
    n_mels: int = 64,
    fmin_hz: float = 300.0,
    fmax_hz: float = 10000.0,
) -> np.ndarray:
    if y.size == 0:
        return np.zeros(n_mels * 7 + 6 * 7 + 3 + len(EVENT_BANDS_HZ) * 7, dtype=np.float32)
    y = librosa.util.normalize(y.astype(np.float32))
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin_hz,
        fmax=min(fmax_hz, sr / 2),
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    band_features = event_band_features(y, sr)
    energy = np.array(
        [
            float(np.mean(y**2)),
            float(np.std(y)),
            float(np.max(np.abs(y))),
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            _stats(log_mel),
            _stats(rms),
            _stats(zcr),
            _stats(centroid),
            _stats(bandwidth),
            _stats(rolloff),
            _stats(flatness),
            energy,
            band_features,
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def event_band_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Frequency-band ratios for the project's current event classes."""
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    frame_total = np.sum(stft, axis=0) + 1e-10
    out: list[float] = []
    for _name, low, high in EVENT_BANDS_HZ:
        mask = (freqs >= low) & (freqs <= min(high, sr / 2))
        if not np.any(mask):
            ratio = np.zeros_like(frame_total)
        else:
            ratio = np.sum(stft[mask], axis=0) / frame_total
        out.extend(
            [
                float(np.mean(ratio)),
                float(np.std(ratio)),
                float(np.max(ratio)),
                float(np.percentile(ratio, 75)),
                float(np.percentile(ratio, 90)),
                float(np.mean(ratio > 0.25)),
                float(np.mean(ratio > 0.50)),
            ]
        )
    return np.asarray(out, dtype=np.float32)


def bandpass_filter(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    high_hz = min(high_hz, sr / 2 - 1.0)
    if high_hz <= low_hz:
        return y.astype(np.float32)
    sos = scipy.signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return scipy.signal.sosfiltfilt(sos, y).astype(np.float32)


def is_ovr_model(model: Any) -> bool:
    return isinstance(model, dict) and model.get("kind") == "bandpass_ovr"


def classify_waveform(
    model: Any,
    y: np.ndarray,
    sr: int,
    *,
    bandpass_candidates: bool = False,
) -> tuple[str, float, dict[str, float]]:
    if is_ovr_model(model):
        scores = bandpass_candidate_scores(model, y, sr)
        if not scores:
            return "unknown", 0.0, {}
        best_label = max(scores, key=scores.get)
        return best_label, scores[best_label], scores

    labels = [str(label) for label in model.classes_]
    if not bandpass_candidates:
        probs = model.predict_proba(extract_features_from_waveform(y, sr)[None, :])[0]
        best_idx = int(np.argmax(probs))
        return labels[best_idx], float(probs[best_idx]), {
            label: float(prob) for label, prob in zip(labels, probs)
        }

    scores: dict[str, float] = {}
    matched_scores: dict[str, float] = {}
    for label in labels:
        band = CLASS_BANDS_HZ.get(label)
        candidate_y = bandpass_filter(y, sr, *band) if band else y
        probs = model.predict_proba(extract_features_from_waveform(candidate_y, sr)[None, :])[0]
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        pred_confidence = float(probs[pred_idx])
        scores[label] = pred_confidence if pred_label == label else 0.0
        if pred_label == label:
            matched_scores[label] = pred_confidence
    if matched_scores:
        best_label = max(matched_scores, key=matched_scores.get)
        return best_label, matched_scores[best_label], scores

    probs = model.predict_proba(extract_features_from_waveform(y, sr)[None, :])[0]
    best_idx = int(np.argmax(probs))
    return labels[best_idx], float(probs[best_idx]), {
        label: float(prob) for label, prob in zip(labels, probs)
    }


def bandpass_candidate_scores(model: Any, y: np.ndarray, sr: int) -> dict[str, float]:
    if is_ovr_model(model):
        scores: dict[str, float] = {}
        for label, clf in model["models"].items():
            band = CLASS_BANDS_HZ.get(label)
            candidate_y = bandpass_filter(y, sr, *band) if band else y
            probs = clf.predict_proba(extract_features_from_waveform(candidate_y, sr)[None, :])[0]
            class_values = list(clf.classes_)
            if 1 in class_values:
                positive_idx = class_values.index(1)
                scores[label] = float(probs[positive_idx])
        return scores

    labels = [str(label) for label in model.classes_]
    scores: dict[str, float] = {}
    for label in labels:
        band = CLASS_BANDS_HZ.get(label)
        candidate_y = bandpass_filter(y, sr, *band) if band else y
        probs = model.predict_proba(extract_features_from_waveform(candidate_y, sr)[None, :])[0]
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        if pred_label == label:
            scores[label] = float(probs[pred_idx])
    return scores


def train_model(
    examples: list[EventExample],
    *,
    test_size: float = 0.25,
    random_seed: int = 42,
    augment: bool = False,
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    if len({ex.label for ex in examples}) < 2:
        raise ValueError("Need at least two labels to train the smoke classifier.")

    x, y = feature_matrix(examples, augment=augment, random_seed=random_seed)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    proba = model.predict_proba(x[test_idx])
    labels = list(model.classes_)

    prediction_rows: list[dict[str, Any]] = []
    for row_idx, true, predicted, probs in zip(test_idx, y[test_idx], pred, proba):
        prediction_rows.append(
            {
                "snippet_id": f"feature_row_{int(row_idx)}",
                "audio_path": "",
                "true_label": true,
                "predicted_label": predicted,
                "confidence": round(float(np.max(probs)), 6),
                "correct": bool(true == predicted),
            }
        )

    report = classification_report(y[test_idx], pred, output_dict=True, zero_division=0)
    metrics = {
        "model": "StandardScaler + LogisticRegression",
        "sample_rate": 22050,
        "n_examples": len(examples),
        "n_feature_rows": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "classes": labels,
        "class_counts": {label: int(np.sum(y == label)) for label in labels},
        "accuracy": round(float(accuracy_score(y[test_idx], pred)), 6),
        "macro_f1": round(float(f1_score(y[test_idx], pred, average="macro")), 6),
        "weighted_f1": round(float(f1_score(y[test_idx], pred, average="weighted")), 6),
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y[test_idx], pred, labels=labels).tolist(),
        },
        "smoke_pass": bool(
            accuracy_score(y[test_idx], pred) >= 0.85
            and f1_score(y[test_idx], pred, average="macro") >= 0.85
        ),
    }
    return model, metrics, prediction_rows


def fit_all_model(
    examples: list[EventExample],
    *,
    random_seed: int = 42,
    augment: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Train on every clean example.

    This is the deployment-style smoke checkpoint requested after the initial
    stratified split smoke: all clean snippets become training data, and
    evaluation moves to separately selected complex clips.
    """
    model, labels, class_counts = fit_bandpass_ovr_model(
        examples,
        random_seed=random_seed,
        augment=augment,
    )
    n_rows = sum(class_counts.values())
    return model, {
        "model": "bandpass one-vs-rest StandardScaler + LogisticRegression",
        "training_mode": "fit_all_clean_snippets",
        "sample_rate": 22050,
        "n_examples": len(examples),
        "n_feature_rows": int(n_rows),
        "augmentation": "padded_clean_event_windows" if augment else "none",
        "classes": labels,
        "ovr_rows_per_class": class_counts,
    }


def fit_bandpass_ovr_model(
    examples: list[EventExample],
    *,
    random_seed: int,
    augment: bool,
) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    labels = sorted({ex.label for ex in examples})
    rng = np.random.default_rng(random_seed)
    loaded: list[tuple[EventExample, np.ndarray, int]] = []
    for ex in examples:
        y, sr = librosa.load(str(ex.audio_path), sr=DEFAULT_SAMPLE_RATE, mono=True)
        loaded.append((ex, y, sr))

    models: dict[str, Pipeline] = {}
    row_counts: dict[str, int] = {}
    for label in labels:
        band = CLASS_BANDS_HZ[label]
        x_rows: list[np.ndarray] = []
        y_rows: list[int] = []
        for ex, y, sr in loaded:
            variants = [y]
            if augment:
                variants.extend(
                    pad_event_window(y, sr, target_s, rng)
                    for target_s in (2.5, 4.0)
                    for _ in range(2)
                )
            for variant in variants:
                filtered = bandpass_filter(variant, sr, *band)
                x_rows.append(extract_features_from_waveform(filtered, sr))
                y_rows.append(1 if ex.label == label else 0)
        clf = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_seed,
                    ),
                ),
            ]
        )
        clf.fit(np.vstack(x_rows), np.array(y_rows))
        models[label] = clf
        row_counts[label] = len(y_rows)

    return {
        "kind": "bandpass_ovr",
        "classes": labels,
        "bands_hz": CLASS_BANDS_HZ,
        "models": models,
    }, labels, row_counts


def feature_matrix(
    examples: list[EventExample],
    *,
    augment: bool,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for ex in examples:
        y, sr = librosa.load(str(ex.audio_path), sr=DEFAULT_SAMPLE_RATE, mono=True)
        rows.append(extract_features_from_waveform(y, sr))
        labels.append(ex.label)
        if augment:
            for target_s in (2.5, 4.0):
                for _ in range(2):
                    rows.append(extract_features_from_waveform(pad_event_window(y, sr, target_s, rng), sr))
                    labels.append(ex.label)
    return np.vstack(rows), np.array(labels)


def pad_event_window(y: np.ndarray, sr: int, target_s: float, rng: np.random.Generator) -> np.ndarray:
    target_len = max(1, int(target_s * sr))
    if len(y) >= target_len:
        return y[:target_len].astype(np.float32)
    out = rng.normal(0.0, 0.0015, target_len).astype(np.float32)
    max_start = target_len - len(y)
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    event = y.astype(np.float32)
    if np.max(np.abs(event)) > 0:
        event = event / np.max(np.abs(event))
    gain = float(rng.uniform(0.45, 0.9))
    out[start:start + len(event)] += gain * event
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out


def find_complex_examples(examples: list[EventExample], *, max_per_class: int = 12) -> list[ComplexExample]:
    """Find less-cropped local audio counterparts for clean snippets."""
    by_label: dict[str, list[ComplexExample]] = {}
    seen_complex: set[Path] = set()
    for ex in examples:
        candidates: list[tuple[str, Path]] = []
        parent = ex.audio_path.parent
        for name in ["audio_full.wav", "audio_extended.wav", "audio.wav"]:
            p = parent / name
            if p.exists() and p.resolve() != ex.audio_path.resolve():
                candidates.append((name.removesuffix(".wav"), p.resolve()))

        # Boobook refined snippets live in a flat refined_call_units_v1 folder;
        # match their audio_event_id to full-audio item folders and 30 s blends.
        if ex.label == "southern_boobook":
            event_id = ex.snippet_id.rsplit("_", 1)[-1]
            boobook_root = (
                REPO_ROOT
                / "resources/site_257_bowra-dry-a/layer_c_retrieval_event_library_split_v1/boobook_audit_v1"
            )
            for p in sorted((boobook_root / "items").glob(f"*_audioevent_{event_id}/audio_full.wav")):
                candidates.append(("audio_full", p.resolve()))
            for p in sorted((boobook_root / "blend_450_1200_full30_v1").glob(f"*_audioevent_{event_id}_blend_450_1200_full30.wav")):
                candidates.append(("full30_blend", p.resolve()))

        for kind, path in candidates:
            if path in seen_complex:
                continue
            seen_complex.add(path)
            by_label.setdefault(ex.label, []).append(
                ComplexExample(
                    snippet_id=ex.snippet_id,
                    label=ex.label,
                    clean_audio_path=ex.audio_path,
                    complex_audio_path=path,
                    source_kind=kind,
                )
            )
            break

    out: list[ComplexExample] = []
    for label in sorted(by_label):
        out.extend(by_label[label][:max_per_class])
    return out


def save_training_outputs(
    out_dir: Path,
    model: Pipeline,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        fieldnames = ["snippet_id", "audio_path", "true_label", "predicted_label", "confidence", "correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def save_fit_all_outputs(out_dir: Path, model: Pipeline, metrics: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def write_complex_manifest(out_dir: Path, examples: list[ComplexExample]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "complex_test_manifest.csv"
    with path.open("w", newline="") as f:
        fieldnames = ["snippet_id", "label", "source_kind", "clean_audio_path", "complex_audio_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex in examples:
            writer.writerow(
                {
                    "snippet_id": ex.snippet_id,
                    "label": ex.label,
                    "source_kind": ex.source_kind,
                    "clean_audio_path": str(ex.clean_audio_path.relative_to(REPO_ROOT)),
                    "complex_audio_path": str(ex.complex_audio_path.relative_to(REPO_ROOT)),
                }
            )
    return path


def predict_file(model_path: Path, audio_path: Path) -> dict[str, Any]:
    model = joblib.load(model_path)
    y, sr = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE, mono=True)
    label, confidence, probabilities = classify_waveform(model, y, sr)
    return {
        "audio_path": str(audio_path),
        "label": label,
        "confidence": round(confidence, 6),
        "probabilities": {k: round(v, 6) for k, v in probabilities.items()},
    }


def detect_windows(
    model_path: Path,
    audio_path: Path,
    *,
    window_s: float = 2.5,
    hop_s: float = 0.5,
    min_confidence: float = 0.65,
    energy_gate_percentile: float = 60.0,
    bandpass_candidates: bool = False,
    sample_rate: int = 22050,
) -> dict[str, Any]:
    model = joblib.load(model_path)
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    win = max(1, int(window_s * sr))
    hop = max(1, int(hop_s * sr))
    rows: list[dict[str, Any]] = []
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    starts = list(range(0, max(1, len(y) - win + 1), hop))
    window_energy = [
        float(np.sqrt(np.mean(y[start:start + win] ** 2) + 1e-12))
        for start in starts
    ]
    energy_gate = float(np.percentile(window_energy, energy_gate_percentile)) if window_energy else 0.0
    for start, rms in zip(starts, window_energy):
        if rms < energy_gate:
            continue
        chunk = y[start:start + win]
        if bandpass_candidates:
            candidate_scores = bandpass_candidate_scores(model, chunk, sr)
            for label, confidence in candidate_scores.items():
                if confidence >= min_confidence:
                    rows.append(
                        {
                            "label": label,
                            "confidence": round(confidence, 6),
                            "onset_s": round(start / sr, 3),
                            "offset_s": round(min(len(y), start + win) / sr, 3),
                            "source": "tiny_event_classifier_bandpass",
                        }
                    )
        else:
            label, confidence, _scores = classify_waveform(model, chunk, sr)
            if confidence >= min_confidence:
                rows.append(
                    {
                        "label": label,
                        "confidence": round(confidence, 6),
                        "onset_s": round(start / sr, 3),
                        "offset_s": round(min(len(y), start + win) / sr, 3),
                        "source": "tiny_event_classifier",
                    }
                )
    return {
        "audio_path": str(audio_path),
        "energy_gate": {
            "percentile": energy_gate_percentile,
            "rms": round(energy_gate, 8),
        },
        "bandpass_candidates": bandpass_candidates,
        "detections": merge_detections(rows),
    }


def evaluate_complex(
    model_path: Path,
    complex_examples: list[ComplexExample],
    *,
    out_dir: Path,
    window_s: float = 2.5,
    hop_s: float = 0.5,
    min_confidence: float = 0.55,
    energy_gate_percentile: float = 65.0,
    bandpass_candidates: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path)
    rows: list[dict[str, Any]] = []
    for ex in complex_examples:
        y, sr = librosa.load(str(ex.complex_audio_path), sr=DEFAULT_SAMPLE_RATE, mono=True)
        whole_label, whole_confidence, _scores = classify_waveform(
            model,
            y,
            sr,
            bandpass_candidates=bandpass_candidates,
        )
        detection_report = detect_windows(
            model_path,
            ex.complex_audio_path,
            window_s=window_s,
            hop_s=hop_s,
            min_confidence=min_confidence,
            energy_gate_percentile=energy_gate_percentile,
            bandpass_candidates=bandpass_candidates,
        )
        detections = detection_report["detections"]
        target_detections = [d for d in detections if d["label"] == ex.label]
        non_target_detections = [d for d in detections if d["label"] != ex.label]
        rows.append(
            {
                "snippet_id": ex.snippet_id,
                "label": ex.label,
                "source_kind": ex.source_kind,
                "complex_audio_path": str(ex.complex_audio_path.relative_to(REPO_ROOT)),
                "whole_predicted_label": whole_label,
                "whole_confidence": round(float(whole_confidence), 6),
                "whole_correct": whole_label == ex.label,
                "n_detections": len(detections),
                "n_target_detections": len(target_detections),
                "n_non_target_detections": len(non_target_detections),
                "target_detected": bool(target_detections),
                "max_target_confidence": round(
                    max([float(d["confidence"]) for d in target_detections], default=0.0),
                    6,
                ),
                "detections_json": json.dumps(detections),
            }
        )

    whole_accuracy = float(np.mean([row["whole_correct"] for row in rows])) if rows else 0.0
    target_recall = float(np.mean([row["target_detected"] for row in rows])) if rows else 0.0
    mean_non_target_detections = float(np.mean([row["n_non_target_detections"] for row in rows])) if rows else 0.0
    labels = sorted({row["label"] for row in rows})
    per_class = {}
    for label in labels:
        class_rows = [row for row in rows if row["label"] == label]
        per_class[label] = {
            "n": len(class_rows),
            "whole_accuracy": round(float(np.mean([row["whole_correct"] for row in class_rows])), 6),
            "target_detection_recall": round(float(np.mean([row["target_detected"] for row in class_rows])), 6),
        }
    metrics = {
        "evaluation": "complex_counterpart_audio",
        "n_complex_examples": len(rows),
        "whole_clip_accuracy": round(whole_accuracy, 6),
        "target_detection_recall": round(target_recall, 6),
        "mean_non_target_detections": round(mean_non_target_detections, 6),
        "complex_smoke_pass": bool(target_recall >= 0.90 and mean_non_target_detections <= 0.25),
        "complex_smoke_thresholds": {
            "min_target_detection_recall": 0.90,
            "max_mean_non_target_detections": 0.25,
        },
        "window_s": window_s,
        "hop_s": hop_s,
        "min_confidence": min_confidence,
        "energy_gate_percentile": energy_gate_percentile,
        "bandpass_candidates": bandpass_candidates,
        "per_class": per_class,
    }
    (out_dir / "complex_eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "complex_eval_predictions.csv").open("w", newline="") as f:
        fieldnames = [
            "snippet_id",
            "label",
            "source_kind",
            "complex_audio_path",
            "whole_predicted_label",
            "whole_confidence",
            "whole_correct",
            "n_detections",
            "n_target_detections",
            "n_non_target_detections",
            "target_detected",
            "max_target_confidence",
            "detections_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return metrics


def merge_detections(rows: list[dict[str, Any]], *, merge_gap_s: float = 0.75) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (r["label"], float(r["onset_s"]), -float(r["confidence"]))):
        if (
            merged
            and row["label"] == merged[-1]["label"]
            and float(row["onset_s"]) - float(merged[-1]["offset_s"]) <= merge_gap_s
        ):
            merged[-1]["offset_s"] = row["offset_s"]
            merged[-1]["confidence"] = round(max(float(merged[-1]["confidence"]), float(row["confidence"])), 6)
        else:
            merged.append(dict(row))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train and evaluate the smoke classifier.")
    train.add_argument("--manifest", action="append", type=Path, default=None)
    train.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--fit-all", action="store_true", help="Train on all clean examples; no held-out split.")
    train.add_argument("--augment", action="store_true", help="Enable padded-window augmentation.")

    complex_eval = sub.add_parser("eval-complex", help="Evaluate the trained model on less-cropped complex audio.")
    complex_eval.add_argument("--manifest", action="append", type=Path, default=None)
    complex_eval.add_argument("--model", type=Path, default=DEFAULT_OUT_DIR / "model.joblib")
    complex_eval.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    complex_eval.add_argument("--max-per-class", type=int, default=12)
    complex_eval.add_argument("--window-s", type=float, default=2.5)
    complex_eval.add_argument("--hop-s", type=float, default=0.5)
    complex_eval.add_argument("--min-confidence", type=float, default=0.55)
    complex_eval.add_argument("--energy-gate-percentile", type=float, default=65.0)
    complex_eval.add_argument("--no-bandpass-candidates", action="store_true")

    predict = sub.add_parser("predict", help="Classify one event snippet.")
    predict.add_argument("--model", type=Path, default=DEFAULT_OUT_DIR / "model.joblib")
    predict.add_argument("--audio", type=Path, required=True)

    detect = sub.add_parser("detect", help="Run sliding-window detection on a longer clip.")
    detect.add_argument("--model", type=Path, default=DEFAULT_OUT_DIR / "model.joblib")
    detect.add_argument("--audio", type=Path, required=True)
    detect.add_argument("--window-s", type=float, default=2.5)
    detect.add_argument("--hop-s", type=float, default=0.5)
    detect.add_argument("--min-confidence", type=float, default=0.65)
    detect.add_argument("--energy-gate-percentile", type=float, default=60.0)
    detect.add_argument("--bandpass-candidates", action="store_true")

    args = parser.parse_args()
    if args.cmd == "train":
        manifests = args.manifest or DEFAULT_MANIFESTS
        examples = read_examples([p if p.is_absolute() else REPO_ROOT / p for p in manifests])
        if args.fit_all:
            model, metrics = fit_all_model(examples, random_seed=args.seed, augment=args.augment)
            save_fit_all_outputs(args.out_dir, model, metrics)
            print(json.dumps({"out_dir": str(args.out_dir), **metrics}, indent=2))
        else:
            model, metrics, predictions = train_model(
                examples,
                test_size=args.test_size,
                random_seed=args.seed,
                augment=args.augment,
            )
            save_training_outputs(args.out_dir, model, metrics, predictions)
            print(json.dumps({"out_dir": str(args.out_dir), **metrics}, indent=2))
    elif args.cmd == "eval-complex":
        manifests = args.manifest or DEFAULT_MANIFESTS
        examples = read_examples([p if p.is_absolute() else REPO_ROOT / p for p in manifests])
        complex_examples = find_complex_examples(examples, max_per_class=args.max_per_class)
        manifest_path = write_complex_manifest(args.out_dir, complex_examples)
        metrics = evaluate_complex(
            args.model,
            complex_examples,
            out_dir=args.out_dir,
            window_s=args.window_s,
            hop_s=args.hop_s,
            min_confidence=args.min_confidence,
            energy_gate_percentile=args.energy_gate_percentile,
            bandpass_candidates=not args.no_bandpass_candidates,
        )
        print(json.dumps({"manifest": str(manifest_path), **metrics}, indent=2))
    elif args.cmd == "predict":
        print(json.dumps(predict_file(args.model, args.audio), indent=2))
    elif args.cmd == "detect":
        print(json.dumps(
            detect_windows(
                args.model,
                args.audio,
                window_s=args.window_s,
                hop_s=args.hop_s,
                min_confidence=args.min_confidence,
                energy_gate_percentile=args.energy_gate_percentile,
                bandpass_candidates=args.bandpass_candidates,
            ),
            indent=2,
        ))


if __name__ == "__main__":
    main()
