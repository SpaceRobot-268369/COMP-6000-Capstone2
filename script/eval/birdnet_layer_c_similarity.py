#!/usr/bin/env python3
"""BirdNET embedding similarity for Layer C generated samples.

The script compares each generated sample against BirdNET embedding centroids
computed from manually approved training clips. It is a diagnostic signal, not
a replacement for manual species audit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def import_birdnet() -> tuple[Any, Any]:
    """Import BirdNET after applying a small compatibility alias if needed."""
    try:
        from perch_hoplite.db import sqlite_usearch_impl

        if (
            not hasattr(sqlite_usearch_impl, "SQLiteUsearchDB")
            and hasattr(sqlite_usearch_impl, "SQLiteUSearchDB")
        ):
            sqlite_usearch_impl.SQLiteUsearchDB = sqlite_usearch_impl.SQLiteUSearchDB
    except Exception:
        pass

    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.analyze.utils import iterate_audio_chunks

    cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
    cfg.LABELS_FILE = cfg.BIRDNET_LABELS_FILE
    cfg.SAMPLE_RATE = cfg.BIRDNET_SAMPLE_RATE
    cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH
    cfg.SIG_OVERLAP = 0.0
    cfg.AUDIO_SPEED = 1.0
    cfg.BANDPASS_FMIN = 0
    cfg.BANDPASS_FMAX = 15000
    cfg.TFLITE_THREADS = 1
    cfg.BATCH_SIZE = 1

    return cfg, iterate_audio_chunks


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def embedding_for_audio(path: Path, iterate_audio_chunks: Any) -> np.ndarray:
    chunks = []
    for _start, _end, embedding in iterate_audio_chunks(str(path), embeddings=True):
        chunks.append(np.asarray(embedding, dtype=np.float32))
    if not chunks:
        raise ValueError(f"No BirdNET embedding chunks extracted from {path}")
    return l2_normalize(np.mean(np.stack(chunks, axis=0), axis=0))


def collect_embeddings(
    rows: list[dict[str, str]],
    audio_column: str,
    iterate_audio_chunks: Any,
) -> tuple[list[np.ndarray], list[str]]:
    embeddings: list[np.ndarray] = []
    paths: list[str] = []
    for row in rows:
        audio_path = Path(row[audio_column])
        if not audio_path.exists():
            continue
        embeddings.append(embedding_for_audio(audio_path, iterate_audio_chunks))
        paths.append(str(audio_path))
    return embeddings, paths


def parse_training_spec(spec: str) -> tuple[str, Path, str]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "training spec must be species=csv_path=audio_column"
        )
    return parts[0], Path(parts[1]), parts[2]


def parse_generated_spec(spec: str) -> tuple[str, Path, str, Path]:
    parts = spec.split("=", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "generated spec must be species=csv_path=audio_column=output_csv"
        )
    return parts[0], Path(parts[1]), parts[2], Path(parts[3])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training",
        action="append",
        required=True,
        type=parse_training_spec,
        help="Repeatable: species=csv_path=audio_column",
    )
    parser.add_argument(
        "--generated",
        action="append",
        required=True,
        type=parse_generated_spec,
        help="Repeatable: species=csv_path=audio_column=output_csv",
    )
    parser.add_argument("--summary_csv", required=True, type=Path)
    parser.add_argument("--cache_npz", type=Path)
    args = parser.parse_args()

    _cfg, iterate_audio_chunks = import_birdnet()

    centroid_by_species: dict[str, np.ndarray] = {}
    training_count_by_species: dict[str, int] = {}

    for species, csv_path, audio_column in args.training:
        rows = read_rows(csv_path)
        embeddings, _paths = collect_embeddings(rows, audio_column, iterate_audio_chunks)
        if not embeddings:
            raise SystemExit(f"No training embeddings extracted for {species}")
        centroid_by_species[species] = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))
        training_count_by_species[species] = len(embeddings)
        print(f"{species}: {len(embeddings)} training embeddings")

    if args.cache_npz:
        args.cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.cache_npz,
            **{f"centroid_{k}": v for k, v in centroid_by_species.items()},
        )

    summary_rows = []
    species_names = list(centroid_by_species)

    for target_species, generated_csv, audio_column, output_csv in args.generated:
        rows = read_rows(generated_csv)
        output_rows = []

        for row in rows:
            audio_path = Path(row[audio_column])
            if not audio_path.exists():
                out = {**row, "birdnet_error": "missing_audio"}
                output_rows.append(out)
                continue

            embedding = embedding_for_audio(audio_path, iterate_audio_chunks)
            sims = {
                species: float(np.dot(embedding, centroid))
                for species, centroid in centroid_by_species.items()
            }
            ranked = sorted(sims.items(), key=lambda item: item[1], reverse=True)
            target_rank = [name for name, _sim in ranked].index(target_species) + 1

            out = {
                **row,
                "birdnet_target_species": target_species,
                "birdnet_top_species": ranked[0][0],
                "birdnet_top_similarity": round(ranked[0][1], 6),
                "birdnet_target_similarity": round(sims[target_species], 6),
                "birdnet_target_rank": target_rank,
                "birdnet_margin_vs_second": round(
                    ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1],
                    6,
                ),
            }
            for species in species_names:
                out[f"birdnet_similarity_{species}"] = round(sims[species], 6)
            output_rows.append(out)

        write_csv(output_csv, output_rows)

        valid = [r for r in output_rows if "birdnet_error" not in r]
        target_top1 = sum(1 for r in valid if str(r.get("birdnet_top_species")) == target_species)
        target_rank1_or2 = sum(1 for r in valid if int(r.get("birdnet_target_rank", 99)) <= 2)
        manual_pass = sum(1 for r in valid if str(r.get("verdict", "")).strip().lower() == "pass")
        manual_usable = sum(1 for r in valid if str(r.get("verdict", "")).strip().lower() in {"pass", "borderline"})
        mean_target_sim = float(np.mean([float(r["birdnet_target_similarity"]) for r in valid])) if valid else 0.0

        summary_rows.append(
            {
                "species": target_species,
                "generated_count": len(valid),
                "training_count": training_count_by_species[target_species],
                "birdnet_target_top1": target_top1,
                "birdnet_target_top1_rate": round(target_top1 / len(valid), 4) if valid else 0.0,
                "birdnet_target_rank1_or2": target_rank1_or2,
                "birdnet_target_rank1_or2_rate": round(target_rank1_or2 / len(valid), 4) if valid else 0.0,
                "mean_target_similarity": round(mean_target_sim, 6),
                "manual_pass_count": manual_pass,
                "manual_usable_count": manual_usable,
            }
        )

        print(f"Wrote {len(output_rows)} rows to {output_csv}")

    write_csv(args.summary_csv, summary_rows)
    print(json.dumps(summary_rows, indent=2))
    print(f"Wrote summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
