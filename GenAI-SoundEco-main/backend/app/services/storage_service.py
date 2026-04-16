from __future__ import annotations

from pathlib import Path

import json
import pandas as pd


def save_feature_table(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported feature table format: {path.suffix}")


def save_embedding_array(embedding: list[float], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(embedding), encoding="utf-8")


def save_embedding_matrix(embeddings: list[list[float]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(embeddings), encoding="utf-8")


def append_processing_log(record: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
