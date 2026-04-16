from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_environment_data(file_path: str | Path) -> pd.DataFrame:
    """Load environment records from CSV or Parquet."""
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported environment data file format: {path.suffix}")


def clean_environment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light normalization to the environment dataframe."""
    cleaned = df.copy()
    cleaned.columns = [column.strip().lower() for column in cleaned.columns]
    return cleaned
