"""Shared helpers for the E-C CLAP probe."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[6]
PARAMS_PATH = ATTEMPT_ROOT / "params.yaml"
PHENOLOGY_PATH = ATTEMPT_ROOT / "data" / "species_phenology.csv"


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "manifest": "local_data/ec_species/manifests/ec_species_13class_no_magpie_manifest.csv",
        "labels": [
            "ninox_boobook",
            "laughing_kookaburra",
            "rhipidura_leucophrys",
            "psophodes_cristatus",
            "cincloramphus_mathewsi",
            "podargus_strigoides",
            "red_capped_robin",
            "anas_superciliosa",
            "australian_raven",
            "peaceful_dove",
            "galah",
            "crested_bellbird",
            "rainbow_bee_eater",
        ],
    },
    "training": {
        "seed": 42,
        "arch": "mlp",
        "hidden": 128,
        "epochs": 500,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "class_weighting": "inverse_frequency",
        "select_metric": "val_macro_f1",
    },
    "output": {
        "embedding_dir": "local_data/ec_species/embeddings/clap_13class_no_magpie",
        "model_dir": "model/production/layer_e_c_species_event_detector",
    },
}


def load_config() -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return DEFAULT_CONFIG
    with PARAMS_PATH.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return deep_merge(DEFAULT_CONFIG, loaded)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_species_phenology(path: Path = PHENOLOGY_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        table: dict[str, dict[str, Any]] = {}
        for row in rows:
            label = str(row.get("label", "")).strip()
            if not label:
                continue
            table[label] = {
                "common_name": clean_text(row.get("common_name")),
                "scientific_name": clean_text(row.get("scientific_name")),
                "diel_signal": clean_text(row.get("diel_signal")),
                "diel_confidence": parse_float(row.get("diel_confidence")),
                "season_signal": clean_text(row.get("season_signal")),
                "season_confidence": parse_float(row.get("season_confidence")),
                "habitat_signal": clean_text(row.get("habitat_signal")),
                "inference_notes": clean_text(row.get("inference_notes")),
                "source_url": clean_text(row.get("source_url")),
            }
        return table


def clean_text(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def parse_float(value: Any) -> float | None:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_probe(in_dim: int, num_classes: int, arch: str, hidden: int) -> torch.nn.Module:
    if arch == "linear":
        return torch.nn.Linear(in_dim, num_classes)
    if arch == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(hidden, num_classes),
        )
    raise ValueError(f"unknown probe arch: {arch}")
