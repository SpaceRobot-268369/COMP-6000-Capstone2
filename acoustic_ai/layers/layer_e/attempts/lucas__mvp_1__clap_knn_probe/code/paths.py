from __future__ import annotations

from pathlib import Path

ATTEMPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_DIR = ATTEMPT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"

AMBIENT_INDEX = (
    REPO_ROOT
    / "acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_index.csv"
)
SEGMENTS_DIR = (
    REPO_ROOT
    / "acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_segments"
)
REGISTRY = REPO_ROOT / "acoustic_ai/registry.yaml"
PROD_ATTEMPT_KEY = "lucas__prod_1__per_cell_loras"

# Trained season-probe checkpoint slot (DVC binary; metadata git).
CANDIDATE_DIR = REPO_ROOT / "model/candidates/lucas/mvp_1__clap_knn_probe"
PROBE_PATH = CANDIDATE_DIR / "season_probe.pt"

SEASON_ORDER = ["spring", "summer", "autumn", "winter"]
DIEL_ORDER = ["dawn", "morning", "afternoon", "night"]

CELL_ORDER = [
    f"{season}_{diel}"
    for season in SEASON_ORDER
    for diel in DIEL_ORDER
]


def season_of(cell: str) -> str:
    return cell.split("_", 1)[0]


def diel_of(cell: str) -> str:
    return cell.split("_", 1)[1]
