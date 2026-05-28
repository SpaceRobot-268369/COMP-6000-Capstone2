"""Layer B weather asset retrieval using pretrained CLAP embeddings."""

from pathlib import Path
from typing import Literal

import laion_clap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

WeatherType = Literal["wind", "rain", "thunder"]

ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_DIR = ATTEMPT_ROOT / "data" / "weather" / "metadata"


_MODEL = None


def get_clap_model():
    """Load CLAP once and reuse it."""
    global _MODEL

    if _MODEL is None:
        _MODEL = laion_clap.CLAP_Module(enable_fusion=False)
        _MODEL.load_ckpt()

    return _MODEL


def load_embedding_index(
    weather_type: WeatherType,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
):
    """Load weather embedding index for one weather type."""
    index_path = metadata_dir / f"weather_embedding_index_{weather_type}.npz"

    if not index_path.exists():
        raise FileNotFoundError(f"Weather embedding index not found: {index_path}")

    data = np.load(index_path, allow_pickle=True)

    return data["filenames"], data["embeddings"]


def retrieve_weather_asset(
    query_text: str,
    weather_type: WeatherType = "wind",
    top_k: int = 3,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
):
    """Retrieve the most semantically relevant weather assets."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    files, embeddings = load_embedding_index(weather_type, metadata_dir)

    model = get_clap_model()
    text_embedding = model.get_text_embedding([query_text])

    scores = cosine_similarity(text_embedding, embeddings)[0]

    ranked = sorted(
        zip(files, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "file": str(file),
            "score": float(score),
            "weather_type": weather_type,
            "query": query_text,
        }
        for file, score in ranked[:top_k]
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve Layer B weather assets.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--type", choices=["wind", "rain", "thunder"], default="wind")
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    results = retrieve_weather_asset(
        query_text=args.query,
        weather_type=args.type,
        top_k=args.top_k,
    )

    for result in results:
        print(f"{result['score']:.4f}\t{result['file']}")
