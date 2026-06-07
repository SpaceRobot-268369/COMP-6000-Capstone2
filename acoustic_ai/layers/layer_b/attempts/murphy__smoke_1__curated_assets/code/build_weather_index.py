from pathlib import Path

import numpy as np
import laion_clap

MODEL = laion_clap.CLAP_Module(enable_fusion=False)
MODEL.load_ckpt()

ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ATTEMPT_ROOT / "data" / "weather"

WEATHER_TYPES = ["wind", "rain", "thunder"]

for weather_type in WEATHER_TYPES:

    folder = BASE_DIR / weather_type

    audio_files = [
        str(path)
        for path in folder.iterdir()
        if path.suffix == ".wav"
    ]

    print(f"\nBuilding index for: {weather_type}")
    print(f"Found {len(audio_files)} files")

    embeddings = MODEL.get_audio_embedding_from_filelist(
        x=audio_files,
        use_tensor=False
    )

    metadata_dir = BASE_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    save_path = metadata_dir / f"weather_embedding_index_{weather_type}.npz"

    np.savez(
        save_path,
        filenames=np.array(audio_files),
        embeddings=np.array(embeddings)
    )

    print(f"Saved -> {save_path}")
