import os
import numpy as np
import laion_clap

MODEL = laion_clap.CLAP_Module(enable_fusion=False)
MODEL.load_ckpt()

BASE_DIR = "/workspace/acoustic_ai/data/weather"

WEATHER_TYPES = ["wind", "rain", "thunder"]

for weather_type in WEATHER_TYPES:

    folder = os.path.join(BASE_DIR, weather_type)

    audio_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".wav")
    ]

    print(f"\nBuilding index for: {weather_type}")
    print(f"Found {len(audio_files)} files")

    embeddings = MODEL.get_audio_embedding_from_filelist(
        x=audio_files,
        use_tensor=False
    )

    metadata_dir = os.path.join(BASE_DIR, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    save_path = os.path.join(
        metadata_dir,
        f"weather_embedding_index_{weather_type}.npz"
    )

    np.savez(
        save_path,
        filenames=np.array(audio_files),
        embeddings=np.array(embeddings)
    )

    print(f"Saved -> {save_path}")
