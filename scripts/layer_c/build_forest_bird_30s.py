import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path

ROOT = Path(".")
MANIFEST = ROOT / "resources/site_257_bowra-dry-a/site_257_training_manifest.csv"
OUT = ROOT / "smoke_test/layer_c/raw_clips/forest_bird"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

df = df[
    (df["hour_local"] >= 5) &
    (df["hour_local"] <= 10)
].sample(frac=1, random_state=42)

count = 0

for _, row in df.iterrows():
    src = ROOT / row["clip_path"]

    if not src.exists():
        continue

    try:
        y, sr = librosa.load(src, sr=22050, mono=True, duration=30)

        if len(y) < 22050 * 30:
            continue

        out = OUT / f"forest_bird_{count:03d}.wav"
        sf.write(out, y[:22050*30], 22050)

        print("saved:", out)
        count += 1

        if count >= 50:
            break

    except Exception as e:
        print("skip:", src, e)

print("done:", count)