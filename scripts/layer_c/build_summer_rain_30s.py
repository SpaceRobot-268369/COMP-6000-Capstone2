import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path

ROOT = Path(".")
MANIFEST = ROOT / "resources/site_257_bowra-dry-a/site_257_training_manifest.csv"
OUT = ROOT / "smoke_test/layer_c/raw_clips/summer_rain"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

# summer + afternoon + has rain
df = df[
    (df["season"].astype(str).str.lower() == "summer") &
    (df["hour_local"] >= 12) &
    (df["hour_local"] <= 17) &
    (df["precipitation_mm"] > 0)
]

df = df.sample(frac=1, random_state=42)

count = 0

for _, row in df.iterrows():
    src = ROOT / row["clip_path"]

    if not src.exists():
        continue

    try:
        y, sr = librosa.load(src, sr=22050, mono=True, duration=30)

        if len(y) < 22050 * 30:
            continue

        out = OUT / f"summer_rain_{count:03d}.wav"
        sf.write(out, y[:22050*30], 22050)

        print("saved:", out)
        count += 1

        if count >= 50:
            break

    except Exception as e:
        print("skip:", src, e)

print("done:", count)