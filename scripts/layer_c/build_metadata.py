import json
from pathlib import Path

ROOT = Path(".")
AUDITED = ROOT / "smoke_test/layer_c/audited_clips"
OUT = ROOT / "smoke_test/layer_c/metadata"

OUT.mkdir(parents=True, exist_ok=True)

scene_prompts = {
    "forest_bird": "forest birds morning natural ambience",
    "summer_rain": "summer rain natural ambience"
}

metadata = []

for scene, prompt in scene_prompts.items():
    scene_dir = AUDITED / scene

    if not scene_dir.exists():
        continue

    for f in scene_dir.glob("*.wav"):
        metadata.append({
            "audio": str(f).replace("\\", "/"),
            "text": prompt
        })

with open(OUT / "metadata.jsonl", "w", encoding="utf-8") as fp:
    for row in metadata:
        fp.write(json.dumps(row) + "\n")

print("metadata samples:", len(metadata))
print("saved:", OUT / "metadata.jsonl")