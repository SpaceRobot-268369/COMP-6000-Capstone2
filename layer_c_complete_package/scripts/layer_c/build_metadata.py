#!/usr/bin/env python3
"""
Build metadata.csv for Layer C audited clips.
"""

import argparse
from pathlib import Path
import csv

PROMPTS = {
    "summer_rain": "summer rain afternoon, realistic clean outdoor nature ambience, only rain and leaves, no speech, no music, no traffic, no machines",
    "winter_snow": "winter snow night, realistic quiet cold wind and snowfall ambience, no footsteps, no speech, no music, no machines",
    "forest_bird": "forest birds morning, realistic clean forest birds ambience, leaves and light wind, no speech, no music, no traffic, no machines",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", default="smoke_test/layer_c/audited_clips")
    parser.add_argument("--output_csv", default="smoke_test/layer_c/metadata.csv")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    rows = []

    for scene, prompt in PROMPTS.items():
        for wav in sorted((clips_dir / scene).glob("*.wav")):
            rows.append({
                "file_path": str(wav),
                "scene": scene,
                "prompt": prompt,
                "duration_sec": 30,
            })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "scene", "prompt", "duration_sec"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")

if __name__ == "__main__":
    main()
