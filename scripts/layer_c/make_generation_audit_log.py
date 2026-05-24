import pandas as pd
from pathlib import Path

ROOT = Path("smoke_test/layer_c/generated")

rows = []

for scene in ["forest_bird", "summer_rain"]:
    scene_dir = ROOT / scene

    for f in sorted(scene_dir.glob("*.