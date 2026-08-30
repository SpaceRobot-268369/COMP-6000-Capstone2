from pathlib import Path
import csv, subprocess, random

CSV = Path("resources/site_257_bowra-dry-a/boobook_crow_birdnet_results.csv")
OUT = Path("resources/my_layer_c_train")
OUT.mkdir(parents=True, exist_ok=True)

targets = {
    "Southern Boobook": ("boobook", 0.90, 150),
    "Australian Raven": ("raven", 0.70, 80),
}

rows = []
with CSV.open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["common_name"] in targets:
            label, min_conf, _ = targets[r["common_name"]]
            if float(r["confidence"]) >= min_conf:
                rows.append((label, r))

random.seed(42)

for common, (label, min_conf, max_n) in targets.items():
    selected = [r for lab, r in rows if lab == label]
    random.shuffle(selected)
    selected = selected[:max_n]

    out_dir = OUT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(selected, 1):
        src = Path(r["file"])
        start = max(0, float(r["start_time"]) - 2)
        dur = 8
        dst = out_dir / f"{label}_{i:04d}.wav"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(src),
            "-t", str(dur),
            "-ac", "1",
            "-ar", "16000",
            str(dst),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(label, len(list(out_dir.glob("*.wav"))))