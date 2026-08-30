from pathlib import Path
import csv

SETS = {
    "boobook": Path("resources/my_layer_c_train/boobook"),
    "raven": Path("resources/my_layer_c_train/raven"),
}

for label, folder in SETS.items():
    out = Path(f"resources/my_layer_c_train/{label}_manifest.csv")
    files = sorted(folder.glob("*.wav"))

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio_path", "caption"])
        w.writeheader()

        for p in files:
            caption = (
                "Southern Boobook owl call, nocturnal bird vocal event"
                if label == "boobook"
                else "Australian Raven call, harsh croaking caw vocalization"
            )
            w.writerow({"audio_path": str(p), "caption": caption})

    print(label, len(files), out)