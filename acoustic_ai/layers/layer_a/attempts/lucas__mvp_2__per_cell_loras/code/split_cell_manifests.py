"""Split the MVP-2 dataset manifest into one CSV per (season, diel_bin) cell.

Each per-cell LoRA in this attempt trains on exactly one cell's clips, the
same recipe that worked for lucas__mvp_1_1__spring_night_replica. This script
slices the shared manifest into 16 subset manifests so the training loop can
point each LoRA at its cell.

Output filenames: cell_<season>_<diel_bin>.csv (e.g. cell_spring_night.csv),
written in the SAME directory as the source manifest. This matters: the
training script derives project_root as manifest_path.parent.parent.parent
.parent, so a per-cell manifest must sit at the same depth as manifest.csv
(a subdir would shift project_root and break the relative audio paths).

Usage (from acoustic_ai/):
    ../.venv/bin/python \
      layers/layer_a/attempts/lucas__mvp_2__per_cell_loras/code/split_cell_manifests.py \
      --manifest ../resources/site_257_bowra-dry-a/mvp2_per_cell_dataset/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Path to the source manifest.csv")
    p.add_argument("--out-dir", default=None,
                   help="Where to write per-cell CSVs. Defaults to the manifest's "
                        "own directory (required for correct project_root derivation).")
    args = p.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir) if args.out_dir else manifest.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in reader:
            by_cell[(row["season"], row["diel_bin"])].append(row)

    print(f"{'cell':<22s} {'train':>6s} {'val':>5s}  file")
    total = 0
    for (season, diel), rows in sorted(by_cell.items()):
        cell = f"{season}_{diel}"
        dst = out_dir / f"cell_{cell}.csv"
        with open(dst, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        n_train = sum(1 for r in rows if r.get("split") == "train")
        n_val = sum(1 for r in rows if r.get("split") == "val")
        total += len(rows)
        print(f"{cell:<22s} {n_train:>6d} {n_val:>5d}  {dst}")

    print(f"\n{len(by_cell)} cells, {total} clips total -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
