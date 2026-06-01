"""Generate 5 held-out per-segment example reports for qualitative review.

Picks 5 fixed val segments (seed-42 spread across distinct cells), runs the
CellMatcher (audio prototypes) on each, and writes examples/<id>.json plus a
human-readable examples.md comparing prediction vs ground truth.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from ambient_cell_match import CellMatcher
from paths import SEGMENTS_DIR, SPLITS_DIR, ATTEMPT_DIR

IDS = [
    "1536463_clip009_s001", "215190_clip007_s000", "215467_clip002_s000",
    "216086_clip015_s000", "216470_clip010_s000",
]

def main() -> None:
    gt = pd.read_csv(SPLITS_DIR / "val.csv").set_index("segment_id")
    out = ATTEMPT_DIR / "examples"; out.mkdir(exist_ok=True)
    m = CellMatcher(variant="audio")
    rows = []
    for sid in IDS:
        r = m.classify(SEGMENTS_DIR / f"{sid}.wav")
        g = gt.loc[sid]
        r["ground_truth"] = {"cell": g.cell, "season": g.season, "diel": g.diel_bin}
        json.dump(r, open(out / f"{sid}.json", "w"), indent=2)
        parts = []
        for t in r["topk"]:
            parts.append("{} ({:.2f})".format(t["cell"], t["score"]))
        top3 = ", ".join(parts)
        sok = "Y" if r["season"] == g.season else "N"
        dok = "Y" if r["diel"] == g.diel_bin else "N"
        cok = "Y" if r["predicted_cell"] == g.cell else "N"
        rows.append((sid, g.cell, r["predicted_cell"], "{:.2f}".format(r["confidence"]), sok, dok, cok, top3))
    lines = [
        "# smoke_2 cell_match - example predictions",
        "",
        "5 held-out val segments (never seen in anchor construction), audio-prototype variant.",
        "season/diel/cell columns: Y = matches ground truth, N = miss.",
        "",
        "| segment | true cell | predicted cell | conf | season | diel | cell | top-3 |",
        "|---|---|---|---|:--:|:--:|:--:|---|",
    ]
    for sid, tc, pc, cf, s, d, c, t3 in rows:
        lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(sid, tc, pc, cf, s, d, c, t3))
    nseason = sum(1 for r in rows if r[4] == "Y")
    ndiel = sum(1 for r in rows if r[5] == "Y")
    ncell = sum(1 for r in rows if r[6] == "Y")
    lines += ["", "**Tally:** season {}/5, diel {}/5, exact cell {}/5.".format(nseason, ndiel, ncell), ""]
    (out / "examples.md").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
