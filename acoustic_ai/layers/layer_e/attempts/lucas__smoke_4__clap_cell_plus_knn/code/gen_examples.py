"""Generate 5 held-out per-segment fused reports for qualitative review.

Runs AmbientFusedAnalyzer on 5 fixed val segments and writes examples/<id>.json
plus examples.md showing head A / head B / fused cell, agreement, OOD flag, and
confidence vs ground truth. Demonstrates the agreement-as-confidence signal.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from ambient_fused import AmbientFusedAnalyzer
from paths import SEGMENTS_DIR, SPLITS_DIR, ATTEMPT_DIR

IDS = [
    "1536463_clip009_s001", "215190_clip007_s000", "215467_clip002_s000",
    "216086_clip015_s000", "216470_clip010_s000",
]

def main() -> None:
    gt = pd.read_csv(SPLITS_DIR / "val.csv").set_index("segment_id")
    out = ATTEMPT_DIR / "examples"; out.mkdir(exist_ok=True)
    a = AmbientFusedAnalyzer(k=5)
    rows = []
    for sid in IDS:
        rep = a.analyze(SEGMENTS_DIR / f"{sid}.wav")
        g = gt.loc[sid]
        rep["ground_truth"] = {"cell": g.cell, "season": g.season, "diel": g.diel_bin}
        json.dump(rep, open(out / f"{sid}.json", "w"), indent=2)
        cok = "Y" if rep["predicted_cell"] == g.cell else "N"
        agree = "agree" if rep["head_agreement"] else "DISAGREE"
        ood = "OOD" if rep["ood_flag"] else "-"
        rows.append((sid, g.cell, rep["head_a_cell"], rep["head_b_cell"],
                     rep["predicted_cell"], cok, agree, "{:.2f}".format(rep["confidence"]), ood))
    lines = [
        "# smoke_4 fused - example predictions",
        "",
        "5 held-out val segments, fused (w_a=w_b=0.5, k=5).",
        "cell: Y = fused cell exactly matches ground truth. agree = the two heads picked the same cell.",
        "",
        "| segment | true cell | head A (cell) | head B (knn) | fused | cell | heads | conf | ood |",
        "|---|---|---|---|---|:--:|:--:|---|:--:|",
    ]
    for sid, tc, ha, hb, fu, c, ag, cf, od in rows:
        lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(sid, tc, ha, hb, fu, c, ag, cf, od))
    ncell = sum(1 for r in rows if r[5] == "Y")
    nagree = sum(1 for r in rows if r[6] == "agree")
    lines += ["", "**Tally:** exact cell {}/5; heads agreed on {}/5.".format(ncell, nagree),
              "When the two independent heads agree, accuracy is far higher (eval: 0.565 vs 0.336) - agreement is the usable confidence/OOD signal.", ""]
    (out / "examples.md").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
