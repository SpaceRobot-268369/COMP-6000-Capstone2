"""Generate 5 held-out per-segment env-estimate reports for qualitative review.

Runs the AmbientRetriever (k=5) on 5 fixed query segments and writes
examples/<id>.json plus examples.md comparing estimated season/diel/hour/month
and neighbour evidence against ground truth.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from ambient_similarity import AmbientRetriever
from paths import SEGMENTS_DIR, SPLITS_DIR, ATTEMPT_DIR

IDS = [
    "1536463_clip009_s001", "215190_clip007_s000", "215467_clip002_s000",
    "216086_clip015_s000", "216470_clip010_s000",
]

def true_hour_month(row):
    h = (math.atan2(row.hour_sin, row.hour_cos) % (2*math.pi)) / (2*math.pi) * 24
    m = (math.atan2(row.month_sin, row.month_cos) % (2*math.pi)) / (2*math.pi) * 12 + 1
    return h, m

def main() -> None:
    gt = pd.read_csv(SPLITS_DIR / "query.csv").set_index("segment_id")
    out = ATTEMPT_DIR / "examples"; out.mkdir(exist_ok=True)
    r = AmbientRetriever(k=5)
    rows = []
    for sid in IDS:
        rep = r.query(SEGMENTS_DIR / f"{sid}.wav")
        g = gt.loc[sid]
        th, tm = true_hour_month(g)
        ec = rep["estimated_conditions"]
        rep["ground_truth"] = {"season": g.season, "diel_bin": g.diel_bin,
                                "hour": round(th, 1), "month": round(tm, 1)}
        json.dump(rep, open(out / f"{sid}.json", "w"), indent=2)
        sok = "Y" if ec["season"] == g.season else "N"
        dok = "Y" if ec["diel_bin"] == g.diel_bin else "N"
        herr = abs(((ec["hour"] - th + 12) % 24) - 12)
        merr = abs(((ec["month"] - tm + 6) % 12) - 6)
        rows.append((sid, g.season, ec["season"], sok, g.diel_bin, ec["diel_bin"], dok,
                     "{:.0f}/{:.0f} ({:.1f})".format(th, ec["hour"], herr),
                     "{:.0f}/{:.0f} ({:.1f})".format(tm, ec["month"], merr),
                     "{:.2f}".format(rep["confidence"])))
    lines = [
        "# smoke_3 knn_env - example predictions",
        "",
        "5 held-out query segments (excluded from the k-NN index), k=5.",
        "season/diel: Y = match. hour/month columns show true/est (abs error).",
        "",
        "| segment | true->est season | s | true->est diel | d | hour t/e (err) | month t/e (err) | conf |",
        "|---|---|:--:|---|:--:|---|---|---|",
    ]
    for sid, ts, es, s, td, ed, d, hh, mm, cf in rows:
        lines.append("| `{}` | {}->{} | {} | {}->{} | {} | {} | {} | {} |".format(sid, ts, es, s, td, ed, d, hh, mm, cf))
    nseason = sum(1 for r in rows if r[3] == "Y")
    ndiel = sum(1 for r in rows if r[6] == "Y")
    lines += ["", "**Tally:** season {}/5, diel {}/5. (hour/month errors in the columns above.)".format(nseason, ndiel), ""]
    (out / "examples.md").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
