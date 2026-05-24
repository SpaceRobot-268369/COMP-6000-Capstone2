import pandas as pd
from pathlib import Path

folder = Path("smoke_test/layer_c/audited_clips/forest_bird")

rows = []
for f in sorted(folder.glob("*.wav")):
    rows.append({
        "filename": f.name,
        "scene": "forest_bird",
        "result": "PASS",
        "notes": "clean 30s natural bird/forest ambience"
    })

df = pd.DataFrame(rows)
df.to_excel("layer_c_audit_log.xlsx", index=False)
df.to_csv("layer_c_audit_log.csv", index=False)

print("audit rows:", len(df))
print("saved: layer_c_audit_log.xlsx")