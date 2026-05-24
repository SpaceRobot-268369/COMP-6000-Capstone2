#!/usr/bin/env python3
"""
Copy PASS clips from raw_clips to audited_clips based on layer_c_audit_log.xlsx.
"""

import argparse
from pathlib import Path
import shutil
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_xlsx", default="layer_c_audit_log.xlsx")
    parser.add_argument("--raw_dir", default="smoke_test/layer_c/raw_clips")
    parser.add_argument("--output_dir", default="smoke_test/layer_c/audited_clips")
    args = parser.parse_args()

    df = pd.read_excel(args.audit_xlsx, sheet_name="Audit_Log")
    required = {"filename", "scene", "pass_fail"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in audit sheet: {missing}")

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    copied = 0

    for _, row in df.iterrows():
        status = str(row.get("pass_fail", "")).strip().upper()
        if status != "PASS":
            continue

        scene = str(row["scene"]).strip()
        filename = str(row["filename"]).strip()
        src = raw_dir / scene / filename
        dst = output_dir / scene / filename
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"[WARN] missing source: {src}")
            continue

        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied PASS clips: {copied}")
    print(f"Output: {output_dir}")

if __name__ == "__main__":
    main()
