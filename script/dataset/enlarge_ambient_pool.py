#!/usr/bin/env python3
"""Enlarge the Layer E ambient pool by batched stage -> extract -> discard.

Builds a NEW pool `ambient_pool_v2` = existing 1,982 segments + freshly extracted
segments from the additional site-257 recordings in site_257_filtered_items_ext.csv.
Source clips are downloaded per batch and DELETED after extraction (they are large,
untracked intermediates) so peak disk stays bounded.

Pipeline per batch (count range [s, e] over the ext CSV):
  1. stage clips from either:
       - A2O API via download_site_257_clips.py, or
       - the S3 raw mirror at dataset/original/.../downloaded_clips/
  2. build_training_manifest.py  --filtered-csv <batch slice> --env-csv ENV --output <batch manifest>
  3. build_ambient_index.py      --manifest <batch manifest> --out-dir POOL/ambient_segments --index-csv <batch index>
  4. rm the batch's downloaded_clips/site_257_item_<id> folders (discard)

NASA env is fetched ONCE up front (cheap, per-year). At the end, all batch index
rows + the existing ambient_index are concatenated into POOL/ambient_index.csv and
the existing 1,982 WAVs are copied in, so POOL is the complete enlarged pool.

Run from repo root, e.g. on serverB:
  python3 script/dataset/enlarge_ambient_pool.py --start 1 --end 5 --batch 5   # smoke
  python3 script/dataset/enlarge_ambient_pool.py                                # full (1..660)
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "resources" / "site_257_bowra-dry-a"
EXT_CSV = SITE / "site_257_filtered_items_ext.csv"
ENV_CSV = SITE / "site_257_env_data_ext.csv"
CLIPS_DIR = SITE / "downloaded_clips"
EXISTING_POOL = ROOT / "acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient"
POOL = SITE / "ambient_pool_v2"
SEG_DIR = POOL / "ambient_segments"
WORK = POOL / "_work"

PY = str((ROOT / "acoustic_ai/.venv/bin/python"))
DEFAULT_S3_CLIPS_PREFIX = (
    "s3://eco-acoustic-data.store.adelaideuni.cloud/"
    "dataset/original/site_257_bowra-dry-a/downloaded_clips"
)


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def check_s3_prefix_exists(uri: str, aws_profile: str, aws_region: str) -> bool:
    cmd = ["aws", "s3", "ls", uri]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])
    if aws_region:
        cmd.extend(["--region", aws_region])
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res.returncode == 0


def sync_s3_prefix(src_uri: str, dst: Path, aws_profile: str, aws_region: str) -> None:
    cmd = ["aws", "s3", "sync", "--only-show-errors", src_uri, str(dst)]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])
    if aws_region:
        cmd.extend(["--region", aws_region])
    run(cmd)


def stage_batch_from_s3(
    batch_rows: list[dict],
    clips_dir: Path,
    s3_clips_prefix: str,
    aws_profile: str,
    aws_region: str,
    fail_on_missing: bool,
) -> list[dict]:
    staged_rows: list[dict] = []
    missing: list[str] = []
    s3_clips_prefix = s3_clips_prefix.rstrip("/")

    for row in batch_rows:
        item_id = row["id"]
        folder_name = f"site_257_item_{item_id}"
        src = f"{s3_clips_prefix}/{folder_name}/"
        dst = clips_dir / folder_name

        if not check_s3_prefix_exists(src, aws_profile, aws_region):
            missing.append(item_id)
            print(f"[S3 SKIP] item {item_id}: no S3 folder at {src}", flush=True)
            continue

        sync_s3_prefix(src, dst, aws_profile, aws_region)
        staged_rows.append(row)

    if missing and fail_on_missing:
        raise RuntimeError(
            f"S3 mirror is missing {len(missing)} item folders: {', '.join(missing)}"
        )
    if missing:
        print(f"[S3] skipped {len(missing)} missing item folders", flush=True)
    print(f"[S3] staged {len(staged_rows)} of {len(batch_rows)} item folders", flush=True)
    return staged_rows


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames), list(r)


def write_slice(fieldnames: list[str], rows: list[dict], out: Path) -> None:
    # Re-number `count` 1..N so build_training_manifest/order is stable per slice.
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            row = dict(row)
            row["count"] = i
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1, help="First ext-CSV count (inclusive).")
    ap.add_argument("--end", type=int, default=0, help="Last ext-CSV count (inclusive); 0 = all.")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-env", action="store_true", help="Reuse an existing env CSV.")
    ap.add_argument("--keep-clips", action="store_true", help="Do not delete clips after extract.")
    ap.add_argument(
        "--clip-source",
        choices=("api", "s3"),
        default="api",
        help="Where to stage source clips from before extraction.",
    )
    ap.add_argument(
        "--s3-clips-prefix",
        default=DEFAULT_S3_CLIPS_PREFIX,
        help="S3 prefix containing site_257_item_<id>/ clip folders when --clip-source=s3.",
    )
    ap.add_argument(
        "--aws-profile",
        default="capstone2",
        help="AWS profile for S3 staging. Use empty string to rely on the default provider chain.",
    )
    ap.add_argument(
        "--aws-region",
        default="ap-southeast-2",
        help="AWS region for S3 staging. Use empty string to omit --region.",
    )
    ap.add_argument(
        "--fail-on-missing-s3",
        action="store_true",
        help="Fail instead of skipping recordings absent from the S3 raw mirror.",
    )
    args = ap.parse_args()

    fieldnames, all_rows = read_rows(EXT_CSV)
    n_total = len(all_rows)
    end = args.end or n_total
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    batch_indexes: list[Path] = []

    # Step 0 — env once for the whole ext set.
    if not args.skip_env:
        run([PY, str(ROOT / "script/env/fetch_nasa_env_data.py"),
             "--csv-path", str(EXT_CSV), "--output", str(ENV_CSV)])

    # Step 1 — batched loop.
    s = args.start
    while s <= end:
        e = min(s + args.batch - 1, end)
        batch_rows = all_rows[s - 1 : e]
        print(f"\n===== BATCH count {s}..{e} ({len(batch_rows)} items) =====", flush=True)

        slice_csv = WORK / f"ext_{s}_{e}.csv"
        staged_rows = batch_rows

        if args.clip_source == "api":
            write_slice(fieldnames, batch_rows, slice_csv)
            run([PY, str(ROOT / "script/download/download_site_257_clips.py"),
                 "--csv-path", str(EXT_CSV), "--output-dir", str(CLIPS_DIR),
                 "--start-item", str(s), "--end-item", str(e),
                 "--workers", str(args.workers)])
        else:
            staged_rows = stage_batch_from_s3(
                batch_rows=batch_rows,
                clips_dir=CLIPS_DIR,
                s3_clips_prefix=args.s3_clips_prefix,
                aws_profile=args.aws_profile,
                aws_region=args.aws_region,
                fail_on_missing=args.fail_on_missing_s3,
            )
            write_slice(fieldnames, staged_rows, slice_csv)

        if not staged_rows:
            print("  no staged clip folders for this batch; skipping extraction", flush=True)
            s = e + 1
            continue

        batch_manifest = WORK / f"manifest_{s}_{e}.csv"
        run([PY, str(ROOT / "script/dataset/build_training_manifest.py"),
             "--filtered-csv", str(slice_csv), "--env-csv", str(ENV_CSV),
             "--clips-dir", str(CLIPS_DIR), "--output", str(batch_manifest)])

        batch_index = WORK / f"index_{s}_{e}.csv"
        run([PY, str(EXISTING_POOL.parent.parent / "precompute/build_ambient_index.py"),
             "--manifest", str(batch_manifest), "--out-dir", str(SEG_DIR),
             "--index-csv", str(batch_index), "--workers", str(args.workers)])
        if batch_index.exists():
            batch_indexes.append(batch_index)

        # Step — discard this batch's source clips.
        if not args.keep_clips:
            for row in staged_rows:
                folder = CLIPS_DIR / f"site_257_item_{row['id']}"
                if folder.exists():
                    shutil.rmtree(folder, ignore_errors=True)
            print(f"  discarded {len(staged_rows)} clip folders", flush=True)
        s = e + 1

    # Step 2 — merge: existing pool + all new batch indexes -> POOL/ambient_index.csv,
    # and copy existing WAVs into POOL/ambient_segments.
    print("\n===== MERGE =====", flush=True)
    existing_index = EXISTING_POOL / "ambient_index.csv"
    out_index = POOL / "ambient_index.csv"
    header, existing_rows = read_rows(existing_index)
    merged = list(existing_rows)
    for bi in batch_indexes:
        _, rows = read_rows(bi)
        merged.extend(rows)
    with open(out_index, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(merged)

    copied = 0
    for row in existing_rows:
        src = EXISTING_POOL / "ambient_segments" / f"{row['segment_id']}.wav"
        dst = SEG_DIR / f"{row['segment_id']}.wav"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    n_new = len(merged) - len(existing_rows)
    n_wav = len(list(SEG_DIR.glob("*.wav")))
    print(f"existing={len(existing_rows)}  new={n_new}  total_index={len(merged)}  "
          f"copied_existing_wav={copied}  wav_on_disk={n_wav}")
    print(f"POOL ready -> {POOL}")


if __name__ == "__main__":
    main()
