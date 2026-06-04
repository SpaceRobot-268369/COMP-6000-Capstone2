"""Build a wind-only training manifest for Layer B generation smoke.

Input:
  - Layer B shared weather index CSV

Output:
  - wind_manifest.csv with AudioLDM2Dataset-compatible fields:
      audio_path, caption, status
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


_ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INDEX = (
    _ATTEMPT_ROOT.parent
    / "lucas__smoke_1__curated_assets"
    / "data"
    / "weather"
    / "asset_index.csv"
)
_DEFAULT_OUT = _ATTEMPT_ROOT / "data" / "wind_manifest.csv"
_DEFAULT_CAPTION = (
    "steady wind through dry eucalyptus woodland, gentle natural breeze, "
    "Bowra, Australia"
)


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Layer B wind training manifest.")
    parser.add_argument(
        "--asset-index",
        type=Path,
        default=_DEFAULT_INDEX,
        help="Path to Layer B asset_index.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output manifest path",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=_DEFAULT_CAPTION,
        help="Locked caption written to every training row",
    )
    parser.add_argument(
        "--max-contamination-score",
        type=float,
        default=0.28,
        help="Reject clips with contamination_score above this threshold",
    )
    parser.add_argument(
        "--only-intensity",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Keep only this wind_intensity tier",
    )
    parser.add_argument(
        "--exclude-quality-flag-token",
        type=str,
        default="nov2019_storm_scout001",
        help="Reject rows whose quality_flags contains this token",
    )
    parser.add_argument(
        "--max-per-recording",
        type=int,
        default=3,
        help="Maximum kept rows per source_recording_id (0 disables cap)",
    )
    return parser.parse_args()


def _resolve_audio_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def main() -> None:
    args = _parse_args()
    repo_root = _ATTEMPT_ROOT.parents[4]

    rows_out_by_recording: dict[str, list[dict[str, str]]] = defaultdict(list)
    reasons = Counter()

    with args.asset_index.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("source_type") != "site":
                reasons["non_site"] += 1
                continue
            if row.get("layer_d_use") == "reject":
                reasons["rejected"] += 1
                continue
            if not _truthy(row.get("has_wind", "")):
                reasons["not_wind"] += 1
                continue
            if _truthy(row.get("has_rain", "")):
                reasons["contains_rain"] += 1
                continue
            if _truthy(row.get("has_thunder", "")):
                reasons["contains_thunder"] += 1
                continue
            if row.get("wind_intensity", "").strip().lower() != args.only_intensity:
                reasons["intensity_mismatch"] += 1
                continue
            quality_flags = row.get("quality_flags", "")
            if args.exclude_quality_flag_token and args.exclude_quality_flag_token in quality_flags:
                reasons["excluded_quality_flag"] += 1
                continue

            contamination_raw = row.get("contamination_score", "")
            try:
                contamination_score = float(contamination_raw)
            except ValueError:
                reasons["invalid_contamination_score"] += 1
                continue

            if contamination_score > args.max_contamination_score:
                reasons["high_contamination_score"] += 1
                continue

            clip_path = row.get("clip_path", "")
            audio_abs = _resolve_audio_path(clip_path, repo_root)
            if not audio_abs.is_file():
                reasons["missing_audio_file"] += 1
                continue

            source_recording_id = row.get("source_recording_id", "")
            rows_out_by_recording[source_recording_id].append(
                {
                    "audio_path": clip_path,
                    "caption": args.caption,
                    "status": "ok",
                    "asset_id": row.get("asset_id", ""),
                    "source_recording_id": source_recording_id,
                    "wind_intensity": row.get("wind_intensity", ""),
                    "contamination_score": f"{contamination_score:.6f}",
                    "quality_flags": quality_flags,
                }
            )

    rows_out: list[dict[str, str]] = []
    for _, recording_rows in rows_out_by_recording.items():
        recording_rows.sort(key=lambda r: float(r["contamination_score"]))
        if args.max_per_recording > 0:
            kept = recording_rows[: args.max_per_recording]
            dropped = max(0, len(recording_rows) - len(kept))
            if dropped:
                reasons["per_recording_capped"] += dropped
            rows_out.extend(kept)
        else:
            rows_out.extend(recording_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audio_path",
        "caption",
        "status",
        "asset_id",
        "source_recording_id",
        "wind_intensity",
        "contamination_score",
        "quality_flags",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    intensity_counts = Counter(r["wind_intensity"] for r in rows_out)
    recording_counts = Counter(r["source_recording_id"] for r in rows_out)
    print(f"[manifest] wrote {len(rows_out)} rows -> {args.out}")
    print(f"[manifest] contamination <= {args.max_contamination_score:.3f}")
    print(f"[manifest] only_intensity = {args.only_intensity}")
    print(f"[manifest] max_per_recording = {args.max_per_recording}")
    print(f"[manifest] intensity counts: {dict(intensity_counts)}")
    print(f"[manifest] distinct recordings: {len(recording_counts)}")
    print(f"[manifest] filtered reasons: {dict(reasons)}")
    if len(rows_out) < 25:
        print("[manifest][warn] fewer than 25 clips kept; consider relaxed threshold or data augmentation.")


if __name__ == "__main__":
    main()
