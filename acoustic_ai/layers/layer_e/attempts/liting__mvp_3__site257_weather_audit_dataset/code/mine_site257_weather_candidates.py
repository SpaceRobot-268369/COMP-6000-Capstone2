"""Mine Site257 candidate clips for E-B weather audit.

This first pass is metadata-first. It uses existing Site257 manifests and the
current audited weather seed snapshot to build a listening/audit queue. It does
not download audio, run PANNs, or train a model.

Run from the repository root:

    ./acoustic_ai/.venv/bin/python acoustic_ai/layers/layer_e/attempts/liting__mvp_3__site257_weather_audit_dataset/code/mine_site257_weather_candidates.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[6]
ATTEMPT_DIR = PROJECT_ROOT / "acoustic_ai" / "layers" / "layer_e" / "attempts" / "liting__mvp_3__site257_weather_audit_dataset"
MVP1_ATTEMPT_DIR = PROJECT_ROOT / "acoustic_ai" / "layers" / "layer_e" / "attempts" / "liting__mvp_1__panns_weather_baseline"

DEFAULT_POLICY_SNAPSHOT = MVP1_ATTEMPT_DIR / "data" / "site257_weather_policy_snapshot.csv"
DEFAULT_MVP2_MANIFEST = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "mvp2_per_cell_dataset" / "manifest.csv"
DEFAULT_MVP1_MANIFEST = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "mvp1_all_conditioned_dataset" / "manifest.csv"
DEFAULT_SITE_TRAINING_MANIFEST = PROJECT_ROOT / "resources" / "site_257_bowra-dry-a" / "site_257_training_manifest.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "debug" / "e_b_site257_audit_candidates"

FIELDNAMES = [
    "clip_id",
    "audio_path",
    "source_site_id",
    "source_recording_id",
    "start_s",
    "end_s",
    "duration_s",
    "split",
    "candidate_source",
    "candidate_rank",
    "selection_reason",
    "wind_env_bucket",
    "wind_speed_ms",
    "wind_intensity",
    "rain_intensity",
    "thunder_status",
    "mixed_weather",
    "bird_activity",
    "insect_activity",
    "background_noise",
    "weather_confidence",
    "audit_status",
    "auditor",
    "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine Site257 E-B weather audit candidates.")
    parser.add_argument("--policy-snapshot", type=Path, default=DEFAULT_POLICY_SNAPSHOT)
    parser.add_argument("--mvp-manifest", type=Path, default=DEFAULT_MVP2_MANIFEST)
    parser.add_argument("--fallback-mvp-manifest", type=Path, default=DEFAULT_MVP1_MANIFEST)
    parser.add_argument("--site-training-manifest", type=Path, default=DEFAULT_SITE_TRAINING_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit-per-bucket", type=int, default=120)
    parser.add_argument("--random-holdout", type=int, default=160)
    parser.add_argument("--seed", type=int, default=257)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    site_rows = load_site_manifest(args.mvp_manifest, args.fallback_mvp_manifest)
    training_rows = read_csv(args.site_training_manifest) if args.site_training_manifest.exists() else []
    seed_rows = read_csv(args.policy_snapshot) if args.policy_snapshot.exists() else []

    if not site_rows:
        print("FAIL: no Site257 MVP manifest rows found.")
        return 1

    candidates: list[dict[str, str]] = []
    candidates.extend(seed_weather_candidates(seed_rows))
    candidates.extend(wind_bucket_candidates(site_rows, args.limit_per_bucket, rng))
    candidates.extend(no_weather_candidates(site_rows, args.limit_per_bucket, rng))
    candidates.extend(random_holdout_candidates(site_rows, training_rows, args.random_holdout, rng))

    deduped = dedupe_candidates(candidates)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "candidate_manifest.csv"
    out_json = args.out_dir / "summary.json"
    write_csv(out_csv, deduped)
    summary = build_summary(deduped, site_rows, seed_rows)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Candidate manifest written to: {out_csv}")
    print(f"Summary written to: {out_json}")
    print(
        "Summary: "
        f"candidates={len(deduped)}, "
        f"seed_weather={summary['candidate_source_counts'].get('audited_seed_weather', 0)}, "
        f"wind_bucket={summary['candidate_source_counts'].get('env_wind_bucket_candidate', 0)}, "
        f"no_weather={summary['candidate_source_counts'].get('low_wind_no_weather_candidate', 0)}, "
        f"holdout={summary['candidate_source_counts'].get('random_sitewide_holdout', 0)}"
    )
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_site_manifest(primary: Path, fallback: Path) -> list[dict[str, str]]:
    path = primary if primary.exists() else fallback
    rows = read_csv(path)
    return [row for row in rows if row.get("status", "ok") == "ok"]


def seed_weather_candidates(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for rank, row in enumerate(rows, start=1):
        if row.get("source_type") != "site":
            continue
        if row.get("analysis_use") not in {"site_ready_pool", "site_backup_pool"}:
            continue
        out.append(
            candidate_row(
                clip_id=row.get("asset_id", f"seed_{rank}"),
                audio_path=row.get("clip_path", ""),
                recording_id=row.get("source_recording_id", ""),
                start_s=row.get("start_s", ""),
                end_s=row.get("end_s", ""),
                duration_s=row.get("duration_s", ""),
                split="seed",
                candidate_source="audited_seed_weather",
                candidate_rank=rank,
                selection_reason=row.get("layer_d_role") or row.get("primary_weather") or "audited_seed_weather",
                wind_env_bucket=row.get("env_bucket", ""),
                wind_speed_ms="",
                wind_intensity=normalise_intensity(row.get("wind_intensity", "")),
                rain_intensity=normalise_intensity(row.get("rain_intensity", "")),
                thunder_status="present" if parse_bool(row.get("has_thunder")) else "insufficient_site_data",
                mixed_weather=str(parse_bool(row.get("mixed_weather"))).lower(),
                weather_confidence=row.get("analysis_label_quality", ""),
                audit_status="audited_seed",
                notes=row.get("human_notes", ""),
            )
        )
    return out


def wind_bucket_candidates(rows: list[dict[str, str]], limit_per_bucket: int, rng: random.Random) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        speed = parse_float(row.get("wind_speed_ms"))
        bucket = wind_bucket(speed)
        grouped[bucket].append(row)

    out = []
    for bucket, bucket_rows in sorted(grouped.items()):
        ranked = diverse_sample(bucket_rows, limit_per_bucket, rng)
        for rank, row in enumerate(ranked, start=1):
            out.append(
                candidate_from_site_row(
                    row,
                    split="train_candidate",
                    candidate_source="env_wind_bucket_candidate",
                    candidate_rank=rank,
                    selection_reason=f"metadata wind bucket {bucket}",
                    wind_env_bucket=bucket,
                    wind_intensity="pending",
                    rain_intensity="pending",
                    thunder_status="insufficient_site_data",
                    mixed_weather="pending",
                    weather_confidence="pending",
                )
            )
    return out


def no_weather_candidates(rows: list[dict[str, str]], limit: int, rng: random.Random) -> list[dict[str, str]]:
    low_wind = [
        row
        for row in rows
        if parse_float(row.get("wind_speed_ms")) < 1.5
        and "rain" not in row.get("caption", "").lower()
        and "storm" not in row.get("caption", "").lower()
    ]
    sampled = diverse_sample(low_wind, limit, rng)
    out = []
    for rank, row in enumerate(sampled, start=1):
        out.append(
            candidate_from_site_row(
                row,
                split="validation_candidate",
                candidate_source="low_wind_no_weather_candidate",
                candidate_rank=rank,
                selection_reason="low env wind metadata; no rain/storm caption token",
                wind_env_bucket=wind_bucket(parse_float(row.get("wind_speed_ms"))),
                wind_intensity="pending",
                rain_intensity="pending",
                thunder_status="insufficient_site_data",
                mixed_weather="pending",
                weather_confidence="pending",
            )
        )
    return out


def random_holdout_candidates(
    mvp_rows: list[dict[str, str]],
    training_rows: list[dict[str, str]],
    limit: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    rows = mvp_rows[:]
    if training_rows:
        rows.extend(convert_training_rows(training_rows))
    sampled = diverse_sample(rows, limit, rng)
    out = []
    for rank, row in enumerate(sampled, start=1):
        out.append(
            candidate_from_site_row(
                row,
                split="holdout_candidate",
                candidate_source="random_sitewide_holdout",
                candidate_rank=rank,
                selection_reason="deterministic random Site257 holdout",
                wind_env_bucket=wind_bucket(parse_float(row.get("wind_speed_ms"))),
                wind_intensity="pending",
                rain_intensity="pending",
                thunder_status="insufficient_site_data",
                mixed_weather="pending",
                weather_confidence="pending",
            )
        )
    return out


def convert_training_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    converted = []
    for row in rows:
        clip_path = row.get("clip_path", "")
        recording_id = row.get("recording_id", "")
        clip_index = row.get("clip_index", "")
        converted.append(
            {
                "clip_id": f"training_{recording_id}_clip{clip_index}",
                "audio_path": clip_path,
                "recording_id": recording_id,
                "duration_s": row.get("clip_duration_seconds", ""),
                "wind_speed_ms": row.get("wind_speed_ms", ""),
                "caption": "",
                "season": row.get("season", ""),
                "diel_bin": row.get("sample_bin", ""),
            }
        )
    return converted


def candidate_from_site_row(
    row: dict[str, str],
    *,
    split: str,
    candidate_source: str,
    candidate_rank: int,
    selection_reason: str,
    wind_env_bucket: str,
    wind_intensity: str,
    rain_intensity: str,
    thunder_status: str,
    mixed_weather: str,
    weather_confidence: str,
) -> dict[str, str]:
    return candidate_row(
        clip_id=row.get("clip_id") or row.get("segment_id") or row.get("audio_path", ""),
        audio_path=row.get("audio_path") or row.get("clip_path", ""),
        recording_id=row.get("recording_id", ""),
        start_s=row.get("clip_start_seconds", ""),
        end_s=row.get("clip_end_seconds", ""),
        duration_s=row.get("duration_s") or row.get("clip_duration_seconds", ""),
        split=split,
        candidate_source=candidate_source,
        candidate_rank=candidate_rank,
        selection_reason=selection_reason,
        wind_env_bucket=wind_env_bucket,
        wind_speed_ms=row.get("wind_speed_ms", ""),
        wind_intensity=wind_intensity,
        rain_intensity=rain_intensity,
        thunder_status=thunder_status,
        mixed_weather=mixed_weather,
        weather_confidence=weather_confidence,
        audit_status="pending",
        notes=site_context_note(row),
    )


def candidate_row(
    *,
    clip_id: str,
    audio_path: str,
    recording_id: str,
    start_s: str,
    end_s: str,
    duration_s: str,
    split: str,
    candidate_source: str,
    candidate_rank: int,
    selection_reason: str,
    wind_env_bucket: str,
    wind_speed_ms: str,
    wind_intensity: str,
    rain_intensity: str,
    thunder_status: str,
    mixed_weather: str,
    weather_confidence: str,
    audit_status: str,
    notes: str,
) -> dict[str, str]:
    return {
        "clip_id": clip_id,
        "audio_path": audio_path,
        "source_site_id": "257",
        "source_recording_id": recording_id,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": duration_s,
        "split": split,
        "candidate_source": candidate_source,
        "candidate_rank": str(candidate_rank),
        "selection_reason": selection_reason,
        "wind_env_bucket": wind_env_bucket,
        "wind_speed_ms": wind_speed_ms,
        "wind_intensity": wind_intensity,
        "rain_intensity": rain_intensity,
        "thunder_status": thunder_status,
        "mixed_weather": mixed_weather,
        "bird_activity": "pending",
        "insect_activity": "pending",
        "background_noise": "pending",
        "weather_confidence": weather_confidence,
        "audit_status": audit_status,
        "auditor": "",
        "notes": notes,
    }


def diverse_sample(rows: list[dict[str, str]], limit: int, rng: random.Random) -> list[dict[str, str]]:
    by_recording: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_recording[row.get("recording_id", "")].append(row)

    selected = []
    for recording_id in sorted(by_recording):
        recording_rows = by_recording[recording_id]
        rng.shuffle(recording_rows)
        selected.append(recording_rows[0])
        if len(selected) >= limit:
            return selected

    remaining = [row for row in rows if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, limit - len(selected))])
    return selected[:limit]


def dedupe_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    priority = {
        "audited_seed_weather": 0,
        "low_wind_no_weather_candidate": 1,
        "env_wind_bucket_candidate": 2,
        "random_sitewide_holdout": 3,
    }
    sorted_rows = sorted(rows, key=lambda row: (priority.get(row["candidate_source"], 99), row["audio_path"]))
    seen = set()
    out = []
    for row in sorted_rows:
        key = row["audio_path"] or row["clip_id"]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_summary(rows: list[dict[str, str]], site_rows: list[dict[str, str]], seed_rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "attempt": "liting__mvp_3__site257_weather_audit_dataset",
        "site_manifest_rows": len(site_rows),
        "seed_policy_rows": len(seed_rows),
        "candidate_count": len(rows),
        "candidate_source_counts": dict(Counter(row["candidate_source"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "wind_env_bucket_counts": dict(Counter(row["wind_env_bucket"] for row in rows)),
        "audit_status_counts": dict(Counter(row["audit_status"] for row in rows)),
        "next_step": "Materialise candidate audio on Server B, run PANNs/CLAP/DSP scoring, then manually audit sampled clips.",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def wind_bucket(speed: float) -> str:
    if speed < 1.5:
        return "none"
    if speed < 3.0:
        return "light"
    if speed < 6.0:
        return "moderate"
    return "strong"


def parse_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def normalise_intensity(value: str) -> str:
    value = (value or "").strip().lower()
    return {"medium": "moderate"}.get(value, value or "none")


def site_context_note(row: dict[str, str]) -> str:
    parts = []
    if row.get("season"):
        parts.append(f"season={row['season']}")
    if row.get("diel_bin"):
        parts.append(f"diel={row['diel_bin']}")
    if row.get("caption"):
        parts.append(f"caption={row['caption'][:120]}")
    return "; ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())

