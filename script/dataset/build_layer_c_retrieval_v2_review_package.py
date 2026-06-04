#!/usr/bin/env python3
"""Build a Layer C retrieval v2 review package from selected metadata rows.

The script is intentionally conservative:

- It uses ffmpeg/ffprobe for audio conversion and filtering.
- It renders audit mel images with the same matplotlib-style layout used by
  the prior Layer C human review packages.
- Prior human-pass rows with local prior audio can be packaged immediately.
- Rows without a local source are represented by metadata/review files and
  marked `needs_audio_download`; a later download pass can fill the audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_ROOT = (
    REPO_ROOT
    / "acoustic_ai"
    / "layers"
    / "layer_c"
    / "attempts"
    / "burger__mvp_2__retrieval_v2_library"
    / "data"
    / "media_asset_bank"
)
DEFAULT_SELECTED = LIB_ROOT / "selected_samples_v2.csv"
DEFAULT_BANDS = LIB_ROOT / "species_band_config_v2.csv"
DEFAULT_OUTPUT = LIB_ROOT / "review_package_pilot_v2"
DEFAULT_LOCAL_REUSE_ROOTS = [
    REPO_ROOT / "resources" / "site_257_bowra-dry-a" / "layer_c_v2_candidate_pool",
    REPO_ROOT / "resources" / "site_257_bowra-dry-a" / "layer_c_smoke_fairywren_robin_bellbird",
]
DEFAULT_DOWNLOADED_REUSE_ROOTS = [
    LIB_ROOT / "review_package_full_v2",
    LIB_ROOT / "review_package_pilot_v2",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_repo_path(value: str) -> Path | None:
    value = str(value or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.exists() else None


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def convert_wav(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "22050", str(dst)])


def bandpass_wav(src: Path, dst: Path, low_hz: int, high_hz: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            f"highpass=f={low_hz},lowpass=f={high_hz},loudnorm=I=-24:TP=-2:LRA=11",
            "-ac",
            "1",
            "-ar",
            "22050",
            str(dst),
        ]
    )


def hz_to_mel(hz: Any) -> Any:
    import numpy as np

    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: Any) -> Any:
    import numpy as np

    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> tuple[Any, Any]:
    import numpy as np

    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.searchsorted(fft_freqs, hz_points)
    filters = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)

    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center > left:
            filters[i, left:center] = (
                fft_freqs[left:center] - fft_freqs[left]
            ) / max(fft_freqs[center] - fft_freqs[left], 1e-12)
        if right > center:
            filters[i, center:right] = (
                fft_freqs[right] - fft_freqs[center:right]
            ) / max(fft_freqs[right] - fft_freqs[center], 1e-12)
    return filters, hz_points[1:-1]


def read_wav_mono(path: Path) -> tuple[Any, int]:
    import numpy as np
    from scipy.io import wavfile

    sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        max_value = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)
    return audio, int(sr)


def render_spectrogram(src: Path, dst: Path, title: str, high_hz: int | None = None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/matplotlib-cache")))

    import matplotlib
    import numpy as np
    from scipy import signal

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audio, sr = read_wav_mono(src)
    n_fft = 2048
    hop_length = 256
    fmax = min(sr / 2.0, max(float(high_hz or sr / 2.0), 11000.0))
    f, _t, spec = signal.spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        mode="magnitude",
        scaling="spectrum",
    )
    keep = f <= fmax
    filters, mel_hz = mel_filterbank(sr, n_fft, 128, 0.0, fmax)
    mel_power = filters[:, keep] @ (spec[keep, :] ** 2)
    t = np.arange(mel_power.shape[1], dtype=np.float32) * hop_length / float(sr)
    ref = max(float(np.max(mel_power)), 1e-12)
    db = 10.0 * np.log10(np.maximum(mel_power, 1e-12) / ref)
    db = np.clip(db, -80.0, 0.0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.pcolormesh(
        t,
        mel_hz,
        db,
        shading="auto",
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Hz")
    ax.set_ylim(0, fmax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(dst, dpi=120)
    plt.close(fig)


def load_bands(path: Path) -> dict[str, dict[str, str]]:
    return {row["species_slug"]: row for row in read_csv(path)}


def load_excluded_event_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    rows = read_csv(path)
    if not rows:
        return set()
    key = "audio_event_id" if "audio_event_id" in rows[0] else next(iter(rows[0]))
    return {str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()}


def build_local_reuse_index(roots: list[Path], target_event_ids: set[str] | None = None) -> dict[str, Path]:
    """Index prior local crops/segments by audio event id.

    The prepared segment manifests are preferred over quality crop manifests
    because they usually include the wider buffered event source.
    """
    candidates: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for manifest in root.rglob("*.csv"):
            try:
                rows = read_csv(manifest)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                event_id = str(row.get("audio_event_id", "")).strip()
                if not event_id:
                    continue
                if target_event_ids is not None and event_id not in target_event_ids:
                    continue
                if row.get("prepare_status") and row["prepare_status"] != "ok":
                    continue
                for priority, key in ((0, "source_webm_path"), (1, "audio_path")):
                    source = resolve_repo_path(row.get(key, ""))
                    if source and source.stat().st_size > 0:
                        candidates[event_id].append((priority, source))
                        break

    out: dict[str, Path] = {}
    for event_id, paths in candidates.items():
        paths.sort(key=lambda item: (item[0], len(str(item[1]))))
        out[event_id] = paths[0][1]
    return out


def source_for_row(
    row: dict[str, str],
    output_dir: Path,
    item_dir: Path,
    *,
    require_s3_origin_for_downloaded: bool,
    local_reuse_index: dict[str, Path],
) -> tuple[Path | None, str]:
    event_id = row.get("audio_event_id", "")
    species_slug = row.get("species_slug", "")
    if event_id:
        downloaded_candidates = [item_dir / "downloaded_source.webm"]
        downloaded_roots = [output_dir, *DEFAULT_DOWNLOADED_REUSE_ROOTS]
        seen_roots: set[Path] = set()
        for root in downloaded_roots:
            root = root.resolve()
            if root in seen_roots:
                continue
            seen_roots.add(root)
            if species_slug:
                downloaded_candidates.extend(
                    sorted(
                        (root / species_slug / "samples").glob(
                            f"*_audioevent_{event_id}/downloaded_source.webm"
                        )
                    )
                )
        for downloaded in downloaded_candidates:
            if not downloaded.exists() or downloaded.stat().st_size <= 0:
                continue
            origin_path = downloaded.with_name("downloaded_source_origin.json")
            if origin_path.exists():
                try:
                    origin = json.loads(origin_path.read_text(encoding="utf-8"))
                    source_kind = str(origin.get("source_kind") or "v2_downloaded_source")
                    if (not require_s3_origin_for_downloaded) or source_kind == "s3_downloaded_clip":
                        return downloaded, source_kind
                except (OSError, ValueError):
                    pass
            if require_s3_origin_for_downloaded:
                continue
            return downloaded, "v2_downloaded_source"

    for key in (
        "prior_source_audio_path",
        "prior_retrieval_audio_path",
    ):
        path = resolve_repo_path(row.get(key, ""))
        if path:
            return path, key
    if event_id and event_id in local_reuse_index:
        return local_reuse_index[event_id], "local_reuse_manifest"
    # Full original recordings are not normally materialised locally in this
    # repo. Keep this hook for future downloaded_originals backfill.
    canonical = row.get("canonical_file_name", "")
    if canonical:
        path = resolve_repo_path(f"resources/site_257_bowra-dry-a/downloaded_originals/{canonical}")
        if path:
            return path, "downloaded_originals"
    return None, ""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def existing_packaged_outputs_ok(item_dir: Path, event_id: str) -> dict[str, Any] | None:
    metadata_path = item_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if str(existing.get("audio_event_id", "")) != str(event_id):
        return None
    required = [
        "original.wav",
        "crop_full.wav",
        "crop_bandpass.wav",
        "mel_full.png",
        "mel_bandpass.png",
    ]
    if all((item_dir / name).exists() and (item_dir / name).stat().st_size > 0 for name in required):
        return existing
    return None


def select_rows(
    rows: list[dict[str, str]],
    per_species_limit: int | None,
    excluded_event_ids: set[str],
) -> list[dict[str, str]]:
    if per_species_limit is None:
        return [row for row in rows if row.get("audio_event_id", "") not in excluded_event_ids]
    out: list[dict[str, str]] = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("audio_event_id", "") in excluded_event_ids:
            continue
        slug = row["species_slug"]
        if counts[slug] >= per_species_limit:
            continue
        out.append(row)
        counts[slug] += 1
    return out


def backfill_manifest_row(row: dict[str, str], item_dir: Path) -> dict[str, Any]:
    event_start = float(row["event_start_s"])
    event_end = float(row["event_end_s"])
    pre_buffer = 0.25
    post_buffer = 0.35
    download_start = max(0.0, event_start - pre_buffer)
    download_end = event_end + post_buffer
    return {
        "species_common_name": row["species_common_name"],
        "species_slug": row["species_slug"],
        "audio_event_id": row["audio_event_id"],
        "recording_id": row["recording_id"],
        "event_start_s": f"{event_start:.4f}",
        "event_end_s": f"{event_end:.4f}",
        "pre_buffer_s": f"{pre_buffer:.3f}",
        "post_buffer_s": f"{post_buffer:.3f}",
        "download_start_s": f"{download_start:.3f}",
        "download_end_s": f"{download_end:.3f}",
        "download_duration_s": f"{download_end - download_start:.3f}",
        "item_dir": str(item_dir.relative_to(REPO_ROOT)),
        "output_path": str((item_dir / "downloaded_source.webm").relative_to(REPO_ROOT)),
        "listen_url": row.get("listen_url", ""),
        "library_url": row.get("library_url", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--band-config", type=Path, default=DEFAULT_BANDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-species-limit", type=int, default=3)
    parser.add_argument("--exclude-event-ids", type=Path, default=None)
    parser.add_argument(
        "--require-s3-origin-for-downloaded",
        action="store_true",
        help="Only accept downloaded_source.webm when it has an S3 origin sidecar.",
    )
    parser.add_argument(
        "--skip-local-reuse-index",
        action="store_true",
        help="Skip scanning prior local review/candidate manifests for reusable audio.",
    )
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = REPO_ROOT / args.output_dir

    bands = load_bands(args.band_config)
    excluded_event_ids = load_excluded_event_ids(args.exclude_event_ids)
    per_species_limit = None if args.per_species_limit <= 0 else args.per_species_limit
    selected = select_rows(read_csv(args.selected_csv), per_species_limit, excluded_event_ids)
    target_event_ids = {str(row.get("audio_event_id", "")).strip() for row in selected}
    target_event_ids.discard("")
    local_reuse_index = (
        {}
        if args.skip_local_reuse_index
        else build_local_reuse_index(DEFAULT_LOCAL_REUSE_ROOTS, target_event_ids)
    )
    packaged_rows: list[dict[str, Any]] = []
    backfill_rows: list[dict[str, Any]] = []

    per_species_index: dict[str, int] = defaultdict(int)
    packaged_audio = 0
    needs_download = 0
    errors = 0

    for row in selected:
        slug = row["species_slug"]
        per_species_index[slug] += 1
        idx = per_species_index[slug]
        event_id = row["audio_event_id"]
        band = bands[slug]
        low = int(float(band["default_low_hz"]))
        high = int(float(band["default_high_hz"]))

        item_dir = args.output_dir / slug / "samples" / f"{idx:03d}_audioevent_{event_id}"
        source, source_kind = source_for_row(
            row,
            args.output_dir,
            item_dir,
            require_s3_origin_for_downloaded=args.require_s3_origin_for_downloaded,
            local_reuse_index=local_reuse_index,
        )
        metadata = {
            **row,
            "sample_index": idx,
            "source_audio_resolved": str(source.relative_to(REPO_ROOT)) if source else "",
            "source_resolution_kind": source_kind,
            "source_is_s3_backed": False,
            "pre_buffer_s": 0.25,
            "post_buffer_s": 0.35,
            "species_low_hz": low,
            "species_high_hz": high,
            "sample_low_hz": low,
            "sample_high_hz": high,
            "bandpass_reason": f"{band['band_source']}; {band['notes']}",
            "package_status": "needs_audio_download",
        }
        review = {
            "verdict": "Needs audio download",
            "reviewer": "",
            "notes": "No local source audio resolved for this metadata row.",
            "complete_call": "",
            "time_crop_ok": "",
            "bandpass_ok": "",
            "overlap_ok": "",
        }

        try:
            existing = existing_packaged_outputs_ok(item_dir, event_id)
            if existing:
                metadata.update(
                    {
                        "crop_full_duration_s": existing.get("crop_full_duration_s", ""),
                        "original_wav": "original.wav",
                        "crop_full_wav": "crop_full.wav",
                        "crop_bandpass_wav": "crop_bandpass.wav",
                        "mel_full_png": "mel_full.png",
                        "mel_bandpass_png": "mel_bandpass.png",
                        "source_audio_resolved": existing.get("source_audio_resolved", ""),
                        "source_resolution_kind": existing.get("source_resolution_kind", "existing_package"),
                        "source_is_s3_backed": existing.get("source_is_s3_backed", False),
                        "package_status": "packaged",
                    }
                )
                review.update(
                    {
                        "verdict": "Pending review",
                        "notes": "Existing packaged audio/images reused for this row.",
                    }
                )
                source_kind = str(metadata["source_resolution_kind"])
                packaged_audio += 1
            elif source:
                original = item_dir / "original.wav"
                crop_full = item_dir / "crop_full.wav"
                crop_bandpass = item_dir / "crop_bandpass.wav"
                mel_full = item_dir / "mel_full.png"
                mel_bandpass = item_dir / "mel_bandpass.png"
                convert_wav(source, original)
                shutil.copy2(original, crop_full)
                bandpass_wav(crop_full, crop_bandpass, low, high)
                score = float(row.get("score") or 0.0)
                title = (
                    f"{row['species_common_name']} audit {idx:03d} "
                    f"event {event_id} score {score:.3f}"
                )
                render_spectrogram(crop_full, mel_full, f"{title} original full")
                render_spectrogram(crop_bandpass, mel_bandpass, f"{title} full bandpass {low}-{high} Hz", high)
                duration = ffprobe_duration(crop_full)
                metadata.update(
                    {
                        "crop_full_duration_s": round(duration, 4),
                        "original_wav": "original.wav",
                        "crop_full_wav": "crop_full.wav",
                        "crop_bandpass_wav": "crop_bandpass.wav",
                        "mel_full_png": "mel_full.png",
                        "mel_bandpass_png": "mel_bandpass.png",
                        "package_status": "packaged",
                    }
                )
                review.update(
                    {
                        "verdict": "Pending review",
                        "notes": "Audio packaged from local prior/source file; review call completeness and bandpass.",
                    }
                )
                packaged_audio += 1
            else:
                needs_download += 1
                backfill_rows.append(backfill_manifest_row(row, item_dir))
        except subprocess.CalledProcessError as exc:
            errors += 1
            metadata["package_status"] = "error"
            metadata["package_error"] = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc)
            review["verdict"] = "Packaging error"
            review["notes"] = metadata["package_error"]

        write_json(item_dir / "metadata.json", metadata)
        write_json(item_dir / "review.json", review)
        packaged_rows.append(
            {
                "species_common_name": row["species_common_name"],
                "species_slug": slug,
                "audio_event_id": event_id,
                "item_dir": str(item_dir.relative_to(REPO_ROOT)),
                "package_status": metadata["package_status"],
                "source_resolution_kind": source_kind,
                "source_audio_resolved": metadata["source_audio_resolved"],
            }
        )

    summary_path = args.output_dir / "review_package_summary_v2.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "species_common_name",
            "species_slug",
            "audio_event_id",
            "item_dir",
            "package_status",
            "source_resolution_kind",
            "source_audio_resolved",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(packaged_rows)

    manifest_path = args.output_dir / "pilot_backfill_event_manifest_v2.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "species_common_name",
            "species_slug",
            "audio_event_id",
            "recording_id",
            "event_start_s",
            "event_end_s",
            "pre_buffer_s",
            "post_buffer_s",
            "download_start_s",
            "download_end_s",
            "download_duration_s",
            "item_dir",
            "output_path",
            "listen_url",
            "library_url",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(backfill_rows)

    print(f"selected_rows={len(selected)}")
    print(f"packaged_audio={packaged_audio}")
    print(f"needs_audio_download={needs_download}")
    print(f"errors={errors}")
    print(f"wrote {manifest_path}")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
