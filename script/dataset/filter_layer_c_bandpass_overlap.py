#!/usr/bin/env python3
"""Filter Layer C bandpass review items for annotation overlap.

This script keeps the existing bandpass/tightcrop audio intact. It only builds
CSV/M3U audit lists that remove likely contaminated snippets:

- non-target bird events overlapping the bandpass crop
- duplicate target-species events overlapping the crop

The output is a prefilter for manual audit, not a final ground truth label.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy import ndimage


TARGET_ALIASES = {
    "chestnut_rumped_thornbill": ("Chestnut-rumped Thornbill", "Acanthiza uropygialis"),
    "southern_boobook": ("Southern Boobook", "Ninox boobook"),
    "crested_bellbird": ("Crested Bellbird", "Oreoica gutturalis"),
    "white_browed_woodswallow": ("White-browed Woodswallows", "Artamus superciliosus"),
    "red_capped_robin": ("Red-capped Robin", "Petroica goodenovii"),
    "superb_fairywren": ("Superb Fairywren", "Malurus cyaneus"),
}

CLUTTER_LIMITS = {
    "chestnut_rumped_thornbill": {"coverage": 0.024, "components_per_s": 12.0, "col_gt25": 0.008},
    "southern_boobook": {"coverage": 0.055, "components_per_s": 8.0, "col_gt25": 0.050},
    "crested_bellbird": {"coverage": 0.050, "components_per_s": 10.0, "col_gt25": 0.040},
    "white_browed_woodswallow": {"coverage": 0.045, "components_per_s": 14.0, "col_gt25": 0.035},
    "red_capped_robin": {"coverage": 0.040, "components_per_s": 14.0, "col_gt25": 0.030},
    "superb_fairywren": {"coverage": 0.050, "components_per_s": 16.0, "col_gt25": 0.040},
}


def has_target_tag(row: pd.Series, event_type: str) -> bool:
    common, sci = TARGET_ALIASES[event_type]
    haystack = " | ".join(
        str(row.get(col, ""))
        for col in ("common_name_tags", "species_name_tags", "other_tags")
    ).lower()
    common_tokens = [common.lower(), common.lower().replace("woodswallows", "woodswallow")]
    return any(token in haystack for token in common_tokens) or sci.lower() in haystack


def overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def row_verdict(
    row: pd.Series,
    min_score: float,
    min_overlap_s: float,
    min_overlap_frac: float,
    non_target_hard_score: float,
    target_overlap_mode: str,
) -> tuple[str, str]:
    event_type = str(row["event_type"])
    annotation_path = Path(str(row["annotation_csv"]))
    if not annotation_path.exists():
        return "Fail", f"missing_annotation_csv={annotation_path}"

    extracted_start = float(row["extracted_start_seconds"])
    crop_start = extracted_start + float(row["tightcrop_start_seconds_in_extracted"])
    crop_end = extracted_start + float(row["tightcrop_end_seconds_in_extracted"])
    crop_duration = max(1e-6, crop_end - crop_start)
    own_event_id = int(row["audio_event_id"])

    ann = pd.read_csv(annotation_path)
    fail_reasons: list[str] = []
    overlap_details: list[str] = []

    for _, other in ann.iterrows():
        try:
            other_id = int(other["audio_event_id"])
            score = float(other.get("score", 0.0))
            other_start = float(other["event_start_seconds"])
            other_end = float(other["event_end_seconds"])
        except (TypeError, ValueError):
            continue
        if other_id == own_event_id or score < min_score:
            continue

        ov = overlap_seconds(crop_start, crop_end, other_start, other_end)
        if ov < min_overlap_s or ov / crop_duration < min_overlap_frac:
            continue

        is_target = has_target_tag(other, event_type)
        label = str(other.get("common_name_tags", "")) or str(other.get("other_tags", "")) or "unknown"
        detail = f"{other_id}:{score:.3f}:{ov:.2f}s:{label}"
        overlap_details.append(detail)
        if is_target:
            if target_overlap_mode == "hard":
                fail_reasons.append(f"target_call_overlap={detail}")
            else:
                overlap_details.append(f"target_call_overlap_warning={detail}")
        elif score >= non_target_hard_score:
            fail_reasons.append(f"non_target_overlap={detail}")
        else:
            overlap_details.append(f"low_score_non_target_overlap_warning={detail}")

    if fail_reasons:
        return "Fail", "; ".join(fail_reasons)
    if overlap_details:
        return "Pass", "; ".join(overlap_details)
    return "Pass", "no hard overlapping annotation in bandpass crop"


def spectral_clutter_verdict(row: pd.Series) -> tuple[str, str]:
    """Reject obviously cluttered bandpass crops.

    This catches cases where BirdNET annotations did not label the contaminant,
    but the mel still shows dense high-energy marks or overlapping calls inside
    the target band.
    """
    event_type = str(row["event_type"])
    limits = CLUTTER_LIMITS[event_type]
    wav_path = Path(str(row["bandpass_tightcrop_wav"]))
    if not wav_path.exists():
        return "Fail", f"missing_bandpass_tightcrop_wav={wav_path}"
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = max(1e-6, len(audio) / sr)
    low = float(row["bandpass_low_hz"])
    high = float(row["bandpass_high_hz"])

    spec = np.abs(librosa.stft(audio, n_fft=1024, hop_length=128)) ** 2
    db = librosa.power_to_db(spec, ref=np.max, top_db=80)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band = db[(freqs >= low) & (freqs <= high), :]
    if band.size == 0:
        return "Fail", "empty_bandpass_spectrogram"

    # Use a strict high-energy mask; this avoids treating the whole filtered
    # background as a call. Dense masks usually mean mixed/overlapping calls.
    binary = band > -25
    _, component_count = ndimage.label(binary, structure=np.ones((3, 3), dtype=bool))
    coverage = float(binary.mean())
    col_coverage = binary.mean(axis=0)
    col_gt25 = float((col_coverage > 0.25).mean())
    components_per_s = float(component_count / duration)

    reasons = []
    if coverage > limits["coverage"]:
        reasons.append(f"spectral_clutter_coverage={coverage:.4f}>{limits['coverage']:.4f}")
    if components_per_s > limits["components_per_s"]:
        reasons.append(f"too_many_bright_call_marks_per_s={components_per_s:.2f}>{limits['components_per_s']:.2f}")
    if col_gt25 > limits["col_gt25"]:
        reasons.append(f"dense_vertical_overlap_fraction={col_gt25:.4f}>{limits['col_gt25']:.4f}")
    if reasons:
        return "Fail", "; ".join(reasons)
    return "Pass", (
        f"spectral_clutter_ok coverage={coverage:.4f}, "
        f"components_per_s={components_per_s:.2f}, col_gt25={col_gt25:.4f}"
    )


def write_playlists(df: pd.DataFrame, out_dir: Path, prefix: str, suffix: str) -> None:
    for verdict in ("Pass", "Fail"):
        subset = df[df["bandpass_overlap_prefilter_verdict"] == verdict]
        with (out_dir / f"{prefix}_{verdict.lower()}_bandpass_tightcrop_{suffix}_absolute.m3u").open("w", encoding="utf-8") as f:
            for path in subset["bandpass_tightcrop_wav"]:
                f.write(str(Path(path).resolve()) + "\n")
        with (out_dir / f"{prefix}_{verdict.lower()}_bandpass_full_{suffix}_absolute.m3u").open("w", encoding="utf-8") as f:
            for path in subset["bandpass_full_wav"]:
                f.write(str(Path(path).resolve()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-score", type=float, default=0.50)
    parser.add_argument("--min-overlap-s", type=float, default=0.25)
    parser.add_argument("--min-overlap-frac", type=float, default=0.10)
    parser.add_argument("--non-target-hard-score", type=float, default=0.90)
    parser.add_argument("--target-overlap-mode", choices=("hard", "warn"), default="hard")
    parser.add_argument("--spectral-mode", choices=("hard", "warn"), default="hard")
    parser.add_argument("--suffix", default="v2")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    annotation_verdicts = [
        row_verdict(
            row,
            args.min_score,
            args.min_overlap_s,
            args.min_overlap_frac,
            args.non_target_hard_score,
            args.target_overlap_mode,
        )
        for _, row in df.iterrows()
    ]
    spectral_verdicts = [spectral_clutter_verdict(row) for _, row in df.iterrows()]
    combined_verdicts = []
    combined_reasons = []
    for (av, ar), (sv, sr) in zip(annotation_verdicts, spectral_verdicts):
        verdict = "Fail" if av == "Fail" or (args.spectral_mode == "hard" and sv == "Fail") else "Pass"
        reason_parts = []
        if av == "Fail":
            reason_parts.append(f"annotation:{ar}")
        if sv == "Fail":
            prefix = "spectrogram" if args.spectral_mode == "hard" else "spectrogram_warning"
            reason_parts.append(f"{prefix}:{sr}")
        if not reason_parts:
            reason_parts.append(f"{ar}; {sr}")
        combined_verdicts.append(verdict)
        combined_reasons.append("; ".join(reason_parts))

    df["bandpass_overlap_prefilter_verdict"] = combined_verdicts
    df["bandpass_overlap_prefilter_reason"] = combined_reasons
    df["manual_verdict"] = ""
    df["manual_notes"] = ""

    suffix = args.suffix
    all_csv = out_dir / f"manual_audit_all_6species_bandpass_overlap_prefilter_{suffix}.csv"
    pass_csv = out_dir / f"manual_audit_all_6species_bandpass_overlap_prefilter_pass_{suffix}.csv"
    fail_csv = out_dir / f"manual_audit_all_6species_bandpass_overlap_prefilter_fail_{suffix}.csv"
    df.to_csv(all_csv, index=False)
    df[df["bandpass_overlap_prefilter_verdict"] == "Pass"].to_csv(pass_csv, index=False)
    df[df["bandpass_overlap_prefilter_verdict"] == "Fail"].to_csv(fail_csv, index=False)
    write_playlists(df, out_dir, "manual_audit_all_6species", suffix)

    for event_type, group in df.groupby("event_type", sort=False):
        species_dir = out_dir / event_type
        species_dir.mkdir(parents=True, exist_ok=True)
        group.to_csv(species_dir / f"manual_audit_{event_type}_bandpass_overlap_prefilter_all_{suffix}.csv", index=False)
        group[group["bandpass_overlap_prefilter_verdict"] == "Pass"].to_csv(
            species_dir / f"manual_audit_{event_type}_bandpass_overlap_prefilter_pass_{suffix}.csv",
            index=False,
        )
        group[group["bandpass_overlap_prefilter_verdict"] == "Fail"].to_csv(
            species_dir / f"manual_audit_{event_type}_bandpass_overlap_prefilter_fail_{suffix}.csv",
            index=False,
        )
        write_playlists(group, species_dir, f"manual_audit_{event_type}", suffix)

    print(f"wrote {out_dir}")
    print(df.groupby(["event_type", "bandpass_overlap_prefilter_verdict"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
