"""Extract real-audio "expected" samples per attempt from training source data.

Each attempt's `expected/` directory holds 2-3 ground-truth recordings drawn
from the same source data the attempt was trained on. These are NOT model
outputs; they're the canonical real-world recordings the model is meant to
imitate, used as the comparison baseline in the Dev UI's Expected Results
column.

Per attempt this writes a triplet `<stem>.{wav, png, metadata.json}` where
the stem is `real_<source_clip_id>`. WAV is then `dvc add`'d separately; PNG
+ JSON are git-tracked per the artifact policy.

Run from project root with the project venv:

    ./acoustic_ai/.venv/bin/python acoustic_ai/scripts/extract_expected_samples.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "acoustic_ai"))

from layers.layer_a.attempts.lucas__smoke_1__audioldm2_spring_night.code.layer_a_visualization import (  # noqa: E402
    render_layer_a_mel_png_bytes,
    waveform_to_layer_a_mel_db,
)

# ---------------------------------------------------------------------------
# Pick set.
#
# Each attempt entry → list of source-clip descriptors that go into expected/.
# Source kinds:
#   "clip_dir" — pre-extracted clip directory with audio.wav + meta.json (the
#                smoke_1 / smoke_2 / boobook style).
#   "webm_slice" — raw .webm recording, slice [t_start, t_end] (smoke_4 style).
# ---------------------------------------------------------------------------

PICKS: dict[tuple[str, str], list[dict]] = {

    ("layer_a", "lucas__smoke_1__audioldm2_spring_night"): [
        {
            "kind": "clip_dir",
            "stem": "real_001_5392_clip001_s000",
            "clip_dir": "resources/site_257_bowra-dry-a/smoking_test_dataset/clips/001_5392_clip001_s000",
            "manifest": "resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv",
            "reason": "canonical 2019 spring-night exemplar (first manifest row)",
        },
        {
            "kind": "clip_dir",
            "stem": "real_004_1400040_clip001_s000",
            "clip_dir": "resources/site_257_bowra-dry-a/smoking_test_dataset/clips/004_1400040_clip001_s000",
            "manifest": "resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv",
            "reason": "same sonic class, 4 years later (2023) — proves stability",
        },
    ],

    ("layer_a", "lucas__smoke_2__audioldm2_insects"): [
        {
            "kind": "clip_dir",
            "stem": "real_001_1679219_clip021_s000",
            "clip_dir": "resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/clips/001_1679219_clip021_s000",
            "manifest": "resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/manifest.csv",
            "reason": "canonical hot-afternoon cicada (40C, first manifest row)",
        },
        {
            "kind": "clip_dir",
            "stem": "real_037_215729_clip015_s000",
            "clip_dir": "resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/clips/037_215729_clip015_s000",
            "manifest": "resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/manifest.csv",
            "reason": "hottest day in set (44C) — upper end of insect density",
        },
    ],

    ("layer_a", "lucas__smoke_4__vae_baseline"): [
        {
            "kind": "webm_slice",
            "stem": "real_1313184_clip009_s000",
            "webm": "resources/site_257_bowra-dry-a/downloaded_clips/site_257_item_1313184/site_257_item_1313184_clip_009.webm",
            "t_start": 65.155, "t_end": 75.651,
            "diel_bin": "dawn", "season": "autumn",
            "manifest": "acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_index.csv",
            "reason": "dawn / autumn — first row of ambient_index, baseline dawn chorus",
        },
        {
            "kind": "webm_slice",
            "stem": "real_1399626_clip001_s000",
            "webm": "resources/site_257_bowra-dry-a/downloaded_clips/site_257_item_1399626/site_257_item_1399626_clip_001.webm",
            "t_start": 257.904, "t_end": 270.048,
            "diel_bin": "night", "season": "spring",
            "manifest": "acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_index.csv",
            "reason": "night / spring — matches smoke_1's specific case for cross-attempt comparison",
        },
        {
            "kind": "webm_slice",
            "stem": "real_1401228_clip001_s000",
            "webm": "resources/site_257_bowra-dry-a/downloaded_clips/site_257_item_1401228/site_257_item_1401228_clip_001.webm",
            "t_start": 241.905, "t_end": 254.63,
            "diel_bin": "afternoon", "season": "summer",
            "manifest": "acoustic_ai/layers/layer_a/attempts/lucas__smoke_4__vae_baseline/data/ambient/ambient_index.csv",
            "reason": "afternoon / summer — matches smoke_2's daytime cicada case",
        },
    ],

    ("layer_c", "lucas__smoke_1__audiogen_boobook"): [
        {
            "kind": "clip_dir",
            "stem": "real_5296_audioevent_6180314",
            "clip_dir": "resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/site_257_item_5296/site_257_item_5296_audioevent_6180314",
            "manifest": "resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/manifest.csv",
            "reason": "highest BirdNET confidence (0.9999) — clean canonical boobook call (2019-08-14 23:07)",
        },
        {
            "kind": "clip_dir",
            "stem": "real_5390_audioevent_2694054",
            "clip_dir": "resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/site_257_item_5390/site_257_item_5390_audioevent_2694054",
            "manifest": "resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/manifest.csv",
            "reason": "high confidence but dawn (04:57) rather than evening — shows diel variation",
        },
    ],
}


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _audio_stats(audio: np.ndarray, sr: int) -> dict:
    return {
        "sample_rate": sr,
        "duration_s": float(audio.shape[0] / sr),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
    }


def _ffmpeg_webm_to_wav(webm: Path, dst: Path, t_start: float | None = None, t_end: float | None = None) -> None:
    """Decode webm → wav. Optionally slice [t_start, t_end] seconds."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if t_start is not None:
        cmd += ["-ss", f"{t_start:.3f}"]
    if t_end is not None and t_start is not None:
        cmd += ["-to", f"{t_end:.3f}"]
    cmd += ["-i", str(webm), "-ac", "1", str(dst)]
    subprocess.run(cmd, check=True)


def _process_clip_dir(spec: dict, attempt_root: Path) -> dict:
    """Source has audio.wav + meta.json already extracted."""
    src_dir = _PROJECT_ROOT / spec["clip_dir"]
    src_wav = src_dir / "audio.wav"
    src_meta = src_dir / "meta.json"

    dst_wav = attempt_root / "expected" / f"{spec['stem']}.wav"
    shutil.copy2(src_wav, dst_wav)

    audio, sr = _load_wav_mono(dst_wav)
    src_metadata = json.loads(src_meta.read_text()) if src_meta.is_file() else {}

    return _finalize(spec, attempt_root, audio, sr, src_metadata, source_kind="clip_dir")


def _process_webm_slice(spec: dict, attempt_root: Path) -> dict:
    """Slice a window from a raw .webm recording."""
    src_webm = _PROJECT_ROOT / spec["webm"]
    dst_wav = attempt_root / "expected" / f"{spec['stem']}.wav"

    _ffmpeg_webm_to_wav(src_webm, dst_wav, t_start=spec["t_start"], t_end=spec["t_end"])

    audio, sr = _load_wav_mono(dst_wav)
    src_metadata = {
        "source_clip": str(spec["webm"]),
        "t_start": spec["t_start"],
        "t_end": spec["t_end"],
        "diel_bin": spec.get("diel_bin"),
        "season": spec.get("season"),
    }
    return _finalize(spec, attempt_root, audio, sr, src_metadata, source_kind="webm_slice")


def _finalize(spec: dict, attempt_root: Path,
              audio: np.ndarray, sr: int, src_metadata: dict,
              *, source_kind: str) -> dict:
    """Render mel PNG + write metadata.json. Returns artefact paths."""

    duration_s = float(audio.shape[0] / sr)
    mel_db = waveform_to_layer_a_mel_db(audio, sr)
    png_bytes = render_layer_a_mel_png_bytes(mel_db, duration_s)

    png_path = attempt_root / "expected" / f"{spec['stem']}.png"
    json_path = attempt_root / "expected" / f"{spec['stem']}.metadata.json"
    png_path.write_bytes(png_bytes)

    metadata = {
        "tier": "expected",
        "source": "real_audio",
        "source_kind": source_kind,
        "source_clip_id": spec["stem"].removeprefix("real_"),
        "source_manifest": spec.get("manifest"),
        "selection_reason": spec.get("reason"),
        "audio": _audio_stats(audio, sr),
        "source_metadata": src_metadata,
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return {
        "wav": str(attempt_root / "expected" / f"{spec['stem']}.wav"),
        "png": str(png_path),
        "json": str(json_path),
    }


def main() -> int:
    for (layer, attempt), specs in PICKS.items():
        attempt_root = _PROJECT_ROOT / "acoustic_ai" / "layers" / layer / "attempts" / attempt
        expected_dir = attempt_root / "expected"
        expected_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {layer}/{attempt} → {len(specs)} expected sample(s) ===")
        for spec in specs:
            try:
                if spec["kind"] == "clip_dir":
                    out = _process_clip_dir(spec, attempt_root)
                elif spec["kind"] == "webm_slice":
                    out = _process_webm_slice(spec, attempt_root)
                else:
                    raise ValueError(f"unknown kind: {spec['kind']!r}")
                print(f"  [OK]  {spec['stem']}")
                print(f"        wav  → {Path(out['wav']).relative_to(_PROJECT_ROOT)}")
                print(f"        png  → {Path(out['png']).relative_to(_PROJECT_ROOT)}")
                print(f"        json → {Path(out['json']).relative_to(_PROJECT_ROOT)}")
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] {spec['stem']}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
