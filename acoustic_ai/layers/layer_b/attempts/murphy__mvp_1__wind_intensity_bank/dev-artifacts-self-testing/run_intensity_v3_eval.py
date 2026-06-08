#!/usr/bin/env python3
"""Generate showcase_intensity_eval_v3 on Server B (one handler load, serial seeds).

Usage (repo root):
  ./acoustic_ai/.venv/bin/python \\
    acoustic_ai/layers/layer_b/attempts/murphy__mvp_1__wind_intensity_bank/dev-artifacts-self-testing/run_intensity_v3_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
ATTEMPT = Path(__file__).resolve().parents[1]
OUT_ROOT = ATTEMPT / "showcase_intensity_eval_v3"
LAYER = "layer_b"
ATTEMPT_ID = "murphy__mvp_1__wind_intensity_bank"
SEEDS = list(range(42, 52))
INTENSITIES = ("light", "medium", "heavy")


def main() -> None:
    sys.path.insert(0, str(REPO / "acoustic_ai"))
    from server import registry  # noqa: E402

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] output_root={OUT_ROOT}")
    print(f"[INFO] seeds={SEEDS} intensities={INTENSITIES}")

    done = 0
    total = len(SEEDS) * len(INTENSITIES)
    for intensity in INTENSITIES:
        print(f"[INFO] === intensity={intensity} ===")
        for seed in SEEDS:
            case = OUT_ROOT / intensity / f"seed_{seed}_generated"
            case.mkdir(parents=True, exist_ok=True)
            wav_path = case / "audio.wav"
            meta_path = case / "metadata.json"

            print(f"[GEN] {intensity} seed={seed} -> {case}")
            result = registry.generate(
                LAYER,
                ATTEMPT_ID,
                seed=seed,
                wind_intensity=intensity,
                weather_type="wind",
            )
            wav_path.write_bytes(result["wav_bytes"])
            meta = dict(result.get("metadata") or {})
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            done += 1
            print(f"[OK] {done}/{total} rms={meta.get('audio', {}).get('rms')}")

    print("ALL_DONE")


if __name__ == "__main__":
    main()
