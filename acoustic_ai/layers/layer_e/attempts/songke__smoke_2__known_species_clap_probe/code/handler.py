"""Registry-facing handler for E-C known-species event detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from clap_backbone import CLAPBackbone  # noqa: E402
from detect import detect_windows  # noqa: E402


class KnownSpeciesEventDetector:
    def __init__(
        self,
        *,
        checkpoint: Path | None = None,
        threshold: float = 0.55,
        window_s: float = 5.0,
        hop_s: float = 1.0,
        merge_gap_s: float = 1.0,
        min_event_windows: int = 7,
    ) -> None:
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.window_s = window_s
        self.hop_s = hop_s
        self.merge_gap_s = merge_gap_s
        self.min_event_windows = min_event_windows
        self.backbone = CLAPBackbone()

    def analyze(self, audio_path: str | Path) -> dict:
        result = detect_windows(
            Path(audio_path),
            checkpoint=self.checkpoint,
            threshold=self.threshold,
            window_s=self.window_s,
            hop_s=self.hop_s,
            merge_gap_s=self.merge_gap_s,
            min_event_windows=self.min_event_windows,
            backbone=self.backbone,
        )
        return {
            "head": "events",
            "detector": "known_species_clap_probe",
            "known_species": result["trained_labels"],
            "duration_s": result["duration_s"],
            "window_s": result["window_s"],
            "hop_s": result["hop_s"],
            "threshold": result["threshold"],
            "merge_gap_s": result["merge_gap_s"],
            "min_event_windows": result["min_event_windows"],
            "effective_min_event_windows": result["effective_min_event_windows"],
            "num_windows": result["num_windows"],
            "num_detected_windows": result["num_detected_windows"],
            "num_events": result["num_events"],
            "events": result["events"],
            "diagnostics": {
                "detected_windows": result["detected_windows"],
            },
        }


def load(
    checkpoint_dir: Path | None,
    params: dict[str, Any],
    extra: dict | None = None,
) -> KnownSpeciesEventDetector:
    checkpoint = params.get("checkpoint")
    if checkpoint is None and checkpoint_dir is not None:
        checkpoint_path = checkpoint_dir / "best_probe.pt"
    elif checkpoint is None:
        checkpoint_path = None
    else:
        checkpoint_path = Path(str(checkpoint))

    return KnownSpeciesEventDetector(
        checkpoint=checkpoint_path,
        threshold=float(params.get("threshold", 0.55)),
        window_s=float(params.get("window_s", 5.0)),
        hop_s=float(params.get("hop_s", 1.0)),
        merge_gap_s=float(params.get("merge_gap_s", 1.0)),
        min_event_windows=int(params.get("min_event_windows", 7)),
    )


def generate(state: KnownSpeciesEventDetector, seed: int | None = None, **_ignored) -> dict:
    raise NotImplementedError(
        "Layer E known-species event detection is upload-based. "
        "Use analyze(state, audio_path) through the registry /analyze dispatch."
    )


def analyze(state: KnownSpeciesEventDetector, audio_path: str | Path) -> dict:
    return state.analyze(audio_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--hop-s", type=float, default=1.0)
    parser.add_argument("--merge-gap-s", type=float, default=1.0)
    parser.add_argument("--min-event-windows", type=int, default=7)
    args = parser.parse_args()

    detector = KnownSpeciesEventDetector(
        threshold=args.threshold,
        window_s=args.window_s,
        hop_s=args.hop_s,
        merge_gap_s=args.merge_gap_s,
        min_event_windows=args.min_event_windows,
    )
    print(json.dumps(detector.analyze(args.audio_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
