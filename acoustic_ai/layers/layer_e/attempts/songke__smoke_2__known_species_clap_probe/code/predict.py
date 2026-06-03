"""Predict trained species probabilities with the CLAP probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clap_backbone import CLAPBackbone
from common import build_probe, device, load_config, project_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="Path to an audio clip.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    result = predict_clip(args.audio, checkpoint=args.checkpoint, threshold=args.threshold)
    print(json.dumps(result, indent=2))
    return 0


def predict_clip(
    audio_path: Path,
    *,
    checkpoint: Path | None = None,
    threshold: float = 0.6,
    backbone: CLAPBackbone | None = None,
) -> dict:
    cfg = load_config()
    labels = list(cfg["data"]["labels"])
    model_dir = project_path(cfg["output"]["model_dir"])
    checkpoint_path = checkpoint or model_dir / "best_probe.pt"
    run_device = device()

    saved = torch.load(checkpoint_path, map_location=run_device)
    probe = build_probe(
        int(saved["in_dim"]),
        len(labels),
        str(saved["arch"]),
        int(saved["hidden"]),
    ).to(run_device)
    probe.load_state_dict(saved["state_dict"])
    probe.eval()

    clap = backbone or CLAPBackbone(device=str(run_device))
    embedding = clap.embed_audio([audio_path], verbose=False)
    features = torch.from_numpy(embedding).float().to(run_device)

    with torch.no_grad():
        probs = torch.softmax(probe(features), dim=1).squeeze(0).cpu()

    scores = {label: float(probs[idx]) for idx, label in enumerate(labels)}
    top_idx = int(torch.argmax(probs).item())
    top_label = labels[top_idx]
    confidence = float(probs[top_idx])

    return {
        "audio_path": str(audio_path),
        "top_label": top_label,
        "confidence": confidence,
        "detected": confidence >= threshold,
        "threshold": threshold,
        "scores": scores,
        "trained_labels": labels,
    }


if __name__ == "__main__":
    raise SystemExit(main())
