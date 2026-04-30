"""Module E — species and acoustic event detector.

Detects biologically meaningful events in an uploaded clip.
Uses BirdNET pseudo-labels or high-confidence A2O annotations.

Inputs:  wav_path or mel spectrogram
Outputs: list of {"label": str, "confidence": float, "onset_s": float, "offset_s": float}
"""
# TODO: implement event detection (BirdNET integration or annotation lookup)
# See .claude/context/analysis_components.md — Component C
