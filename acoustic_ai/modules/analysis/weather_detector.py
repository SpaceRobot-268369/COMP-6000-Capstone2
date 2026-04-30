"""Module E — weather intensity detector.

Estimates audible wind and rain intensity from an uploaded audio clip.
Uses spectral heuristics (broadband energy, low-freq modulation, high-freq
rain texture) or a small classifier trained on curated weather labels.

Inputs:  mel spectrogram (128, T)
Outputs: {"wind_intensity": "none|light|moderate|strong",
          "rain_intensity": "none|light|moderate|heavy",
          "confidence": float}
"""
# TODO: implement weather detection (start with spectral heuristics)
# See .claude/context/analysis_components.md — Component B
