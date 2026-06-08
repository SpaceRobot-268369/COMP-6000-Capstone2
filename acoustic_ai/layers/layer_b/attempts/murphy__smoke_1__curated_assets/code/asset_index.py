"""Module B — weather asset library.

Manages the curated weather clip index stored in this attempt's
data/weather/asset_index.csv and the audio files under data/weather/.

The index is component-based so a clip can carry rain, wind, thunder, or a
mixture of those components. See:
  acoustic_ai/layers/layer_b/attempts/murphy__smoke_1__curated_assets/weather_asset_index_schema.md

Intensity buckets:
  wind: none (<2 m/s), light (2-6 m/s), medium (6-10 m/s), heavy (>10 m/s)
  rain: none (0 mm), light (0-2 mm), medium (2-5 mm), heavy (>5 mm)
  thunder: none, light, medium, heavy, or unclear
"""
# TODO: implement asset index loading and clip selection
# See .claude/context/generation_layers.md — Layer B
