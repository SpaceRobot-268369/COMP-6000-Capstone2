# E-B MVP5 Demo Notes

This is an analysis attempt, not a generation attempt. Given an uploaded audio
clip, it estimates the audible weather layer:

```json
{
  "wind_intensity": "none | light | moderate | strong",
  "rain_intensity": "none | light | moderate | heavy",
  "thunder_intensity": "none",
  "confidence": 0.0
}
```

The detector uses a frozen CLAP audio encoder to turn the clip into an audio
embedding, then trains small rain and wind heads on audited Site257 weather
clips. DSP features are included as an explainability/support channel.

The key comparison is whether CLAP gives a better wind boundary than the
previous PANNs/DSP heads.
