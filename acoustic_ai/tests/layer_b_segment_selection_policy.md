# Layer B Segment Selection Policy

## Purpose

Layer B should not only retrieve relevant weather audio files, but also provide useful audio segments for later mixing in Layer D.

For short outputs, Layer B may return one selected segment.

For longer outputs, such as a 1-minute generated soundscape, Layer B should return a sequence of compatible segments rather than repeating one 10-second segment.

## Core Idea

A segment is not the final output.

A segment is an analysis and mixing unit.

Layer B should use short windows to understand and rank useful parts of audio, while Layer D arranges these segments into a longer final soundscape.

```text
Weather query
→ retrieve relevant weather assets
→ split assets into candidate segments
→ rank segments using CLAP embeddings
→ validate segments using spectrogram/audio features
→ return segment sequence metadata
→ Layer D mixes and arranges the final audio
```

## Segment Window

For MVP and near-future implementation, each weather audio file can be split into fixed-length windows.

Recommended default:

```text
10-second window
2-second overlap
```

The 10-second window is used for analysis, not necessarily as the final output duration.

## Why Use Segment Windows?

Short segments help the system identify useful parts inside a longer audio file.

A full audio file may contain:

- useful rain texture
- silence
- sudden noise
- unstable wind
- thunder events
- irrelevant background sound

Segment-level analysis allows Layer B to choose better parts instead of blindly using the whole file.

## Segment Sequence for Longer Generation

If the final requested output is longer than one segment, Layer B should return multiple compatible segments.

For example, for a 60-second output, Layer B may return:

```json
{
  "target_duration": 60,
  "segments": [
    {
      "file": "acoustic_ai/data/weather/rain/forest_rain_canopy.wav",
      "start_time": 12.0,
      "duration": 12.0,
      "role": "base_rain",
      "fade_in": 1.0,
      "fade_out": 1.0
    },
    {
      "file": "acoustic_ai/data/weather/rain/forest_drizzle_light.wav",
      "start_time": 33.0,
      "duration": 10.0,
      "role": "rain_texture",
      "fade_in": 1.0,
      "fade_out": 1.0
    },
    {
      "file": "acoustic_ai/data/weather/thunder/distant_thunder_rolling.wav",
      "start_time": 5.0,
      "duration": 8.0,
      "role": "thunder_accent",
      "fade_in": 0.5,
      "fade_out": 2.0
    }
  ]
}
```

Layer D is responsible for arranging these segments on the final timeline.

## Layer B vs Layer D Responsibility

Layer B is responsible for:

- retrieving relevant weather assets
- selecting useful candidate segments
- ranking segments by semantic similarity
- validating segments using spectrogram/audio features
- returning segment metadata

Layer D is responsible for:

- arranging segments on a timeline
- looping or extending ambience beds
- applying volume control
- applying crossfades
- mixing multiple layers into the final output

## Weather-Type Specific Policy

### Wind

Wind segments should be:

- continuous
- not overly muddy
- not dominated by microphone turbulence
- suitable for looping or layering
- stable enough to work as a background texture

For long outputs, Layer B can return several compatible wind segments to avoid obvious repetition.

### Rain

Rain segments should be:

- texture-rich
- consistent in density
- not dominated by insects, traffic, or speech
- suitable as an ambience layer
- stable enough to crossfade with other rain segments

For long outputs, Layer B should prefer rain segments that can act as a base bed, then optionally add lighter or heavier rain texture segments.

### Thunder

Thunder segments should be:

- event-focused
- contain a clear thunder roll or thunder peak
- include natural decay after the thunder event
- avoid overly sharp or cinematic thunder effects

Thunder should usually be treated as an accent layer, not a continuous loop.

For long outputs, Layer B may return one or more thunder event segments, while Layer D decides where to place them.

## Spectrogram-Based Validation

After candidate segments are ranked by CLAP similarity, Layer B should inspect audio or spectrogram features.

Useful validation checks include:

- avoid long silent sections
- avoid clipping or sudden unwanted spikes
- avoid unstable background noise
- prefer continuous texture for wind and rain
- prefer clear transient events for thunder
- avoid segments where another sound dominates the target weather type

## MVP Implementation

The current MVP supports file-level CLAP retrieval.

The next implementation step is segment-level selection:

```text
file-level retrieval
→ split selected files into windows
→ compute segment-level embeddings
→ rank segments by query similarity
→ validate using spectrogram/audio features
→ return segment sequence metadata to Layer D
```

## Smoke Test

Run the Layer B segment-selection smoke test from the repository root:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/tests/layer_b_segment_selection_smoke.py
```

The smoke test checks three cases:

- wind: strong natural forest wind ambience
- rain: light drizzle under forest canopy
- thunder: distant rolling thunderstorm ambience

Expected prerequisites:

- `laion_clap` is available in `acoustic_ai/.venv`
- weather embedding indexes exist under `acoustic_ai/data/weather/metadata/`
- selected weather WAV files exist at the indexed paths

Pass criteria:

- each case returns at least one segment for the requested weather type
- selected files exist locally
- segment start/duration metadata is valid
- wind/rain segments are not mostly silent and are stable enough for texture use
- thunder segments are not mostly silent
- selected clips are exported to `model/candidates/murphy/layer-b-segment-selection-smoke/outputs/` for manual listening

The smoke test is not a substitute for listening. It catches broken retrieval,
missing assets, bad segment metadata, silence, and clipping; the developer still
needs to listen to the exported clips to confirm semantic quality.

## Output Format

Layer B should eventually return:

```json
{
  "ok": true,
  "query": "heavy forest rain with distant thunder",
  "target_duration": 60,
  "results": [
    {
      "weather_type": "rain",
      "file": "acoustic_ai/data/weather/rain/forest_rain_canopy.wav",
      "score": 0.53,
      "segment": {
        "start_time": 12.0,
        "duration": 12.0,
        "fade_in": 1.0,
        "fade_out": 1.0,
        "role": "base_rain"
      },
      "reason": "High semantic similarity and stable rain texture."
    },
    {
      "weather_type": "thunder",
      "file": "acoustic_ai/data/weather/thunder/distant_thunder_rolling.wav",
      "score": 0.46,
      "segment": {
        "start_time": 5.0,
        "duration": 8.0,
        "fade_in": 0.5,
        "fade_out": 2.0,
        "role": "thunder_accent"
      },
      "reason": "Clear distant thunder roll with natural decay."
    }
  ]
}
```

## Summary

Layer B should not simply select one audio file or one fixed 10-second clip.

For realistic procedural soundscape generation, Layer B should select useful audio units, and for longer outputs it should return a compatible sequence of segments.

Layer D then turns those selected segments into the final mixed soundscape.
