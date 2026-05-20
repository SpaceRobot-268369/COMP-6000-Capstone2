# Layer B Weather Asset Policy (MVP)

## 1. Purpose

Layer B is a controllable weather modulation layer within the speculative
soundscape generation pipeline.

Its purpose is to provide weather-related acoustic textures such as wind, rain,
and thunder for Layer D mixing, rather than generating complete environmental
soundscapes.

Layer B should:

- enhance environmental realism;
- respond to weather and environmental parameters;
- remain controllable and explainable;
- avoid overpowering other layers.

Layer B does not:

- replace Layer A ambience;
- generate full ecological scenes;
- act as a standalone audio generation model.

## 2. Layer B Design Philosophy

Layer B follows a weather-dominant ecological texture retrieval approach instead
of studio-isolated weather SFX generation.

This means:

- slight natural background textures are acceptable;
- perfectly isolated weather sounds are not required;
- retrieval quality is based on dominant weather characteristics.

The system prioritizes:

- controllability;
- ecological realism;
- stable Layer D mixing compatibility.

## 3. Accepted Asset Types

| Category | Description |
|---|---|
| Wind ambience | Wind-dominant natural ambience |
| Rain ambience | Rain-dominant environmental recordings |
| Thunder ambience | Thunder or storm environmental textures |
| Storm ambience | Combined thunder and rain where weather remains dominant |

## 4. Allowed Background Elements

The following low-level background elements are acceptable if weather remains
dominant:

| Background element | Allowed |
|---|---|
| Leaves rustling | Yes |
| Faint insects | Yes |
| Distant birds | Limited |
| Light environmental room tone | Yes |
| Subtle forest texture | Yes |

These elements are considered part of ecological realism.

## 5. Rejected Asset Types

| Rejected type | Reason |
|---|---|
| Music | Contaminates mixing |
| Human speech | Retrieval interference |
| Podcasts or dialogue | Non-environmental |
| Traffic-dominant recordings | Conflicts with ecological focus |
| Bird-dominant recordings | Conflicts with Layer C |
| Cinematic or game SFX | Unrealistic texture |
| Meme or edited audio | Inconsistent quality |
| Heavily compressed or distorted audio | Embedding degradation |

## 6. Technical Asset Requirements

### Preferred Formats

Priority order:

```text
wav > flac > webm
```

### Duration

Recommended duration: 30 seconds to 5 minutes.

Very short clips should be avoided unless they are highly reusable.

### Audio Quality

Assets should:

- avoid clipping;
- avoid excessive normalization;
- maintain stable ambience;
- preserve environmental continuity.

Preferred asset qualities:

- stereo recordings;
- field recordings;
- natural ambience.

## 7. Weather Categories

### Current MVP Categories

- `wind`
- `rain`
- `thunder`

### Optional Future Categories

- `light_rain`
- `heavy_rain`
- `drizzle`
- `strong_wind`
- `distant_thunder`
- `storm_front`
- `coastal_wind`
- `snow`
- `hail`

## 8. Asset Metadata Requirements

Every accepted asset should include metadata.

Assets without clear source and license metadata should not be accepted.

Minimum required fields:

| Field | Description |
|---|---|
| `filename` | Asset filename |
| `category` | `wind`, `rain`, or `thunder` |
| `intensity` | `light`, `medium`, or `heavy` |
| `duration` | Clip duration |
| `source` | Source such as Freesound or BBC |
| `license` | License such as CC0 or Attribution |
| `tags` | Descriptive tags |
| `local_path` | Server path |
| `s3_path` | S3 object path |

## 9. Storage Structure

### Local Server Structure

```text
~/layer_b_assets/
    raw_downloads/
    wind/
    rain/
    thunder/
    metadata/
```

### S3 Structure

Full S3 prefix:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/training_dataset/layer-b/weather_assets/
```

```text
dataset/training_dataset/layer-b/weather_assets/
    wind/
    rain/
    thunder/
    metadata/
```

## 10. Retrieval Pipeline Philosophy

Layer B retrieval should prioritize:

1. weather dominance;
2. ecological realism;
3. mixing controllability;
4. stable ambience continuity.

The retrieval system should not prioritize:

- dramatic cinematic effects;
- isolated synthetic SFX;
- maximum loudness;
- overly complex sound events.

## 11. Embedding / Pretrained Model Usage

Layer B may use pretrained audio models for:

- embedding extraction;
- semantic similarity;
- reranking;
- weather classification assistance.

Examples:

- CLAP;
- YAMNet;
- PANNs.

Pretrained models are used to improve retrieval quality, not to generate weather
audio directly.

## 12. Current MVP Scope

| Category | Approximate asset count |
|---|---:|
| `wind` | 20-30 |
| `rain` | 20-30 |
| `thunder` | 10-15 |

The MVP focuses on:

- curated asset quality;
- stable retrieval;
- Layer D integration;
- reproducible workflow.

Large-scale automated crawling and full generative weather synthesis are outside
current MVP scope.

## 13. Integration with Layer D

Layer B outputs:

- selected weather assets;
- gain and intensity parameters;
- retrieval metadata;
- explanation JSON.

Layer D is responsible for:

- final mixing;
- balancing Layer A, Layer B, and Layer C;
- rendering the final soundscape.

Layer B should remain modular and independently testable.
