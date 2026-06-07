# Eco-Acoustic — Immersive Experience Screen
### Visual Design Spec · v1

---

## 1. Concept & Mood

> *"This is what the recording remembers."*

After analysis finishes, the listener doesn't get a dashboard — they get **placed**. The screen
becomes the woodland edge the recording came from: scattered trees, undergrowth, layered
atmospheric depth, re-lit and re-dressed by season and time of day. A few **hero tree
silhouettes** anchor every frame; everything else (sky, haze, light, particle life, grade) shifts
to tell us *when* we are standing there.

The grade bias is **moody & filmic**: heavy contrast, deep crushed shadows, generous negative
space in the sky, and a single dominant key light. The scene is never static — fog drifts, particles
hang and fall, the camera breathes — but it is *restrained*. Taste over spectacle. The signature
moment is the analysis text resolving one word at a time, like the opening titles of a film.

**One site, sixteen memories.** Same trees, same horizon line, sixteen different states of light.

---

## 2. The 16-Cell Mood Matrix

Same physical woodland edge across all cells. Season drives *vegetation state, particle life,
saturation, foliage*; time of day drives *key-light angle & colour, sky gradient, brightness,
atmosphere*. They compose.

| | **Dawn** (low key, ~4°) | **Morning** (~28°) | **Afternoon** (~52°) | **Night** (moon ~46°) |
|---|---|---|---|---|
| **Spring** | Violet-to-rose sky, cool mist, pale pollen drifting. Tender, just-woken. | Clean blue-white sky, fresh green undertone, crisp light, pollen catching sun. | Hazy gold-green, soft warmth, lazy pollen, gentle bloom. | Deep blue-black, cold moon, faint pollen like static. Quiet, expectant. |
| **Summer** | Warm amber horizon over indigo, heavy dew haze, dust motes. Heavy, humid. | Bright, saturated, high clear sky, shimmering dust, lush. | **Hot & hazy** — golden, heat shimmer, bleached highlights, slow dust. The reference "warm afternoon". | Warm-dark, dense, cricket-still, moon through humidity. Velvet night. |
| **Autumn** | **Default.** Rose-gold low sun, blue mist, amber leaves falling. Dry woodland waking. | Cool clean light, ochre cast, leaves on the breeze. Brisk. | Long golden light, rust & amber grade, leaves spiralling down. Elegiac. | Cold indigo, sparse leaves, bare-ish branches, distant moon. Melancholy. |
| **Winter** | Pale grey-pink, desaturated, snow drifting, breath-cold mist. Brittle. | Flat blue-white, near-monochrome, slow snow, bare silhouettes. Silent. | Low weak sun, long blue shadows, snow glittering. Stark, beautiful. | **Cold, desaturated, moonlit, still** — blue-black, sharp moon, snow falling through dark. The reference "winter night". |

**Reading the axes**
- **Dawn** — sun near the horizon, warm key against a cool violet sky, strongest haze, cool shadows.
- **Morning** — sun risen, cleaner air, neutral-cool, the brightest, crispest state.
- **Afternoon** — high warm key, golden haze, bleached highlights, the warmest grade.
- **Night** — moon as key, cold blue, desaturated, dark, stars overhead, lowest brightness.
- **Spring** pollen · **Summer** dust + shimmer · **Autumn** falling leaves · **Winter** snow + bare trees + crushed saturation.

---

## 3. Layering Model

Render order, back to front. Each layer is independent and composites over the one beneath.

```
┌─ TYPOGRAPHY  ── DOM, one word at a time, soft scrim for legibility
├─ POST        ── bloom → colour grade (per-scene) → vignette → film grain → wet-grade (if rain)
├─ WEATHER     ── overlay layer, composes on ANY scene:
│                  · RAIN   (line-streak system + darkened/wet grade + ripple hint)
│                  · THUNDER(screen flash + sky illumination + optional bolt, audio-loose)
├─ SCENE       ── the 16 states:
│                  · sky dome shader (gradient + sun/moon disc + glow + stars + horizon haze)
│                  · exponential fog tinted to horizon
│                  · ground plane (dark, graded)
│                  · LAYERED TREE SILHOUETTES (foreground hero → mid → far, fog-faded)
│                  · ambient PARTICLES (pollen / dust / leaves / snow)
│                  · slow camera breath (+ loose audio sway)
└─ (clear)
```

Weather is a **layer, not a scene** — `autumn-dawn + rain + thunder` all compose together, and rain
& thunder toggle independently.

---

## 4. Motion & Timing

| Element | Timing |
|---|---|
| **Word cadence** | ~280 ms per word, +220 ms extra pause after sentence-final `.` |
| **Per-word reveal** | opacity 0→1, blur(10px)→0, translateY(7px)→0 over ~560 ms, ease-out |
| **Camera breath** | sinusoidal sway, ~18 s period, ±0.4° + tiny dolly; audio amplitude adds up to ~30% |
| **Fog drift** | continuous, ~0.01 u/s, never loops visibly |
| **Particles** | snow slow + sway · leaves flutter + fall · pollen/dust hover + brownian drift |
| **Thunder** | flash to ~0.85 in 40 ms, double-flicker, decay over ~1.1 s; sky lifts in sync; bolt shown ~half the time for ~120 ms |
| **Scene transition** | uniforms cross-fade over ~1.4 s (sky, light, fog, grade) — no hard cut |

**Audio reactivity (loose, tasteful):** smoothed amplitude nudges camera sway + bloom strength;
large transients *may* reinforce a thunder flash. Never a beat-synced music visualizer.

---

## 5. Legibility Approach

The background is moving and photoreal-leaning, so the title sequence gets help:
- A soft **radial scrim** (transparent edges → ~45% black centre) sits between POST and TYPOGRAPHY.
- Title set in a high-contrast **serif** (film-title register), generous tracking, with a low text-shadow.
- Words land **centred**, wrapping naturally; the scrim follows the text block.
- During rain the wet-grade darkens the frame, which only *helps* contrast.

---

## 6. Rendering Approach

Procedural Three.js — composition is dynamic, not a static image swap:
- **Time-of-day is carried by the key light, not just colour.** The sun/moon is *placed in frame*
  and rises across the four times (low on the horizon at dawn → mid at morning → high at afternoon →
  high cold moon at night), so the strongest "what time is it" cue — light position — is always visible.
- **Sky** is a shader on a back-side dome: vertical gradient, sun/moon disc + multi-falloff glow,
  hashed twinkling stars at night, horizon haze band, thunder-flash term.
- **Directional ground** — a shader floor lit from the sun azimuth: a warm light pool and a tight
  streak (longest when the sun is low) read as "light coming from there," giving the same geometry a
  different lit feel each time.
- **Volumetric light shafts (god-rays)** — a radial-blur post pass from the sun's screen position;
  long and golden at dawn/afternoon, short and pale at morning, near-absent at night.
- **Fog** is exponential, colour-matched to the horizon so trees dissolve into the sky.
- **Trees** are procedurally-generated silhouette textures placed at several depth bands; distance +
  fog give layered woodland depth cheaply and read as crushed-black hero silhouettes.
- **Particles** pick up the key-light colour (warm dawn, neutral morning, gold afternoon, cold night).
- **Post** is a hand-rolled pipeline (bright-pass → separable blur → composite with grade/grain/
  vignette) so the whole thing runs from a single folder with **no build step and no ES-module
  loading** — it degrades gracefully on weaker GPUs by dropping bloom resolution and particle counts.

---

## 7. What Would Change for Production

- **Drive state from analysis**, not hardcoded toggles: season/time from timestamp + location,
  rain/thunder from detected acoustic events with confidence + timestamps (flash on the actual
  thunder onset in the waveform).
- **Real audio pipeline**: stream the user's uploaded file, decode for a true amplitude/onset
  envelope; align thunder flashes to detected onsets rather than loose amplitude.
- **Asset-grounded realism**: optional HDRI sky + photoscanned bark/foliage/ground textures and a
  GPU instanced tree system; LOD + frustum culling for the woodland.
- **Adaptive quality**: measure frame time, scale particle counts, bloom passes, and resolution.
- **Accessibility**: honour `prefers-reduced-motion` (freeze drift, instant text), captions for
  detected events, contrast-safe text scrim.
- **Engineering**: move to ES-module Three.js + EffectComposer/UnrealBloom, a real state machine for
  scene transitions, and decouple the dev control panel from the shipping UI.
