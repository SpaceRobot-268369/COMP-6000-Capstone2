/* composeNarration.js — resolved scene params → a second-person narration.

   The narration is the center text on the immersive screen, revealed one word
   at a time. It's assembled from template fragments so it always reads as a
   placed, sensory moment rather than a parameter dump:

     place  →  season/time mood  →  weather clause  →  event clause  →  closing

   Mood language is drawn from the immersive DESIGN_SPEC 16-cell matrix. A later
   version could swap this for LLM-written prose without touching the UI. */

// [season][time] — one sensory sentence describing light + scene for the cell.
const MOOD = {
  spring: {
    dawn: "It's a spring dawn — a violet-to-rose sky, cool mist low to the ground, pale pollen drifting past you.",
    morning: "It's a spring morning — clean blue-white light, a fresh green undertone, pollen catching the sun.",
    afternoon: "It's a spring afternoon — hazy gold-green warmth, pollen hanging lazy in the air.",
    night: "It's a spring night — deep blue-black, a cold moon, faint pollen drifting like static.",
  },
  summer: {
    dawn: "It's a summer dawn — a warm amber horizon over indigo, heavy dew haze, dust motes turning in the air.",
    morning: "It's a summer morning — bright and saturated, a high clear sky, dust shimmering around you.",
    afternoon: "It's a hot summer afternoon — golden and hazy, the heat shimmering, the light bleached and slow.",
    night: "It's a summer night — warm and dense, the moon soft through the humidity.",
  },
  autumn: {
    dawn: "It's an autumn dawn — rose-gold sun low through blue mist, amber leaves letting go around you.",
    morning: "It's an autumn morning — cool clean light with an ochre cast, leaves loose on the breeze.",
    afternoon: "It's an autumn afternoon — long golden light, a rust-and-amber world, leaves spiralling down.",
    night: "It's an autumn night — cold indigo, sparse leaves, a distant moon over near-bare branches.",
  },
  winter: {
    dawn: "It's a winter dawn — pale grey-pink and desaturated, snow drifting, your breath sharp in the cold mist.",
    morning: "It's a winter morning — flat blue-white, near-monochrome, slow snow over bare silhouettes.",
    afternoon: "It's a winter afternoon — a low weak sun, long blue shadows, snow glittering.",
    night: "It's a cold winter night — blue-black and still, a sharp moon, snow falling through the dark.",
  },
};

function joinList(items) {
  if (items.length === 0) return '';
  if (items.length === 1) return items[0];
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

function weatherClause({ rain, thunder, rainAmount }) {
  const heavy = rainAmount >= 0.85;
  const light = rainAmount <= 0.4;
  if (rain && thunder) {
    return ` Rain moves through the canopy overhead, and somewhere far off, thunder.`;
  }
  if (rain) {
    if (heavy) return ` Rain comes down hard through the canopy overhead.`;
    if (light) return ` A light rain drifts through the canopy overhead.`;
    return ` Rain moves steadily through the canopy overhead.`;
  }
  if (thunder) return ` Somewhere far off, thunder rolls across the sky.`;
  return '';
}

function eventClause(events) {
  if (!events || events.length === 0) return '';
  return ` You can pick out ${joinList(events)} in the soundscape.`;
}

export function composeNarration(resolved) {
  const { season = 'autumn', time = 'dawn' } = resolved || {};
  const mood = (MOOD[season] && MOOD[season][time]) || MOOD.autumn.dawn;

  return (
    "You're standing at the edge of a dry woodland. " +
    mood +
    weatherClause(resolved || {}) +
    eventClause(resolved && resolved.events) +
    ' This is what the recording remembers.'
  );
}
