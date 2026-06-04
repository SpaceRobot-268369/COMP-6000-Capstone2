/* resolvePrompt.js — turn a free-text demo prompt into immersive scene params.

   This is the single swap-point for the demo's "understanding" of the prompt.
   For the quick demo it's a frontend keyword heuristic; a later version can
   replace the body with a backend / LLM call that returns the same shape:

     { season, time, rain, rainAmount, thunder, events, raw }

   - season ∈ {spring, summer, autumn, winter}   (default autumn)
   - time   ∈ {dawn, morning, afternoon, night}  (default dawn)
   - rain        : boolean
   - rainAmount  : 0.15–1   (light → heavy)
   - thunder     : boolean
   - events      : string[] — natural-language noun phrases for the narration
*/

const SEASON_WORDS = {
  spring: ['spring', 'blossom', 'bloom'],
  summer: ['summer', 'hot', 'humid', 'cicada', 'cicadas'],
  autumn: ['autumn', 'fall', 'falling leaves', 'leaves'],
  winter: ['winter', 'snow', 'snowy', 'frost', 'frozen', 'cold'],
};

const TIME_WORDS = {
  dawn: ['dawn', 'sunrise', 'daybreak', 'first light', 'early morning'],
  morning: ['morning', 'mid-morning'],
  afternoon: ['afternoon', 'midday', 'noon', 'daytime'],
  night: ['night', 'evening', 'dusk', 'nightfall', 'midnight', 'moonlit', 'moon', 'twilight'],
};

// keyword → narration noun phrase. Order matters: more specific first.
const EVENT_WORDS = [
  [['boobook', 'owl', 'owls'], 'a boobook owl'],
  [['kookaburra', 'kookaburras'], 'a kookaburra'],
  [['frog', 'frogs'], 'frogs'],
  [['cicada', 'cicadas'], 'cicadas'],
  [['cricket', 'crickets'], 'crickets'],
  [['birdsong', 'birds', 'bird', 'chorus'], 'birdsong'],
  [['insect', 'insects'], 'insects'],
  [['wind', 'breeze', 'gust', 'gusts', 'windy'], 'wind'],
];

const RAIN_WORDS = ['rain', 'rainy', 'raining', 'drizzle', 'shower', 'showers', 'downpour', 'wet', 'storm', 'stormy', 'thunderstorm'];
const THUNDER_WORDS = ['thunder', 'thundery', 'lightning', 'storm', 'stormy', 'thunderstorm'];

// Match on whole words so e.g. autumn's "fall" doesn't fire inside "falling"
// (which is a winter/snow cue), and "light" doesn't fire inside "first light".
function matchCount(text, words) {
  let n = 0;
  for (const w of words) {
    const esc = w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (new RegExp(`\\b${esc}\\b`).test(text)) n += 1;
  }
  return n;
}

function has(text, words) {
  return matchCount(text, words) > 0;
}

// Score every candidate and take the strongest signal (ties → declaration
// order). More robust than first-match when a prompt mentions several cues.
function pick(text, table, fallback) {
  let best = fallback;
  let bestScore = 0;
  for (const [key, words] of Object.entries(table)) {
    const score = matchCount(text, words);
    if (score > bestScore) {
      bestScore = score;
      best = key;
    }
  }
  return best;
}

export function resolvePrompt(raw) {
  const text = ` ${String(raw || '').toLowerCase()} `;

  const season = pick(text, SEASON_WORDS, 'autumn');
  const time = pick(text, TIME_WORDS, 'dawn');

  const rain = has(text, RAIN_WORDS);
  const thunder = has(text, THUNDER_WORDS);

  let rainAmount = 0.6;
  if (rain) {
    if (has(text, ['downpour', 'heavy', 'torrential', 'pouring'])) rainAmount = 0.9;
    else if (has(text, ['drizzle', 'faint', 'gentle', 'soft'])) rainAmount = 0.35;
  }

  const events = [];
  for (const [words, phrase] of EVENT_WORDS) {
    if (has(text, words) && !events.includes(phrase)) events.push(phrase);
  }

  return { season, time, rain, rainAmount, thunder, events, raw: String(raw || '').trim() };
}
