import { useEffect, useRef, useState } from "react";

/* Preset prompts. `tag` marks demo intent so we (and an audience) know a chip
   is deliberately crafted:
     - "safe"    : built only from species BOTH Layer C models share
                   (Bronze-cuckoo / Nightjar), so it generates on either model.
                   The two safe chips differ in EVENT FREQUENCY (sparse vs
                   frequent) to show off event-density control.
     - "partial" : partly voiceable — a real site species plus an element the
                   site/models can't produce (ocean). Kept on a shared species
                   so it stays "partial" regardless of the selected model.
     - "invalid" : nothing the remote dry-woodland site or our models can voice
                   (urban scene, no site fauna) — a deliberate rejection demo.
   Untagged chips are ordinary free-form examples. */
const PRESET_PROMPTS = [
  { text: "Summer night, a lone Spotted Nightjar churring every so often", tag: "safe" },
  { text: "Summer morning, restless Horsfield's Bronze-cuckoo whistling over and over", tag: "safe" },
  { text: "Summer night, a Spotted Nightjar calling, with ocean waves breaking on a beach", tag: "partial" },
  { text: "Midday city traffic downtown, car horns, sirens and a passing subway train", tag: "invalid" },
  { text: "Windy spring afternoon, distant birdsong" },
  { text: "Cold winter dawn, light drizzle" },
  { text: "Gusty autumn morning, rain on the wind" },
  { text: "Warm spring dusk, insects and a breeze" },
];

// Short badge label + class suffix per tag. Untagged presets render no badge.
const PRESET_TAGS = {
  safe: { label: "Safe", className: "safe" },
  partial: { label: "Partial", className: "partial" },
  invalid: { label: "Invalid demo", className: "invalid" },
};

/* ---- Guided option selectors ----
   The rails let users build a prompt by choosing rather than typing. Each
   selection is serialized back into the composer text via composeSelections()
   so the same resolvePrompt() keyword path still drives the AI workflow — the
   labels/phrases below are written to contain the keywords resolvePrompt looks
   for (season/time/weather/event words). Source of truth stays the textarea.

   Left rail = the "stage" (Layer A season + diel, Layer B weather).
   Right rail = the "voices" (Layer C species — a retrieval set, so it can grow
   large; rendered in a constrained-height, edge-faded scroll area). */

// Layer A — single-choice backdrop.
const SEASONS = ["spring", "summer", "autumn", "winter"];
const TIMES = ["dawn", "morning", "afternoon", "night"];

// Layer B — weather, single-choice intensity + thunder/wind toggles.
// `phrase` re-parses through resolvePrompt's RAIN_WORDS / amount cues.
const WEATHERS = [
  { key: "none", label: "Clear", phrase: "" },
  { key: "light", label: "Light drizzle", phrase: "light drizzle" },
  { key: "rain", label: "Steady rain", phrase: "rain" },
  { key: "downpour", label: "Heavy downpour", phrase: "heavy downpour" },
];

// Layer C — species checklist. The list is supplied by the parent (it depends
// on the Layer C model selected on /dev/settings — see demo/layerCSpecies.js);
// `DEFAULT_SPECIES` is only a fallback for standalone rendering.
const DEFAULT_SPECIES = [];

const EMPTY_SEL = {
  season: null,
  time: null,
  weather: "none",
  thunder: false,
  wind: false,
  species: [],
};

function titleCase(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

/* Turn the current selections into a natural-language prompt that mirrors the
   preset style ("Cold winter dawn, light drizzle, a southern boobook owl"). */
function composeSelections(sel, speciesList) {
  const parts = [];

  if (sel.season || sel.time) {
    parts.push([sel.season, sel.time].filter(Boolean).join(" "));
  }

  const weather = WEATHERS.find((w) => w.key === sel.weather);
  if (weather && weather.phrase) parts.push(weather.phrase);
  if (sel.thunder) parts.push("thunder");
  if (sel.wind) parts.push("wind");

  for (const key of sel.species) {
    const sp = speciesList.find((s) => s.key === key);
    if (sp) parts.push(sp.phrase);
  }

  const text = parts.join(", ");
  return text ? titleCase(text) : "";
}

/* The demo entry interface: a calm, chatbot-style prompt screen (phase
   "prompt") that becomes a thinking transcript while the scene resolves
   (phase "generating"). Presentational — DemoPage owns the phase, the echoed
   user message, and the staged status line. */
export default function PromptChat({ phase, userMessage, statusLine, species = DEFAULT_SPECIES, notice = null, onSubmit }) {
  const [value, setValue] = useState("");
  const [sel, setSel] = useState(EMPTY_SEL);
  const inputRef = useRef(null);
  const generating = phase === "generating";

  useEffect(() => {
    if (phase === "prompt") inputRef.current?.focus();
  }, [phase]);

  // The species set can change when the active Layer C model changes; drop any
  // chosen species that the new model can't voice so the prompt stays coherent.
  useEffect(() => {
    setSel((prev) => {
      const allowed = prev.species.filter((k) => species.some((s) => s.key === k));
      if (allowed.length === prev.species.length) return prev;
      const next = { ...prev, species: allowed };
      setValue(composeSelections(next, species));
      return next;
    });
  }, [species]);

  // Apply a selection change and push the rebuilt prompt into the composer.
  function applySel(next) {
    setSel(next);
    setValue(composeSelections(next, species));
  }

  function setSeason(season) {
    applySel({ ...sel, season: sel.season === season ? null : season });
  }
  function setTime(time) {
    applySel({ ...sel, time: sel.time === time ? null : time });
  }
  function setWeather(weather) {
    applySel({ ...sel, weather });
  }
  function toggleThunder() {
    applySel({ ...sel, thunder: !sel.thunder });
  }
  function toggleWind() {
    applySel({ ...sel, wind: !sel.wind });
  }
  function toggleSpecies(key) {
    const species = sel.species.includes(key)
      ? sel.species.filter((s) => s !== key)
      : [...sel.species, key];
    applySel({ ...sel, species });
  }

  function submit() {
    const text = value.trim();
    if (!text || generating) return;
    onSubmit(text);
  }

  // Prevent submit on shift+enter, submit on enter
  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  const isPrompt = phase === "prompt";

  return (
    <div className={`demo-chat theme-generation${generating ? " generating" : ""}`}>
      <div className="demo-chat-inner">
        <header className="demo-chat-head">
          <p className="demo-eyebrow">SPECULATIVE SOUNDSCAPE</p>
          <h1>Describe the soundscape you want to step into</h1>
          <p className="demo-sub">
            A scene, a season, the weather, what you can hear — we'll place you inside it.
          </p>
        </header>

        <div className="demo-transcript" aria-live="polite">
          {userMessage && (
            <div className="demo-bubble user">
              <span>{userMessage}</span>
            </div>
          )}
          {generating && (
            <div className="demo-bubble assistant thinking">
              <span className="demo-dots" aria-hidden="true"><i /><i /><i /></span>
              <span className="demo-status">{statusLine}</span>
            </div>
          )}
          {notice && (
            <div className={`demo-bubble assistant notice ${notice.kind}`} role="status">
              <span>{notice.text}</span>
            </div>
          )}
        </div>

        {isPrompt && (
          <>
            <aside className="demo-rail demo-rail-left" aria-label="Scene and weather">
                <div className="demo-rail-group">
                  <p className="demo-rail-label">Season</p>
                  <div className="demo-seg">
                    {SEASONS.map((s) => (
                      <button
                        key={s}
                        type="button"
                        className={`demo-seg-btn${sel.season === s ? " active" : ""}`}
                        onClick={() => setSeason(s)}
                      >
                        {titleCase(s)}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="demo-rail-group">
                  <p className="demo-rail-label">Time of day</p>
                  <div className="demo-seg">
                    {TIMES.map((t) => (
                      <button
                        key={t}
                        type="button"
                        className={`demo-seg-btn${sel.time === t ? " active" : ""}`}
                        onClick={() => setTime(t)}
                      >
                        {titleCase(t)}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="demo-rail-group">
                  <p className="demo-rail-label">Weather</p>
                  <div className="demo-seg">
                    {WEATHERS.map((w) => (
                      <button
                        key={w.key}
                        type="button"
                        className={`demo-seg-btn${sel.weather === w.key ? " active" : ""}`}
                        onClick={() => setWeather(w.key)}
                      >
                        {w.label}
                      </button>
                    ))}
                  </div>
                  <div className="demo-toggle-row">
                    <button
                      type="button"
                      className={`demo-opt-chip${sel.thunder ? " active" : ""}`}
                      onClick={toggleThunder}
                      aria-pressed={sel.thunder}
                    >
                      Thunder
                    </button>
                    <button
                      type="button"
                      className={`demo-opt-chip${sel.wind ? " active" : ""}`}
                      onClick={toggleWind}
                      aria-pressed={sel.wind}
                    >
                      Wind
                    </button>
                  </div>
                </div>
              </aside>

              <aside className="demo-rail demo-rail-right" aria-label="Species">
                <div className="demo-rail-group">
                  <p className="demo-rail-label">
                    Species
                    <span className="demo-rail-count"> · {species.length} available</span>
                    {sel.species.length > 0 && (
                      <span className="demo-rail-count"> · {sel.species.length} chosen</span>
                    )}
                  </p>
                  <div className="demo-species-scroll">
                    <div className="demo-opt-grid">
                      {species.map((s) => (
                        <button
                          key={s.key}
                          type="button"
                          className={`demo-opt-chip${sel.species.includes(s.key) ? " active" : ""}`}
                          onClick={() => toggleSpecies(s.key)}
                          aria-pressed={sel.species.includes(s.key)}
                        >
                          {s.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </aside>

            <div className="demo-center">
              <div className="demo-presets">
              {PRESET_PROMPTS.map((p) => {
                const tag = p.tag ? PRESET_TAGS[p.tag] : null;
                return (
                  <button
                    key={p.text}
                    type="button"
                    className={`demo-preset-chip${tag ? ` tagged ${tag.className}` : ""}${value === p.text ? " selected" : ""}`}
                    onClick={() => { setSel(EMPTY_SEL); setValue(p.text); inputRef.current?.focus(); }}
                  >
                    {tag && <span className={`demo-preset-tag ${tag.className}`}>{tag.label}</span>}
                    {p.text}
                  </button>
                );
              })}
              </div>
            </div>

            <div className="demo-composer">
              <textarea
                ref={inputRef}
                rows={1}
                value={value}
                placeholder="e.g. a misty autumn dawn with rain and distant thunder, a boobook calling…"
                onChange={(e) => setValue(e.target.value)}
                onKeyDown={onKeyDown}
              />
              <button
                type="button"
                className="demo-send"
                onClick={submit}
                disabled={!value.trim()}
                aria-label="Generate scene"
              >
                ↵
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
