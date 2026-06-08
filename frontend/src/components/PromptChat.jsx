import { useEffect, useRef, useState } from "react";

const PRESET_PROMPTS = [
  "Windy spring afternoon, distant birdsong",
  "Still summer night, crickets and a boobook owl",
  "Cold winter dawn, light drizzle",
  "Summer dawn, kookaburra chorus",
  "Gusty autumn morning, rain on the wind",
  "Frosty winter afternoon, gusting wind",
  "Heavy autumn downpour at night, thunder",
  "Warm spring dusk, insects and a breeze",
];

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

// Layer C — species checklist (retrieval set), multi-select.
// Known-to-resolvePrompt species keep their matching keyword so the demo
// narration still fires; the wider set is carried through as prompt text.
const SPECIES = [
  { key: "boobook", label: "Southern boobook", phrase: "a southern boobook owl" },
  { key: "kookaburra", label: "Laughing kookaburra", phrase: "a laughing kookaburra" },
  { key: "magpie", label: "Australian magpie", phrase: "an Australian magpie" },
  { key: "butcherbird", label: "Pied butcherbird", phrase: "a pied butcherbird" },
  { key: "shrikethrush", label: "Grey shrike-thrush", phrase: "a grey shrike-thrush" },
  { key: "williewagtail", label: "Willie wagtail", phrase: "a willie wagtail" },
  { key: "magpielark", label: "Magpie-lark", phrase: "a magpie-lark" },
  { key: "galah", label: "Galah", phrase: "a galah" },
  { key: "corella", label: "Little corella", phrase: "a little corella" },
  { key: "cockatoo", label: "Sulphur-crested cockatoo", phrase: "a sulphur-crested cockatoo" },
  { key: "cockatiel", label: "Cockatiel", phrase: "a cockatiel" },
  { key: "budgerigar", label: "Budgerigar", phrase: "budgerigars" },
  { key: "apostlebird", label: "Apostlebird", phrase: "apostlebirds" },
  { key: "babbler", label: "Grey-crowned babbler", phrase: "a grey-crowned babbler" },
  { key: "noisyminer", label: "Noisy miner", phrase: "a noisy miner" },
  { key: "honeyeater", label: "Spiny-cheeked honeyeater", phrase: "a spiny-cheeked honeyeater" },
  { key: "whistler", label: "Rufous whistler", phrase: "a rufous whistler" },
  { key: "bellbird", label: "Crested bellbird", phrase: "a crested bellbird" },
  { key: "raven", label: "Australian raven", phrase: "an Australian raven" },
  { key: "crow", label: "Torresian crow", phrase: "a Torresian crow" },
  { key: "zebrafinch", label: "Zebra finch", phrase: "zebra finches" },
  { key: "redrumpedparrot", label: "Red-rumped parrot", phrase: "a red-rumped parrot" },
  { key: "mulgaparrot", label: "Mulga parrot", phrase: "a mulga parrot" },
  { key: "bronzewing", label: "Common bronzewing", phrase: "a common bronzewing" },
  { key: "pallidcuckoo", label: "Pallid cuckoo", phrase: "a pallid cuckoo" },
  { key: "frogmouth", label: "Tawny frogmouth", phrase: "a tawny frogmouth" },
  { key: "barkingowl", label: "Barking owl", phrase: "a barking owl" },
  { key: "emu", label: "Emu", phrase: "an emu" },
  { key: "birdsong", label: "Distant birdsong", phrase: "distant birdsong" },
  { key: "frogs", label: "Frogs", phrase: "frogs" },
  { key: "cicadas", label: "Cicadas", phrase: "cicadas" },
  { key: "crickets", label: "Crickets", phrase: "crickets" },
  { key: "insects", label: "Insects", phrase: "insects" },
];

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
function composeSelections(sel) {
  const parts = [];

  if (sel.season || sel.time) {
    parts.push([sel.season, sel.time].filter(Boolean).join(" "));
  }

  const weather = WEATHERS.find((w) => w.key === sel.weather);
  if (weather && weather.phrase) parts.push(weather.phrase);
  if (sel.thunder) parts.push("thunder");
  if (sel.wind) parts.push("wind");

  for (const key of sel.species) {
    const sp = SPECIES.find((s) => s.key === key);
    if (sp) parts.push(sp.phrase);
  }

  const text = parts.join(", ");
  return text ? titleCase(text) : "";
}

/* The demo entry interface: a calm, chatbot-style prompt screen (phase
   "prompt") that becomes a thinking transcript while the scene resolves
   (phase "generating"). Presentational — GenerationPage owns the phase, the echoed
   user message, and the staged status line.

   Phases:
     prompt      — the initial composer + rails + presets
     parsing     — thinking dots while the LLM parses the prompt
     confirm     — parser corrected the prompt; show note + accept/cancel
     rejected    — parser rejected the prompt entirely; show note + try again
     generating  — thinking dots while audio is being generated */
export default function PromptChat({
  phase,
  userMessage,
  statusLine,
  errorMessage = "",
  rejectionNote = "",
  confirmNote = "",
  confirmSummary = "",
  onSubmit,
  onConfirm,
  onCancel,
  onDismissRejection,
}) {
  const [value, setValue] = useState("");
  const [sel, setSel] = useState(EMPTY_SEL);
  const inputRef = useRef(null);
  const isBusy = phase === "generating" || phase === "parsing";

  useEffect(() => {
    if (phase === "prompt") inputRef.current?.focus();
  }, [phase]);

  // Apply a selection change and push the rebuilt prompt into the composer.
  function applySel(next) {
    setSel(next);
    setValue(composeSelections(next));
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
    if (!text || isBusy) return;
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
  const isConfirm = phase === "confirm";
  const isRejected = phase === "rejected";

  return (
    <div className={`demo-chat theme-generation${isBusy ? " generating" : ""}`}>
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

          {/* Parsing / generating: animated thinking dots + status text */}
          {isBusy && (
            <div className="demo-bubble assistant thinking">
              <span className="demo-dots" aria-hidden="true"><i /><i /><i /></span>
              <span className="demo-status">{statusLine}</span>
            </div>
          )}

          {/* Corrected prompt — show note and confirm/cancel buttons */}
          {isConfirm && confirmNote && (
            <div className="demo-bubble assistant demo-bubble-confirm">
              <div className="demo-confirm-content">
                <p className="demo-confirm-note">{confirmNote}</p>
                {confirmSummary && (
                  <p className="demo-confirm-summary">
                    <span className="demo-confirm-summary-label">Adjusted scene:</span>{" "}
                    {confirmSummary}
                  </p>
                )}
                <div className="demo-confirm-actions">
                  <button
                    type="button"
                    className="demo-confirm-btn demo-confirm-accept"
                    onClick={onConfirm}
                  >
                    ✦ Generate this scene
                  </button>
                  <button
                    type="button"
                    className="demo-confirm-btn demo-confirm-cancel"
                    onClick={onCancel}
                  >
                    ← Try a different prompt
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Rejected prompt — show note and try-again button */}
          {isRejected && rejectionNote && (
            <div className="demo-bubble assistant demo-bubble-rejected">
              <div className="demo-confirm-content">
                <p className="demo-confirm-note">{rejectionNote}</p>
                <div className="demo-confirm-actions">
                  <button
                    type="button"
                    className="demo-confirm-btn demo-confirm-cancel"
                    onClick={onDismissRejection}
                  >
                    ← Try a different prompt
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Generic error (network / non-parser) */}
          {errorMessage && (
            <div className="demo-bubble assistant">
              <span>{errorMessage}</span>
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
                    {sel.species.length > 0 && (
                      <span className="demo-rail-count"> · {sel.species.length}</span>
                    )}
                  </p>
                  <div className="demo-species-scroll">
                    <div className="demo-opt-grid">
                      {SPECIES.map((s) => (
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
              {PRESET_PROMPTS.map((s) => (
                <button
                  key={s}
                  type="button"
                  className={`demo-preset-chip${value === s ? " selected" : ""}`}
                  onClick={() => { setSel(EMPTY_SEL); setValue(s); inputRef.current?.focus(); }}
                >
                  {s}
                </button>
              ))}
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

