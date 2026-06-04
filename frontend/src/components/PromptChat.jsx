import { useEffect, useRef, useState } from "react";

const PRESET_PROMPTS = [
  "Rainy autumn dawn, distant thunder",
  "Still winter night, moonlit",
  "Hot summer afternoon, cicadas",
  "Spring morning, birdsong",
];

/* The demo entry interface: a calm, chatbot-style prompt screen (phase
   "prompt") that becomes a thinking transcript while the scene resolves
   (phase "generating"). Presentational — DemoPage owns the phase, the echoed
   user message, and the staged status line. */
export default function PromptChat({ phase, userMessage, statusLine, onSubmit }) {
  const [value, setValue] = useState("");
  const inputRef = useRef(null);
  const generating = phase === "generating";

  useEffect(() => {
    if (phase === "prompt") inputRef.current?.focus();
  }, [phase]);

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

  return (
    <div className={`demo-chat${generating ? " generating" : ""}`}>
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
        </div>

        {phase === "prompt" && (
          <>
            <div className="demo-presets">
              {PRESET_PROMPTS.map((s) => (
                <button
                  key={s}
                  type="button"
                  className="demo-preset-chip"
                  onClick={() => onSubmit(s)}
                >
                  {s}
                </button>
              ))}
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
