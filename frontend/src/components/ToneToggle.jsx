import { useRef, useState } from "react";
import { narrateReport } from "../lib/api.js";
import "./ToneToggle.css";

/* Top-center tone toggle for the immersive scene (plan §3.5). Switches the
   analysis report narrative between Immersive and Analytical registers.

   Both registers are rendered through the LLM-OSS report writer
   (/api/analysis/narrative) *upfront* — at analysis/generation time, before
   this page mounts — and handed in via `narratives`. Switching is then a pure
   cache read with no LLM call. `report` is kept only as a lazy fallback: if a
   register was not pre-rendered (e.g. its upfront call failed), the first
   selection renders it on demand.

   Renders nothing unless there is a pre-rendered narrative or a `report` to
   render from. Generated scenes supply a synthesized report — see
   lib/generationReport.js. */

const REGISTERS = [
  { id: "immersive", label: "Immersive" },
  { id: "analytical", label: "Analytical" },
];

export default function ToneToggle({ report, narratives = {}, defaultRegister = "immersive" }) {
  const [register, setRegister] = useState(defaultRegister);
  // Seed the cache with every pre-rendered register so toggling never re-calls.
  const cache = useRef({ ...narratives });
  const [text, setText] = useState(cache.current[defaultRegister] || "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function select(next) {
    if (next === register || loading) return;
    setRegister(next);
    setError("");
    if (cache.current[next]) {
      setText(cache.current[next]);
      return;
    }
    if (!report) return;
    setLoading(true);
    try {
      const data = await narrateReport(report, next);
      const rendered = data?.narrative?.text || "";
      cache.current[next] = rendered;
      setText(rendered);
    } catch (e) {
      setError(e?.message || "Could not re-render the narrative.");
    } finally {
      setLoading(false);
    }
  }

  const hasAnyNarrative = Object.values(cache.current).some(Boolean);
  if (!report && !hasAnyNarrative) return null;

  return (
    <div className="tone-toggle">
      <div className="tone-toggle-switch" role="group" aria-label="Narration tone">
        {REGISTERS.map((r) => (
          <button
            key={r.id}
            type="button"
            className={`tone-toggle-btn ${register === r.id ? "is-active" : ""}`}
            aria-pressed={register === r.id}
            disabled={loading}
            onClick={() => select(r.id)}
          >
            {r.label}
          </button>
        ))}
      </div>
      {(text || loading || error) && (
        <p className={`tone-toggle-narrative ${error ? "is-error" : ""}`} aria-live="polite">
          {loading ? "Re-rendering…" : error || text}
        </p>
      )}
    </div>
  );
}
