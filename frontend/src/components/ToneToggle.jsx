import { useRef, useState } from "react";
import { narrateReport } from "../lib/api.js";
import "./ToneToggle.css";

/* Top-center tone toggle for the immersive scene (plan §3.5). Switches the
   analysis report narrative between Immersive and Analytical registers by
   re-rendering it through the LLM-OSS report writer (/api/analysis/narrative)
   — no detectors re-run, so toggling is cheap.

   Renders nothing unless a fused `report` is supplied. The current generation
   flow has no analysis report on this page; route one into the immersive page
   state (e.g. location.state.resolved.report) to activate the toggle. */

const REGISTERS = [
  { id: "immersive", label: "Immersive" },
  { id: "analytical", label: "Analytical" },
];

export default function ToneToggle({ report, defaultRegister = "immersive", initialText = "" }) {
  const [register, setRegister] = useState(defaultRegister);
  const [text, setText] = useState(initialText);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Cache rendered text per register so re-toggling is instant (no re-call).
  const cache = useRef(initialText ? { [defaultRegister]: initialText } : {});

  async function select(next) {
    if (next === register || loading) return;
    setRegister(next);
    setError("");
    if (cache.current[next] !== undefined) {
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

  if (!report) return null;

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
