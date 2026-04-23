import { useState } from "react";
import AudioPlayer from "../components/AudioPlayer.jsx";
import EnvControls, { DEFAULT_CONDITIONS } from "../components/EnvControls.jsx";
import { generateSoundscape } from "../lib/api.js";

export default function GenerationPage() {
  const [conditions, setConditions] = useState({ ...DEFAULT_CONDITIONS });
  const [status,     setStatus]     = useState("idle");    // idle | generating | done | error
  const [result,     setResult]     = useState(null);
  const [errorMsg,   setErrorMsg]   = useState("");

  async function handleGenerate() {
    setStatus("generating");
    setErrorMsg("");
    setResult(null);
    try {
      const data = await generateSoundscape(conditions);
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }

  function handleDownload() {
    if (!result?.audioUrl) return;
    const a = document.createElement("a");
    a.href = result.audioUrl;
    a.download = `soundscape_${conditions.season}_${conditions.sample_bin}.wav`;
    a.click();
  }

  const isGenerating = status === "generating";
  const isDone       = status === "done";

  return (
    <section className="generation-page">
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">SONIC LABORATORY</p>
          <span>Speculative Generation Pipeline</span>
        </div>
      </header>

      <div className="generation-grid">
        {/* ── Left: condition encoding ── */}
        <aside className="panel generation-sidebar-card">
          <div className="generation-card-head">
            <h2>Condition Encoding</h2>
            <p>Environmental variables</p>
          </div>
          <div className="generation-sidebar-body">
            <EnvControls value={conditions} onChange={setConditions} />

            <div className="gen-info-block">
              <p>Site information</p>
              <div className="gen-site-card">
                <div>
                  <span>Location</span>
                  <strong>Bowra, QLD — Site 257</strong>
                </div>
                <div>
                  <span>Coordinates</span>
                  <strong>28.3°S 145.5°E</strong>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* ── Centre: canvas / status ── */}
        <main className="panel generation-canvas-card">
          <div className="generation-canvas">
            <div className="gen-wave gen-wave-back" />
            <div className="gen-wave gen-wave-mid" />
            <div className="gen-wave gen-wave-front" />

            {isGenerating && (
              <div className="gen-computing-overlay">
                <div className="gen-computing-ring" />
                <p>Synthesising latent space…</p>
              </div>
            )}

            {isDone && result?.mock && (
              <span className="gen-node gen-node-one">● Demo mode active</span>
            )}
            {isDone && !result?.mock && (
              <span className="gen-node gen-node-two">● Model inference complete</span>
            )}
          </div>

          <div className="generation-caption">
            <h1>Latent Soundscape Synthesis</h1>
            <div className="generation-caption-row">
              <i />
              <p>
                {isGenerating
                  ? "Processing conditions…"
                  : isDone
                    ? `${conditions.season} · ${conditions.sample_bin} · ${conditions.temperature_c}°C`
                    : "Set conditions and generate"}
              </p>
            </div>
            {errorMsg && <p className="analysis-error" style={{ marginTop: 12 }}>{errorMsg}</p>}
          </div>
        </main>

        {/* ── Right: output ── */}
        <aside className="panel generation-output-card">
          <div className="generation-card-head">
            <h2>Output Reconstruction</h2>
            <p>Generated artifact</p>
          </div>
          <div className="generation-output-body">
            {/* Progress / idle state */}
            <div className="gen-progress-block">
              <span>{isGenerating ? "Progress" : isDone ? "Complete" : "Ready"}</span>
              <div className="gen-progress-line">
                <strong>{isGenerating ? "…" : isDone ? "100%" : "—"}</strong>
                <p>{isGenerating ? "Reconstructing…" : isDone ? "Done" : "Awaiting input"}</p>
              </div>
              <div className="gen-progress-track">
                <i style={{
                  right: isDone ? "0%" : isGenerating ? "40%" : "100%",
                  transition: "right 2s ease",
                }} />
              </div>
            </div>

            {/* Output file card */}
            <article className="gen-file-card">
              <div className="gen-file-head">
                <div className="gen-file-icon">▤</div>
                <div>
                  <span>Output file</span>
                  <strong>
                    {isDone
                      ? `${conditions.season.toUpperCase()}_${conditions.sample_bin.toUpperCase()}`
                      : "—"}
                  </strong>
                </div>
              </div>
              <OutputBars active={isDone} />
            </article>

            {isDone && result?.mock && (
              <p className="mock-badge">Demo mode — connect model server for real audio</p>
            )}

            <button
              type="button"
              className="gen-primary-btn"
              onClick={handleGenerate}
              disabled={isGenerating}
            >
              {isGenerating ? "Generating…" : "▣ Generate Soundscape"}
            </button>

            {isDone && result?.audioUrl && (
              <button type="button" className="gen-secondary-btn" onClick={handleDownload}>
                ↓ Download WAV
              </button>
            )}
            {isDone && !result?.audioUrl && (
              <button type="button" className="gen-secondary-btn" disabled>
                Model not connected
              </button>
            )}
          </div>
        </aside>
      </div>

      <AudioPlayer
        src={result?.audioUrl ?? null}
        label={isDone ? `${conditions.season} · ${conditions.sample_bin}` : "Generation Output"}
        detail={!isDone ? "Generate a soundscape to play it here" : undefined}
      />
    </section>
  );
}

function OutputBars({ active }) {
  const heights = [22, 44, 68, 31, 83, 53, 18, 40, 66, 75, 29, 55];
  return (
    <div className="gen-render-bars" aria-hidden="true">
      {heights.map((h, i) => (
        <i
          key={i}
          style={{
            height: `${active ? h : 4}px`,
            transition: `height 0.4s ease ${i * 30}ms`,
          }}
        />
      ))}
    </div>
  );
}
