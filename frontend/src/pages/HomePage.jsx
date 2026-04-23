import { useEffect, useRef, useState } from "react";
import AudioPlayer from "../components/AudioPlayer.jsx";
import SpectrogramCanvas from "../components/SpectrogramCanvas.jsx";
import { analyseAudio } from "../lib/api.js";

const apiBase = import.meta.env.VITE_API_URL || "";

export default function HomePage() {
  const fileInputRef = useRef(null);

  const [health,    setHealth]    = useState({ loading: true });
  const [file,      setFile]      = useState(null);       // uploaded File
  const [audioUrl,  setAudioUrl]  = useState(null);       // blob URL for player
  const [status,    setStatus]    = useState("idle");     // idle | analysing | done | error
  const [result,    setResult]    = useState(null);       // analysis result object
  const [errorMsg,  setErrorMsg]  = useState("");
  const [dragging,  setDragging]  = useState(false);

  // Health check
  useEffect(() => {
    fetch(`${apiBase}/api/health`)
      .then((r) => r.json())
      .then((d) => setHealth(d))
      .catch((e) => setHealth({ ok: false, message: String(e) }));
  }, []);

  const healthText = health.loading
    ? "Scanning node telemetry…"
    : health.ok
      ? `Operational${health.message ? ` • ${health.message}` : ""}`
      : `Attention required${health.message ? ` • ${health.message}` : ""}`;

  // ── File handling ────────────────────────────────────────────────────────────

  function acceptFile(f) {
    if (!f) return;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setFile(f);
    setAudioUrl(URL.createObjectURL(f));
    setResult(null);
    setStatus("idle");
    setErrorMsg("");
  }

  function onFileChange(e) {
    acceptFile(e.target.files?.[0] ?? null);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    acceptFile(e.dataTransfer.files?.[0] ?? null);
  }

  async function runAnalysis() {
    if (!file) return;
    setStatus("analysing");
    setErrorMsg("");
    try {
      const data = await analyseAudio(file);
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }

  // ── Derived display values ───────────────────────────────────────────────────

  const metrics = result
    ? [
        { label: "Avian biophony",    value: Math.round(result.avian_biophony    * 100), tone: "cyan" },
        { label: "Anthropogenic noise", value: Math.round(result.anthropogenic   * 100), tone: "rose" },
        { label: "Insect geophony",   value: Math.round(result.insect_geophony   * 100), tone: "mint" },
      ]
    : [
        { label: "Avian biophony",    value: 0, tone: "cyan" },
        { label: "Anthropogenic noise", value: 0, tone: "rose" },
        { label: "Insect geophony",   value: 0, tone: "mint" },
      ];

  return (
    <section className="dashboard-page">
      <header className="topbar">
        <div>
          <p className="eyebrow">SONIC LABORATORY</p>
          <h1>ANALYSIS PIPELINE</h1>
          <div className="status-line">
            <span className="status-accent" />
            <p>System status: {healthText} / {apiBase || "Local node"}</p>
          </div>
        </div>
        <div className="topbar-tools">
          <label className="search-box">
            <span>⌕</span>
            <input type="text" placeholder="Search parameters…" />
          </label>
          <button type="button" className="icon-button" aria-label="Settings">⚙</button>
        </div>
      </header>

      <div className="content-grid">
        {/* ── Upload / analyse panel ── */}
        <section
          className={`hero-upload panel panel-hero${dragging ? " drag-over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => !file && fileInputRef.current?.click()}
          style={{ cursor: file ? "default" : "pointer" }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.flac,.mp3,.webm"
            onChange={onFileChange}
            style={{ display: "none" }}
          />

          {!file ? (
            <>
              <div className="upload-icon">⇪</div>
              <h2>DROP AUDIO FOR DEEP ANALYSIS</h2>
              <p>FLAC, WAV, MP3 • click or drag to upload</p>
            </>
          ) : (
            <div className="upload-loaded">
              <div className="upload-icon">◫</div>
              <div className="upload-file-info">
                <strong>{file.name}</strong>
                <span>{(file.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>
              <div className="upload-actions">
                <button
                  type="button"
                  className="analyse-btn"
                  onClick={runAnalysis}
                  disabled={status === "analysing"}
                >
                  {status === "analysing" ? "Analysing…" : "✦ Run Analysis"}
                </button>
                <button
                  type="button"
                  className="upload-change-btn"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Change file
                </button>
              </div>
              {result?.mock && (
                <p className="mock-badge">Demo mode — connect model server for real results</p>
              )}
              {errorMsg && (
                <p className="analysis-error">{errorMsg}</p>
              )}
            </div>
          )}
        </section>

        {/* ── Spectrogram ── */}
        <section className="panel spectral-panel">
          <div className="panel-heading">
            <h3>Spectral Mapping</h3>
            <p>{file ? file.name : "No file loaded"}</p>
          </div>
          <SpectrogramCanvas file={file} />
          <WaveBars result={result} />
        </section>

        {/* ── Metrics ── */}
        <section className="panel metrics-panel">
          <div className="panel-heading">
            <h3>Species Activity Indicators</h3>
            <p>{status === "done" ? "Live reading" : "—"}</p>
          </div>
          <div className="metric-list">
            {metrics.map((m) => (
              <div key={m.label} className="metric-row">
                <div className="metric-label-row">
                  <span>{m.label}</span>
                  <span>{m.value}%</span>
                </div>
                <div className="metric-track">
                  <span
                    className={`metric-fill metric-${m.tone}`}
                    style={{ width: `${m.value}%`, transition: "width 0.8s ease" }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Stat cards ── */}
        <section className="stats-split">
          <article className="panel stat-card">
            <p className="stat-label">Acoustic Diversity</p>
            <strong>{result ? result.acoustic_diversity.toFixed(3) : "—"}</strong>
            <span>{result ? "Analysis complete" : "Upload audio to analyse"}</span>
          </article>
          <article className="panel stat-card">
            <p className="stat-label">Soundscape Structure</p>
            <strong>{result ? result.soundscape_complexity : "—"}</strong>
            <span>{result ? `${result.layers_detected} layers detected` : "Awaiting input"}</span>
          </article>
        </section>

        {/* ── Summary ── */}
        <section className="panel summary-panel">
          <div className="panel-heading">
            <h3>Neural Summary</h3>
            <p>{result ? `Peak: ${result.peak_db.toFixed(1)} dBFS` : "—"}</p>
          </div>
          <p className="summary-text">
            {result
              ? `Dominant frequency ${result.dominant_freq_hz.toLocaleString()} Hz. ` +
                `Acoustic diversity index ${result.acoustic_diversity.toFixed(3)} — ` +
                `${result.soundscape_complexity.toLowerCase()} structural signature ` +
                `with ${result.layers_detected} distinct layers identified.`
              : "Upload an audio file and run analysis to generate a neural summary."}
          </p>
        </section>
      </div>

      <AudioPlayer
        src={audioUrl}
        label={file ? file.name : "Media Controller"}
        detail={file ? undefined : "No file loaded"}
      />
    </section>
  );
}

// Animated wave bars that respond to analysis results
function WaveBars({ result }) {
  const base = [10, 24, 44, 62, 51, 18, 37, 55, 29, 8, 40, 59, 32, 15, 48, 24, 11, 31, 57, 63, 54, 19, 29, 55, 36, 7, 24];
  const bars = result
    ? base.map((h) => Math.round(h * (0.6 + result.acoustic_diversity * 0.7)))
    : base;

  return (
    <div className="wave-bars" aria-hidden="true">
      {bars.map((height, i) => (
        <span
          key={i}
          className="wave-bar"
          style={{ height: `${height}px`, transition: "height 0.6s ease" }}
        />
      ))}
    </div>
  );
}
