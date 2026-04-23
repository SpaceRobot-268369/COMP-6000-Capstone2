import { useRef, useState } from "react";
import AudioPlayer from "../components/AudioPlayer.jsx";
import EnvControls, { DEFAULT_CONDITIONS } from "../components/EnvControls.jsx";
import { transformSoundscape } from "../lib/api.js";

export default function TransformationPage() {
  const fileInputRef = useRef(null);

  const [file,       setFile]       = useState(null);
  const [sourceUrl,  setSourceUrl]  = useState(null);
  const [conditions, setConditions] = useState({ ...DEFAULT_CONDITIONS });
  const [status,     setStatus]     = useState("idle");
  const [result,     setResult]     = useState(null);
  const [errorMsg,   setErrorMsg]   = useState("");
  const [dragging,   setDragging]   = useState(false);
  const [activePlayer, setActivePlayer] = useState("source"); // "source" | "output"

  function acceptFile(f) {
    if (!f) return;
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    setFile(f);
    setSourceUrl(URL.createObjectURL(f));
    setResult(null);
    setStatus("idle");
    setErrorMsg("");
  }

  function onFileChange(e) {
    acceptFile(e.target.files?.[0] ?? null);
  }

  async function handleSynthesize() {
    if (!file) return;
    setStatus("synthesizing");
    setErrorMsg("");
    setResult(null);
    try {
      const data = await transformSoundscape(file, conditions);
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
    a.download = `transformed_${conditions.season}_${conditions.sample_bin}.wav`;
    a.click();
  }

  const isSynthesizing = status === "synthesizing";
  const isDone         = status === "done";
  const playerSrc      = activePlayer === "source" ? sourceUrl : result?.audioUrl ?? null;
  const playerLabel    = activePlayer === "source"
    ? (file?.name ?? "Source audio")
    : `Transformed · ${conditions.season} · ${conditions.sample_bin}`;

  return (
    <section className="transform-page">
      <header className="transform-topbar">
        <p className="eyebrow">SONIC LABORATORY</p>
        <div className="transform-tools">
          <label className="search-box transform-search">
            <span>⌕</span>
            <input type="text" placeholder="Search experiments…" />
          </label>
          <button type="button" className="icon-button" aria-label="Settings">⚙</button>
        </div>
      </header>

      <div className="transform-header-row">
        <div>
          <p className="eyebrow transform-section-label">Source Input</p>
          <h1>Original Audio</h1>
        </div>
        <div className="transform-center-heading">
          <p className="eyebrow transform-section-label">Processor Hub</p>
          <h2>Transformation Controls</h2>
        </div>
        <div className="transform-right-heading">
          <p className="eyebrow transform-section-label">Target Generation</p>
          <h2>Modified Output</h2>
        </div>
      </div>

      <div className="transform-grid">
        {/* ── Source column ── */}
        <section className="panel transform-column transform-source">
          <div
            className={`transform-wave-frame transform-drop-zone${dragging ? " drag-over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); acceptFile(e.dataTransfer.files?.[0]); }}
            onClick={() => fileInputRef.current?.click()}
            style={{ cursor: "pointer" }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.flac,.mp3,.webm"
              onChange={onFileChange}
              style={{ display: "none" }}
            />
            <div className="transform-waveform input" aria-hidden="true">
              <SourceWaveform file={file} />
            </div>
            {file ? (
              <div className="transform-badges">
                <span>{file.name.split(".").pop().toUpperCase()}</span>
                <span>{(file.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>
            ) : (
              <div className="transform-drop-hint">
                <span>⇪</span>
                <p>Drop or click to upload audio</p>
              </div>
            )}
          </div>

          {file && (
            <>
              <div className="transform-tag-head">
                <p>Source file</p>
              </div>
              <div className="transform-tags">
                <span className="transform-tag cyan"><i />{file.name}</span>
              </div>
            </>
          )}

          <div className="transform-source-player">
            <button
              type="button"
              className={`transform-mini-play${activePlayer === "source" ? " active-player" : ""}`}
              aria-label="Play source"
              onClick={() => setActivePlayer("source")}
              disabled={!sourceUrl}
              style={{ opacity: sourceUrl ? 1 : 0.4 }}
            >
              ▶
            </button>
            <div className="transform-mini-track">
              <span style={{ width: activePlayer === "source" ? "32%" : "0%" }} />
            </div>
            <p>{file ? "Source" : "No file"}</p>
          </div>
        </section>

        {/* ── Controls column ── */}
        <section className="panel transform-column transform-controls">
          <EnvControls value={conditions} onChange={setConditions} />

          {errorMsg && <p className="analysis-error">{errorMsg}</p>}

          <button
            type="button"
            className="transform-cta"
            onClick={handleSynthesize}
            disabled={!file || isSynthesizing}
            style={{ opacity: file ? 1 : 0.5 }}
          >
            {isSynthesizing ? "Synthesising…" : "✦ Synthesize"}
          </button>
        </section>

        {/* ── Output column ── */}
        <section className="panel transform-column transform-output">
          <div className="transform-wave-frame">
            {isDone && <div className="transform-live-pill">● Render complete</div>}
            <div className="transform-waveform output" aria-hidden="true">
              <OutputWaveform active={isDone} />
            </div>
          </div>

          {isDone && (
            <>
              <div className="transform-meta-head">
                <p>Output Metadata</p>
              </div>
              <div className="transform-meta-grid">
                <article className="transform-meta-card">
                  <span>Target season</span>
                  <strong>{conditions.season[0].toUpperCase() + conditions.season.slice(1)}</strong>
                </article>
                <article className="transform-meta-card">
                  <span>Time of day</span>
                  <strong>{conditions.sample_bin[0].toUpperCase() + conditions.sample_bin.slice(1)}</strong>
                </article>
                <article className="transform-meta-card">
                  <span>Temperature</span>
                  <strong>{conditions.temperature_c}°C</strong>
                </article>
                <article className="transform-meta-card">
                  <span>Wind</span>
                  <strong>{conditions.wind_speed_ms} m/s</strong>
                </article>
              </div>

              {result?.mock && (
                <p className="mock-badge" style={{ marginTop: 14 }}>
                  Demo mode — connect model server for real transformation
                </p>
              )}

              <div className="transform-output-footer">
                <div className="transform-output-actions">
                  <button
                    type="button"
                    className={`transform-mini-play${activePlayer === "output" ? " active-player" : ""}`}
                    aria-label="Play output"
                    onClick={() => setActivePlayer("output")}
                    style={{ width: 46, height: 46 }}
                  >
                    ▶
                  </button>
                  <button
                    type="button"
                    className="transform-icon-chip"
                    aria-label="Download output"
                    onClick={handleDownload}
                    disabled={!result?.audioUrl}
                  >
                    ↓
                  </button>
                </div>
                <div className="transform-file-size">
                  <span>Est. file size</span>
                  <strong>
                    {result?.audioUrl ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "—"}
                  </strong>
                </div>
              </div>
            </>
          )}

          {!isDone && !isSynthesizing && (
            <p className="transform-output-placeholder">
              Upload source audio and set target conditions, then click Synthesize.
            </p>
          )}

          {isSynthesizing && (
            <p className="transform-output-placeholder">Processing transformation…</p>
          )}
        </section>
      </div>

      <AudioPlayer
        src={playerSrc}
        label={playerLabel}
        detail={!playerSrc ? "No audio loaded" : undefined}
      />
    </section>
  );
}

// Static bars that animate when a file is loaded
function SourceWaveform({ file }) {
  const heights = [20, 42, 68, 36, 82, 28, 56, 74, 34, 62, 25, 50, 70, 30, 58, 76, 39, 61, 31, 47, 66, 41, 59, 24, 52, 72];
  return heights.map((h, i) => (
    <span
      key={i}
      style={{
        height: `${file ? h : 8}%`,
        transition: `height 0.5s ease ${i * 15}ms`,
      }}
    />
  ));
}

function OutputWaveform({ active }) {
  const heights = [18, 35, 61, 27, 73, 31, 44, 80, 38, 67, 22, 54, 86, 33, 49, 78, 43, 71, 29, 57, 83, 36, 64, 25, 46, 75];
  return heights.map((h, i) => (
    <span
      key={i}
      style={{
        height: `${active ? h : 8}%`,
        transition: `height 0.5s ease ${i * 20}ms`,
      }}
    />
  ));
}
