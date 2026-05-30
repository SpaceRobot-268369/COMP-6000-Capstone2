import { useEffect, useRef, useState } from "react";
import SpectrogramCanvas from "../components/SpectrogramCanvas.jsx";
import { analyseUpload } from "../lib/api.js";

/**
 * Dev page for Analysis Mode (Layer E).
 *
 * Three components, matching .claude/context/ai/pipeline_design.md
 * § Analysis Mode:
 *
 *   1. Audio uploader (drag/drop or click).
 *   2. Mel-spectrogram preview of the uploaded clip (client-side render).
 *   3. Analysis results panel — three heads:
 *        E-A  ambient context  (similarity, estimated conditions)
 *        E-B  weather          (wind / rain intensity)
 *        E-C  events           (species detections + onsets)
 *      Each head shows label(s) plus a confidence score.
 *
 * The backend endpoint (POST /api/analysis) is still being wired up; the
 * page surfaces a clear error if the request fails so the UI is testable
 * end-to-end as soon as the server is ready.
 */

export default function DevAnalysisPage() {
  const fileInputRef = useRef(null);

  const [file,      setFile]      = useState(null);
  const [audioUrl,  setAudioUrl]  = useState(null);
  const [status,    setStatus]    = useState("idle"); // idle | analysing | done | error
  const [report,    setReport]    = useState(null);
  const [errorMsg,  setErrorMsg]  = useState("");
  const [dragging,  setDragging]  = useState(false);
  const [progress,  setProgress]  = useState(0);

  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }, [audioUrl]);

  // Fake progress while analysing.
  useEffect(() => {
    if (status !== "analysing") {
      setProgress(status === "done" ? 100 : 0);
      return undefined;
    }
    const startedAt = Date.now();
    setProgress(6);
    const timer = window.setInterval(() => {
      const elapsedS = (Date.now() - startedAt) / 1000;
      const next = Math.min(92, 6 + (1 - Math.exp(-elapsedS / 12)) * 88);
      setProgress((cur) => Math.max(cur, Math.round(next)));
    }, 400);
    return () => window.clearInterval(timer);
  }, [status]);

  function acceptFile(f) {
    if (!f) return;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setFile(f);
    setAudioUrl(URL.createObjectURL(f));
    setReport(null);
    setErrorMsg("");
    setStatus("idle");
  }

  function onFileChange(e) { acceptFile(e.target.files?.[0] ?? null); }
  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    acceptFile(e.dataTransfer.files?.[0] ?? null);
  }

  async function runAnalysis() {
    if (!file) return;
    setStatus("analysing");
    setErrorMsg("");
    setReport(null);
    try {
      const data = await analyseUpload(file);
      setReport(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }

  const isAnalysing = status === "analysing";
  const isDone      = status === "done";

  return (
    <section className="generation-page">
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">DEVELOPER TOOLS — ANALYSIS</p>
          <span>Upload · Spectrogram · E-A / E-B / E-C report</span>
        </div>
      </header>

      {/* ── Row 1: uploader (full width) ─────────────────────────────────── */}
      <div className="dev-controls-row">
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
            accept=".wav,.flac,.mp3,.ogg,.webm"
            onChange={onFileChange}
            style={{ display: "none" }}
          />

          {!file ? (
            <>
              <div className="upload-icon">⇪</div>
              <h2>DROP AUDIO FOR ANALYSIS</h2>
              <p>WAV, FLAC, MP3 · click or drag to upload</p>
            </>
          ) : (
            <div className="upload-loaded">
              <div className="upload-icon">◫</div>
              <div className="upload-file-info">
                <strong>{file.name}</strong>
                <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>

              <div className="upload-actions">
                <button
                  type="button"
                  className="analyse-btn"
                  onClick={runAnalysis}
                  disabled={isAnalysing}
                >
                  {isAnalysing ? "Analysing…" : "✦ Run Analysis"}
                </button>
                <button
                  type="button"
                  className="upload-change-btn"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Change file
                </button>
              </div>

              {audioUrl && (
                <audio
                  controls
                  src={audioUrl}
                  style={{ width: "100%", marginTop: 12 }}
                />
              )}

              {errorMsg && <p className="analysis-error">{errorMsg}</p>}
            </div>
          )}
        </section>
      </div>

      {/* ── Row 2: spectrogram + analysis ───────────────────────────────── */}
      <div className="dev-results-row">
        <main className="panel dev-result-card">
          <div className="generation-card-head">
            <h2>Mel-Spectrogram</h2>
            <p>{file ? file.name : "Upload a clip to render its spectrogram"}</p>
          </div>

          <div className="dev-result-body">
            <ReviewSection title="▤ Spectral Mapping">
              {file ? (
                <SpectrogramCanvas file={file} />
              ) : (
                <Placeholder kind="image">
                  Spectrogram appears once an audio file is loaded.
                </Placeholder>
              )}
            </ReviewSection>
          </div>
        </main>

        <aside className="panel dev-result-card">
          <div className="generation-card-head">
            <h2>Analysis Results</h2>
            <p>
              {isDone
                ? "Per-head detections + confidence"
                : isAnalysing
                  ? "Running E-A / E-B / E-C in parallel…"
                  : "Three detector heads run on the raw mixture"}
            </p>
          </div>

          <div className="dev-result-body">
            {isAnalysing && <ProgressBlock progress={progress} />}

            <AmbientHead report={report} loading={isAnalysing} done={isDone} />
            <WeatherHead report={report} loading={isAnalysing} done={isDone} />
            <EventsHead  report={report} loading={isAnalysing} done={isDone} />

            <ReviewSection title="{ } Raw report">
              {isDone && report ? (
                <pre className="layer-a-json">
                  {JSON.stringify(report, null, 2)}
                </pre>
              ) : (
                <Placeholder kind="json" loading={isAnalysing}>
                  {isAnalysing
                    ? "Collecting per-head outputs…"
                    : "Raw report appears after analysis."}
                </Placeholder>
              )}
            </ReviewSection>
          </div>
        </aside>
      </div>
    </section>
  );
}

// ─── Per-head sections ──────────────────────────────────────────────────────

function AmbientHead({ report, loading, done }) {
  const a = report?.ambient;
  const cond = a?.estimated_conditions;
  const sims = a?.similar_clips || [];
  return (
    <ReviewSection title="◫ E-A — Ambient context">
      {done && a ? (
        <div className="dev-controls-meta">
          <div className="gen-info-block">
            <p>Estimated context</p>
            <code>
              {cond
                ? [cond.diel_bin, cond.season].filter(Boolean).join(" · ") || "—"
                : "—"}
            </code>
          </div>
          <div className="gen-info-block">
            <p>Similar clips</p>
            <code>{sims.length ? `${sims.length} hits` : "—"}</code>
          </div>
          <ConfidenceBar value={a.confidence} />
          {sims.length > 0 && (
            <ul className="dev-sim-list" style={{ margin: "8px 0 0", paddingLeft: 16 }}>
              {sims.slice(0, 5).map((c, i) => (
                <li key={c.segment_id || i} style={{ fontSize: 12, opacity: 0.8 }}>
                  <code>{c.segment_id}</code>
                  {typeof c.similarity === "number" && (
                    <> · sim {c.similarity.toFixed(3)}</>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <Placeholder kind="json" loading={loading}>
          {loading
            ? "Embedding clip + k-NN against ambient_index.csv…"
            : "Awaiting analysis."}
        </Placeholder>
      )}
    </ReviewSection>
  );
}

function WeatherHead({ report, loading, done }) {
  const w = report?.weather;
  return (
    <ReviewSection title="≋ E-B — Weather">
      {done && w ? (
        <div className="dev-controls-meta">
          <div className="gen-info-block">
            <p>Wind intensity</p>
            <code>{w.wind_intensity || "—"}</code>
          </div>
          <div className="gen-info-block">
            <p>Rain intensity</p>
            <code>{w.rain_intensity || "—"}</code>
          </div>
          <ConfidenceBar value={w.confidence} />
        </div>
      ) : (
        <Placeholder kind="json" loading={loading}>
          {loading
            ? "Scoring PANNs / CLAP weather logits…"
            : "Awaiting analysis."}
        </Placeholder>
      )}
    </ReviewSection>
  );
}

function EventsHead({ report, loading, done }) {
  const e = report?.events;
  const dets = e?.detections || [];
  return (
    <ReviewSection title="✦ E-C — Events">
      {done && e ? (
        <div className="dev-controls-meta">
          <div className="gen-info-block">
            <p>Detections</p>
            <code>{dets.length ? `${dets.length} events` : "—"}</code>
          </div>
          <ConfidenceBar value={e.confidence} />
          {dets.length > 0 && (
            <table className="dev-event-table"
                   style={{ width: "100%", marginTop: 8, fontSize: 12,
                            borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left", opacity: 0.7 }}>
                  <th>Label</th><th>Onset</th><th>Offset</th><th>Conf.</th>
                </tr>
              </thead>
              <tbody>
                {dets.slice(0, 20).map((d, i) => (
                  <tr key={i}>
                    <td><code>{d.label}</code></td>
                    <td>{fmtTime(d.onset_s)}</td>
                    <td>{fmtTime(d.offset_s)}</td>
                    <td>{fmtPct(d.confidence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ) : (
        <Placeholder kind="json" loading={loading}>
          {loading
            ? "Running BirdNET + zero-shot fallbacks on raw mixture…"
            : "Awaiting analysis."}
        </Placeholder>
      )}
    </ReviewSection>
  );
}

// ─── Small helpers ──────────────────────────────────────────────────────────

function ConfidenceBar({ value }) {
  const pct = typeof value === "number"
    ? Math.max(0, Math.min(100, Math.round(value * 100)))
    : null;
  return (
    <div className="gen-info-block" style={{ gridColumn: "1 / -1" }}>
      <p>Confidence</p>
      {pct == null ? (
        <code>—</code>
      ) : (
        <div className="gen-progress-track"
             role="progressbar"
             aria-valuemin="0" aria-valuemax="100" aria-valuenow={pct}
             aria-label="Head confidence"
             style={{ marginTop: 4 }}>
          <i style={{ width: `${Math.max(4, pct)}%` }} />
          <span style={{ marginLeft: 8, fontSize: 12 }}>{pct}%</span>
        </div>
      )}
    </div>
  );
}

function ProgressBlock({ progress }) {
  return (
    <div className="gen-progress-block layer-a-progress" aria-live="polite">
      <div className="gen-progress-line">
        <strong>{Math.round(progress)}%</strong>
        <p>Analysing…</p>
      </div>
      <div className="gen-progress-track"
           role="progressbar"
           aria-valuemin="0" aria-valuemax="100"
           aria-valuenow={Math.round(progress)}
           aria-label="Analysis progress">
        <i style={{ width: `${Math.max(4, Math.min(100, progress))}%` }} />
      </div>
    </div>
  );
}

function ReviewSection({ title, children }) {
  return (
    <section className="dev-review-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Placeholder({ kind, loading, children }) {
  return (
    <div className={`dev-placeholder dev-placeholder-${kind}${loading ? " is-loading" : ""}`}>
      <div className="dev-placeholder-art" aria-hidden="true">
        {kind === "image" && <div className="dev-placeholder-image" />}
        {kind === "json" && (
          <div className="dev-placeholder-json">
            <span style={{ width: "30%" }} />
            <span style={{ width: "65%" }} />
            <span style={{ width: "50%" }} />
            <span style={{ width: "72%" }} />
            <span style={{ width: "40%" }} />
          </div>
        )}
      </div>
      <p className="dev-placeholder-caption">{children}</p>
    </div>
  );
}

function fmtTime(s) {
  if (typeof s !== "number") return "—";
  return `${s.toFixed(2)} s`;
}

function fmtPct(v) {
  if (typeof v !== "number") return "—";
  return `${Math.round(v * 100)}%`;
}
