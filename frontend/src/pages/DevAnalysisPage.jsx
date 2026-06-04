import { useEffect, useMemo, useState } from "react";
import SpectrogramCanvas from "../components/SpectrogramCanvas.jsx";
import { analyseUpload, fetchLayerRegistry } from "../lib/api.js";

/**
 * Dev page for Analysis Mode (Layer E) — a per-head testing harness.
 *
 * This is NOT a product surface. Per .claude/context/ai/pipeline_design.md
 * § Analysis Mode, analysis runs three *independent* detector heads on the
 * raw mixture, each owning its own question and its own model(s):
 *
 *   E-A  Ambient context  — "what kind of bed is this?" (CLAP k-NN + probe)
 *   E-B  Weather          — "wind / rain intensity?"      (no model yet)
 *   E-C  Events           — "which species, when?"        (no model yet)
 *
 * So the page does NOT run one analysis for everything. Each head has:
 *   • its own model picker, filtered to attempts whose registry `head:`
 *     matches that head (an attempt belongs to exactly one head);
 *   • its own Analyze button that posts the upload to that one attempt
 *     (POST /api/layers/layer_e/attempts/{id}/analyze);
 *   • its own independent loading / result / error state.
 *
 * Heads with no real model (E-B, E-C today) render an explicit empty state —
 * no picker, no button — rather than pretending the ambient models apply.
 */

const LAYER_ID = "layer_e";

const HEADS = [
  {
    id: "ambient",
    code: "E-A",
    icon: "◫",
    label: "Ambient context",
    blurb: "Locate the clip in soundscape space — season / diel estimate plus nearest training clips.",
    loadingText: "Embedding clip + k-NN against the ambient pool…",
  },
  {
    id: "weather",
    code: "E-B",
    icon: "≋",
    label: "Weather",
    blurb: "Audible wind / rain intensity, detected directly on the raw mixture.",
    loadingText: "Scoring weather logits…",
  },
  {
    id: "events",
    code: "E-C",
    icon: "✦",
    label: "Events",
    blurb: "Species and acoustic events with onsets / offsets.",
    loadingText: "Running event detection…",
  },
];

const emptyHeadState = () => ({ attemptId: "", status: "idle", report: null, error: "" });

export default function DevAnalysisPage() {
  const [file,     setFile]     = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [dragging, setDragging] = useState(false);

  const [registry, setRegistry] = useState(null);
  const [regError, setRegError] = useState("");

  // Independent per-head runtime state (selected attempt + analysis result).
  const [heads, setHeads] = useState(() => ({
    ambient: emptyHeadState(),
    weather: emptyHeadState(),
    events:  emptyHeadState(),
  }));

  // Load the registry and pick a default attempt for each head.
  useEffect(() => {
    fetchLayerRegistry()
      .then((doc) => {
        setRegistry(doc);
        const layerE = (doc.layers || []).find((l) => l.id === LAYER_ID);
        const attempts = layerE?.attempts || [];
        const def = layerE?.default;
        setHeads((prev) => {
          const next = { ...prev };
          for (const h of HEADS) {
            const forHead = attempts.filter((a) => a.head === h.id);
            const chosen =
              forHead.find((a) => a.id === def)?.id || forHead[0]?.id || "";
            next[h.id] = { ...prev[h.id], attemptId: chosen };
          }
          return next;
        });
      })
      .catch((e) => setRegError(e.message));
  }, []);

  const attemptsByHead = useMemo(() => {
    const layerE = registry?.layers?.find((l) => l.id === LAYER_ID);
    const attempts = layerE?.attempts || [];
    const map = { ambient: [], weather: [], events: [] };
    for (const a of attempts) {
      if (map[a.head]) map[a.head].push(a);
    }
    return map;
  }, [registry]);

  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }, [audioUrl]);

  function acceptFile(f) {
    if (!f) return;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setFile(f);
    setAudioUrl(URL.createObjectURL(f));
    // New file → clear every head's previous result.
    setHeads((s) => {
      const next = { ...s };
      for (const h of HEADS) {
        next[h.id] = { ...s[h.id], status: "idle", report: null, error: "" };
      }
      return next;
    });
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    acceptFile(e.dataTransfer.files?.[0] ?? null);
  }

  function updateHead(headId, patch) {
    setHeads((s) => ({ ...s, [headId]: { ...s[headId], ...patch } }));
  }

  async function runHead(headId) {
    const { attemptId } = heads[headId];
    if (!file || !attemptId) return;
    updateHead(headId, { status: "analysing", error: "", report: null });
    try {
      const data = await analyseUpload(LAYER_ID, attemptId, file);
      updateHead(headId, { status: "done", report: data.report ?? data });
    } catch (err) {
      updateHead(headId, { status: "error", error: err.message });
    }
  }

  return (
    <section className="generation-page">
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">DEVELOPER TOOLS — ANALYSIS</p>
          <span>Per-head Layer E testing · E-A / E-B / E-C run independently</span>
        </div>
      </header>

      {/* ── Row 1: uploader (holds the clip only — no global run) ─────────── */}
      <div className="dev-controls-row">
        <FileUploader
          file={file}
          audioUrl={audioUrl}
          dragging={dragging}
          setDragging={setDragging}
          onFile={acceptFile}
          onDrop={onDrop}
        />
      </div>

      {/* ── Row 2: spectrogram + per-head analysis ──────────────────────── */}
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
            <h2>Analysis Heads</h2>
            <p>Each head picks its own model and runs on its own button</p>
          </div>
          <div className="dev-result-body">
            {HEADS.map((head) => (
              <HeadCard
                key={head.id}
                head={head}
                state={heads[head.id]}
                attempts={attemptsByHead[head.id]}
                regError={regError}
                hasFile={!!file}
                onAttemptChange={(id) => updateHead(head.id, { attemptId: id })}
                onRun={() => runHead(head.id)}
              />
            ))}
          </div>
        </aside>
      </div>
    </section>
  );
}

// ─── Uploader ────────────────────────────────────────────────────────────────

function FileUploader({ file, audioUrl, dragging, setDragging, onFile, onDrop }) {
  const inputId = "dev-analysis-file";
  return (
    <section
      className={`hero-upload panel panel-hero${dragging ? " drag-over" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        id={inputId}
        type="file"
        accept=".wav,.flac,.mp3,.ogg,.webm"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        style={{ display: "none" }}
      />

      {!file ? (
        <label htmlFor={inputId} style={{ cursor: "pointer", display: "block" }}>
          <div className="upload-icon">⇪</div>
          <h2>DROP AUDIO FOR ANALYSIS</h2>
          <p>WAV, FLAC, MP3 · click or drag to upload · then run each head below</p>
        </label>
      ) : (
        <div className="upload-loaded">
          <div className="upload-icon">◫</div>
          <div className="upload-file-info">
            <strong>{file.name}</strong>
            <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
          </div>
          <div className="upload-actions">
            <label htmlFor={inputId} className="upload-change-btn" style={{ cursor: "pointer" }}>
              Change file
            </label>
          </div>
          {audioUrl && (
            <audio controls src={audioUrl} style={{ width: "100%", marginTop: 12 }} />
          )}
        </div>
      )}
    </section>
  );
}

// ─── Per-head card ────────────────────────────────────────────────────────────

function HeadCard({ head, state, attempts, regError, hasFile, onAttemptChange, onRun }) {
  const hasModel = attempts.length > 0;
  const { attemptId, status, report, error } = state;
  const analysing = status === "analysing";
  const done = status === "done";
  const selected = attempts.find((a) => a.id === attemptId);

  return (
    <ReviewSection title={`${head.icon} ${head.code} — ${head.label}`}>
      <p className="dev-head-blurb" style={{ fontSize: 12, opacity: 0.7, margin: "0 0 8px" }}>
        {head.blurb}
      </p>

      {!hasModel ? (
        <EmptyHead label={head.label} regError={regError} />
      ) : (
        <>
          <AttemptPicker
            headCode={head.code}
            attemptId={attemptId}
            attempts={attempts}
            onChange={onAttemptChange}
          />

          <ModelDescription attempt={selected} />

          <div className="upload-actions" style={{ marginBottom: 8 }}>
            <button
              type="button"
              className="analyse-btn"
              onClick={onRun}
              disabled={!hasFile || !attemptId || analysing}
            >
              {analysing ? "Analysing…" : `✦ Run ${head.code}`}
            </button>
            {!hasFile && (
              <span style={{ fontSize: 12, opacity: 0.6, alignSelf: "center" }}>
                Upload a clip first
              </span>
            )}
          </div>

          {error && <p className="analysis-error">{error}</p>}

          {done && report ? (
            <HeadResult headId={head.id} report={report} />
          ) : (
            <Placeholder kind="json" loading={analysing}>
              {analysing ? head.loadingText : "Run this head to see results."}
            </Placeholder>
          )}

          {done && report && (
            <details style={{ marginTop: 8 }}>
              <summary style={{ cursor: "pointer", fontSize: 12, opacity: 0.7 }}>
                {"{ } Raw report"}
              </summary>
              <pre className="layer-a-json">{JSON.stringify(report, null, 2)}</pre>
            </details>
          )}
        </>
      )}
    </ReviewSection>
  );
}

// "About this model" panel — documents the selected attempt straight from the
// registry `description`. Scoped to lucas's models for now (other authors show
// a muted not-yet-documented note); split this gate out once teammates add
// documented Layer E attempts.
function ModelDescription({ attempt }) {
  if (!attempt) return null;

  const isLucas = attempt.author === "lucas";
  const text = (attempt.description || "").trim();

  return (
    <div
      className="dev-model-doc"
      style={{
        margin: "0 0 10px",
        padding: "10px 12px",
        borderRadius: 8,
        background: "rgba(127,127,127,0.08)",
        border: "1px solid rgba(127,127,127,0.18)",
      }}
    >
      <p style={{ margin: "0 0 6px", fontSize: 11, letterSpacing: "0.06em", textTransform: "uppercase", opacity: 0.6 }}>
        About this model · {attempt.stage} · {attempt.status}
      </p>
      {isLucas && text ? (
        text.split(/\n{2,}/).map((para, i) => (
          <p key={i} style={{ margin: i === 0 ? 0 : "8px 0 0", fontSize: 13, lineHeight: 1.5, opacity: 0.85 }}>
            {para}
          </p>
        ))
      ) : (
        <p style={{ margin: 0, fontSize: 13, opacity: 0.6, fontStyle: "italic" }}>
          {isLucas
            ? "No description provided for this model yet."
            : `Not yet documented (author: ${attempt.author || "unknown"}).`}
        </p>
      )}
    </div>
  );
}

function EmptyHead({ label, regError }) {
  return (
    <div className="dev-placeholder dev-placeholder-json">
      <p className="dev-placeholder-caption">
        {regError
          ? `Registry unavailable: ${regError}`
          : `No ${label.toLowerCase()} model on the AI service yet — nothing to test on this head.`}
      </p>
    </div>
  );
}

// ─── Per-head result renderers ───────────────────────────────────────────────

function HeadResult({ headId, report }) {
  if (headId === "ambient") return <AmbientResult report={report} />;
  if (headId === "weather") return <WeatherResult report={report} />;
  if (headId === "events") return <EventsResult report={report} />;
  return (
    <pre className="layer-a-json">{JSON.stringify(report, null, 2)}</pre>
  );
}

function AmbientResult({ report }) {
  const cond = report?.estimated_conditions;
  const sims = report?.similar_clips || [];
  return (
    <div className="dev-controls-meta">
      <div className="gen-info-block">
        <p>Estimated context</p>
        <code>
          {cond ? [cond.diel_bin, cond.season].filter(Boolean).join(" · ") || "—" : "—"}
        </code>
      </div>
      <div className="gen-info-block">
        <p>Hour · Month</p>
        <code>{cond ? `${fmtNum(cond.hour)} · ${fmtNum(cond.month)}` : "—"}</code>
      </div>
      <div className="gen-info-block">
        <p>Season conf.</p>
        <code>{fmtPct(report?.season_confidence)}</code>
      </div>
      <div className="gen-info-block">
        <p>OOD</p>
        <code>{report?.ood_flag ? "flagged" : "no"}</code>
      </div>
      <ConfidenceBar value={report?.confidence} label="Neighbour similarity" />
      {sims.length > 0 && (
        <ul className="dev-sim-list" style={{ gridColumn: "1 / -1", margin: "8px 0 0", paddingLeft: 16 }}>
          {sims.slice(0, 5).map((c, i) => (
            <li key={c.segment_id || i} style={{ fontSize: 12, opacity: 0.8 }}>
              <code>{c.segment_id}</code>
              {typeof c.similarity === "number" && <> · sim {c.similarity.toFixed(3)}</>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function WeatherResult({ report }) {
  const weather = report?.observations?.weather || {};
  const legacyWeather = report?.weather || {};
  const derivedLabel = weather.derived_label || legacyWeather.overall_label || "—";
  const warnings = weather.warnings || legacyWeather.warnings || [];
  return (
    <div className="dev-controls-meta">
      <div className="gen-info-block">
        <p>Derived label</p>
        <code>{derivedLabel}</code>
      </div>
      <div className="gen-info-block">
        <p>Confidence</p>
        <code>{fmtNum(weather.confidence)}</code>
      </div>
      {["rain", "wind", "thunder"].map((element) => (
        <WeatherElementBlock
          key={element}
          element={element}
          summary={weather?.[element]?.summary}
        />
      ))}
      {warnings.length > 0 && (
        <div className="gen-info-block" style={{ gridColumn: "1 / -1" }}>
          <p>Warnings</p>
          <code>{warnings.join(" · ")}</code>
        </div>
      )}
    </div>
  );
}

function WeatherElementBlock({ element, summary }) {
  const label = summary?.label || "none";
  const intensity = typeof summary?.intensity === "number" ? summary.intensity : null;
  const confidence = typeof summary?.confidence === "number" ? summary.confidence : null;
  const coverage = typeof summary?.coverage === "number" ? summary.coverage : null;
  return (
    <div className="gen-info-block">
      <p>{element}</p>
      <code>
        {label}
        {intensity == null ? "" : ` · ${intensity.toFixed(3)}`}
        {confidence == null ? "" : ` · conf ${confidence.toFixed(3)}`}
        {coverage == null ? "" : ` · cov ${coverage.toFixed(2)}`}
      </code>
    </div>
  );
}

// ─── Shared bits ──────────────────────────────────────────────────────────────

function EventsResult({ report }) {
  const events = Array.isArray(report?.events) ? report.events : [];
  const species = Array.isArray(report?.known_species) ? report.known_species : [];
  return (
    <div className="dev-controls-meta">
      <div className="gen-info-block">
        <p>Detected events</p>
        <code>{report?.num_events ?? events.length}</code>
      </div>
      <div className="gen-info-block">
        <p>Windows hit</p>
        <code>{report?.num_detected_windows ?? 0} / {report?.num_windows ?? 0}</code>
      </div>
      <div className="gen-info-block">
        <p>Known species</p>
        <code>{species.length}</code>
      </div>
      <div className="gen-info-block">
        <p>Threshold</p>
        <code>{fmtNum(report?.threshold)}</code>
      </div>

      {events.length === 0 ? (
        <div className="dev-placeholder dev-placeholder-json" style={{ gridColumn: "1 / -1" }}>
          <p className="dev-placeholder-caption">
            No species event passed the current threshold.
          </p>
        </div>
      ) : (
        <div className="event-result-list">
          <div className="event-timeline-head">
            <p>Detected species timeline</p>
            <span>{events.length} time segments</span>
          </div>
          {events.map((event, idx) => (
            <article className="event-result-item" key={`${event.label}-${event.onset_s}-${idx}`}>
              <div className="event-result-main">
                <span>{fmtNum(event.onset_s)}s - {fmtNum(event.offset_s)}s</span>
                <strong>{formatSpeciesLabel(event.label)}</strong>
              </div>
              <div className="event-result-stats">
                <span>Mean {fmtPct(event.confidence_mean)}</span>
                <span>Max {fmtPct(event.confidence_max)}</span>
                <span>{event.window_count ?? 0} windows</span>
              </div>
              <ConfidenceBar value={event.confidence_mean} label="Event confidence" />
              <SpeciesMatchList matches={event.species_matches} />
              <PhenologyBlock phenology={event.phenology} />
            </article>
          ))}
        </div>
      )}
      <AnalysisReportSummary report={report?.analysis_report} />
      <DetectionTimeline windows={report?.diagnostics?.detected_windows} />
    </div>
  );
}

function AnalysisReportSummary({ report }) {
  if (!report) return null;
  const observations = Array.isArray(report.observations) ? report.observations : [];
  const inferred = Array.isArray(report.inferred_context) ? report.inferred_context : [];
  const disagreements = Array.isArray(report.disagreements) ? report.disagreements : [];
  return (
    <div className="analysis-report-summary">
      <div className="analysis-report-head">
        <p>Report-ready summary</p>
        <span>
          {observations.length} observations · {inferred.length} inferences · {disagreements.length} disagreements
        </span>
      </div>
      {inferred.length > 0 && (
        <div className="analysis-report-chips">
          {inferred.slice(0, 6).map((item, idx) => (
            <span key={`${item.type}-${idx}`}>
              {formatSignal(item.type)}: {formatSignal(item.value)} · {fmtPct(item.confidence)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function PhenologyBlock({ phenology }) {
  if (!phenology) return null;
  return (
    <div className="phenology-block">
      <p>Ecological signal</p>
      <div className="phenology-grid">
        <span>Common name</span>
        <strong>{phenology.common_name || "Unknown"}</strong>
        <span>Scientific name</span>
        <strong>{phenology.scientific_name || "Unknown"}</strong>
        <span>Active time</span>
        <strong>{formatSignal(phenology.diel_signal)} · {fmtPct(phenology.diel_confidence)}</strong>
        <span>Season signal</span>
        <strong>{formatSignal(phenology.season_signal)} · {fmtPct(phenology.season_confidence)}</strong>
        <span>Habitat</span>
        <strong>{formatSignal(phenology.habitat_signal)}</strong>
      </div>
      {phenology.inference_notes && <em>{phenology.inference_notes}</em>}
    </div>
  );
}

function SpeciesMatchList({ matches }) {
  const rows = Array.isArray(matches) ? matches.slice(0, 5) : [];
  if (rows.length === 0) return null;
  return (
    <div className="species-match-list">
      <p>Species match</p>
      {rows.map((row) => {
        const score = typeof row.score === "number" ? Math.max(0, Math.min(1, row.score)) : 0;
        return (
          <div className="species-match-row" key={row.label}>
            <span>{formatSpeciesLabel(row.label)}</span>
            <div className="species-match-track" aria-hidden="true">
              <i style={{ width: `${Math.max(3, Math.round(score * 100))}%` }} />
            </div>
            <code>{fmtPct(score)}</code>
          </div>
        );
      })}
    </div>
  );
}

function DetectionTimeline({ windows }) {
  const rawRows = Array.isArray(windows) ? windows : [];
  const rows = mergeDetectionWindows(rawRows);
  if (rows.length === 0) return null;
  return (
    <div className="detection-timeline">
      <div className="detection-timeline-head">
        <p>Supporting window segments</p>
        <span>{rawRows.length} windows merged into {rows.length} segments</span>
      </div>
      <div className="detection-window-list">
        {rows.map((row) => (
          <div className="detection-window-row" key={`${row.label}-${row.start_s}-${row.end_s}`}>
            <code>{fmtNum(row.start_s)}s - {fmtNum(row.end_s)}s</code>
            <span>{formatSpeciesLabel(row.label)}</span>
            <div className="detection-window-score">
              <i style={{ width: `${Math.max(3, Math.round(clamp01(row.confidence_mean) * 100))}%` }} />
            </div>
            <strong>{fmtPct(row.confidence_mean)}</strong>
            <em>Max {fmtPct(row.confidence_max)} · {row.window_count} windows</em>
          </div>
        ))}
      </div>
    </div>
  );
}

function mergeDetectionWindows(windows) {
  const sorted = windows
    .filter((row) => row && typeof row.start_s === "number" && typeof row.end_s === "number")
    .slice()
    .sort((a, b) => (a.start_s - b.start_s) || String(a.top_label).localeCompare(String(b.top_label)));
  const segments = [];
  for (const row of sorted) {
    const label = String(row.top_label || "unknown");
    const confidence = typeof row.confidence === "number" ? row.confidence : 0;
    const last = segments[segments.length - 1];
    if (last && last.label === label && row.start_s <= last.end_s + 1.01) {
      last.end_s = Math.max(last.end_s, row.end_s);
      last.confidence_sum += confidence;
      last.confidence_max = Math.max(last.confidence_max, confidence);
      last.window_count += 1;
    } else {
      segments.push({
        label,
        start_s: row.start_s,
        end_s: row.end_s,
        confidence_sum: confidence,
        confidence_max: confidence,
        window_count: 1,
      });
    }
  }
  return segments.map((segment) => ({
    label: segment.label,
    start_s: segment.start_s,
    end_s: segment.end_s,
    confidence_mean: segment.confidence_sum / segment.window_count,
    confidence_max: segment.confidence_max,
    window_count: segment.window_count,
  }));
}

function AttemptPicker({ headCode, attemptId, attempts, onChange }) {
  return (
    <div className="dev-controls-meta" style={{ marginBottom: 8 }}>
      <label className="layer-a-field" style={{ gridColumn: "1 / -1" }}>
        <span>{headCode} model</span>
        <select
          className="layer-a-input"
          value={attemptId}
          onChange={(e) => onChange?.(e.target.value)}
        >
          {attempts.map((a) => (
            <option key={a.id} value={a.id}>
              {a.label} ({a.stage}, {a.status})
              {a.available === false ? " — unavailable" : ""}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

function ConfidenceBar({ value, label = "Confidence" }) {
  const hasValue = typeof value === "number";
  const clamped = hasValue ? Math.max(0, Math.min(1, value)) : null;
  const pct = clamped == null ? null : Math.round(clamped * 100);
  return (
    <div className="gen-info-block" style={{ gridColumn: "1 / -1" }}>
      <p>{label}</p>
      {clamped == null ? (
        <code>—</code>
      ) : (
        <div className="dev-score-gauge">
          <div className="dev-score-track"
               role="meter"
               aria-valuemin="0" aria-valuemax="1" aria-valuenow={value}
               aria-label={label}>
            <i style={{ width: `${Math.max(4, pct)}%` }} />
            <span className="dev-score-tick" style={{ left: "25%" }} />
            <span className="dev-score-tick" style={{ left: "50%" }} />
            <span className="dev-score-tick" style={{ left: "75%" }} />
          </div>
          <code className="dev-score-readout">{value.toFixed(3)} / 1.00</code>
        </div>
      )}
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

function formatSpeciesLabel(value) {
  if (!value) return "Unknown species";
  return String(value)
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatSignal(value) {
  if (!value) return "Unknown";
  return String(value).replaceAll("_", " ");
}

function fmtNum(v) {
  return typeof v === "number" ? v.toFixed(2) : "—";
}

function fmtPct(v) {
  if (typeof v !== "number") return "—";
  return `${Math.round(v * 100)}%`;
}

function clamp01(value) {
  return typeof value === "number" ? Math.max(0, Math.min(1, value)) : 0;
}
