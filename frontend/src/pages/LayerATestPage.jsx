import { useState } from "react";
import { generateLayerA } from "../lib/api.js";

const DIEL_BINS = ["dawn", "morning", "afternoon", "night"];
const SEASONS   = ["spring", "summer", "autumn", "winter"];

const DURATION_MIN = 5;
const DURATION_MAX = 120;  // hard cap — 2 min keeps WAV/PNG payloads sane

const DEFAULT_PARAMS = {
  diel_bin: "dawn",
  season:   "spring",
  hour:     6,
  month:    9,
  duration: 60,
  k:        5,
};

export default function LayerATestPage() {
  const [params,   setParams]   = useState({ ...DEFAULT_PARAMS });
  const [status,   setStatus]   = useState("idle"); // idle | loading | done | error
  const [result,   setResult]   = useState(null);
  const [errorMsg, setErrorMsg] = useState("");

  function update(key, value) {
    setParams((p) => ({ ...p, [key]: value }));
  }

  async function handleRun() {
    setStatus("loading");
    setErrorMsg("");
    setResult(null);
    try {
      const duration = Math.max(
        DURATION_MIN,
        Math.min(DURATION_MAX, Number(params.duration) || DURATION_MIN),
      );
      const data = await generateLayerA({
        diel_bin: params.diel_bin,
        season:   params.season,
        hour:     Number(params.hour),
        month:    Number(params.month),
        duration,
        k:        Number(params.k),
      });
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }

  function downloadDataUrl(href, filename) {
    const a = document.createElement("a");
    a.href = href;
    a.download = filename;
    a.click();
  }

  function downloadJson() {
    if (!result?.metadata) return;
    const blob = new Blob([JSON.stringify(result.metadata, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    downloadDataUrl(url, `layer_a_explanation_${params.diel_bin}_${params.season}.json`);
    URL.revokeObjectURL(url);
  }

  const isLoading = status === "loading";
  const isDone    = status === "done";
  const tag       = `${params.diel_bin}_${params.season}_h${params.hour}_m${params.month}`;

  return (
    <section className="generation-page">
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">DEVELOPER TOOLS</p>
          <span>Layer A — Ambient Bed Test</span>
        </div>
      </header>

      <div className="generation-grid">
        {/* ── Left: input parameters ── */}
        <aside className="panel generation-sidebar-card">
          <div className="generation-card-head">
            <h2>Request Parameters</h2>
            <p>Environment for retrieval</p>
          </div>

          <div className="generation-sidebar-body" style={{ display: "grid", gap: 14 }}>
            <LabeledSelect
              label="Diel bin"
              value={params.diel_bin}
              options={DIEL_BINS}
              onChange={(v) => update("diel_bin", v)}
            />
            <LabeledSelect
              label="Season"
              value={params.season}
              options={SEASONS}
              onChange={(v) => update("season", v)}
            />
            <LabeledNumber
              label="Hour (0–23)"
              value={params.hour}
              min={0}
              max={23}
              onChange={(v) => update("hour", v)}
            />
            <LabeledNumber
              label="Month (1–12)"
              value={params.month}
              min={1}
              max={12}
              onChange={(v) => update("month", v)}
            />
            <LabeledNumber
              label={`Duration (s, ${DURATION_MIN}–${DURATION_MAX})`}
              value={params.duration}
              min={DURATION_MIN}
              max={DURATION_MAX}
              step={5}
              onChange={(v) => {
                const n = Number(v);
                if (Number.isNaN(n)) return update("duration", v);
                update("duration", Math.max(DURATION_MIN, Math.min(DURATION_MAX, n)));
              }}
            />
            <LabeledNumber
              label="Top-k blend"
              value={params.k}
              min={1}
              max={10}
              onChange={(v) => update("k", v)}
            />

            <button
              type="button"
              className="gen-primary-btn"
              onClick={handleRun}
              disabled={isLoading}
              style={{ marginTop: 4 }}
            >
              {isLoading ? "Retrieving…" : "▶ Run Layer A"}
            </button>

            {errorMsg && (
              <p className="analysis-error" style={{ marginTop: 8 }}>
                {errorMsg}
              </p>
            )}
          </div>
        </aside>

        {/* ── Centre: viewer ── */}
        <main className="panel generation-canvas-card">
          <div
            className="generation-canvas"
            style={{
              display: "block",
              padding: isDone ? 20 : 0,
              overflow: "auto",
            }}
          >
            {!isDone && !isLoading && (
              <div
                style={{
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  opacity: 0.6,
                }}
              >
                <p style={{ fontSize: 13, letterSpacing: 1 }}>
                  Set parameters and run Layer A to view outputs
                </p>
              </div>
            )}

            {isLoading && (
              <div className="gen-computing-overlay">
                <div className="gen-computing-ring" />
                <p>Retrieving + blending segments…</p>
              </div>
            )}

            {isDone && (
              <div style={{ display: "grid", gap: 18 }}>
                {/* Audio */}
                {result?.audio_b64 && (
                  <ReviewSection title="♪ Audio">
                    <audio
                      controls
                      src={`data:audio/wav;base64,${result.audio_b64}`}
                      style={{ width: "100%" }}
                    />
                    <p style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
                      {result.sample_rate} Hz · {result.duration_s.toFixed(1)} s · gain{" "}
                      {result.gain_db} dB
                    </p>
                  </ReviewSection>
                )}

                {/* Spectrogram */}
                {result?.image_b64 && (
                  <ReviewSection title="▤ Mel-Spectrogram">
                    <img
                      src={`data:image/png;base64,${result.image_b64}`}
                      alt="Layer A mel-spectrogram"
                      style={{
                        width: "100%",
                        height: "auto",
                        display: "block",
                        borderRadius: 6,
                      }}
                    />
                  </ReviewSection>
                )}

                {/* JSON metadata */}
                {result?.metadata && (
                  <ReviewSection title="{ } Metadata">
                    <pre className="layer-a-json">
                      {JSON.stringify(result.metadata, null, 2)}
                    </pre>
                  </ReviewSection>
                )}
              </div>
            )}
          </div>
        </main>

        {/* ── Right: outputs / downloads / summary ── */}
        <aside className="panel generation-output-card">
          <div className="generation-card-head">
            <h2>Outputs</h2>
            <p>Scrutinization triplet</p>
          </div>

          <div className="generation-output-body">
            <article className="gen-file-card">
              <div className="gen-file-head">
                <div className="gen-file-icon">▤</div>
                <div>
                  <span>Run tag</span>
                  <strong>{isDone ? tag.toUpperCase() : "—"}</strong>
                </div>
              </div>
            </article>

            {isDone && result?.metadata?.low_confidence && (
              <p className="mock-badge">
                Low confidence — diel bin was relaxed to find enough segments
              </p>
            )}

            {isDone && result?.metadata?.retrieved_clips && (
              <div className="gen-info-block">
                <p>Retrieved clips ({result.metadata.retrieved_clips.length})</p>
                <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, lineHeight: 1.6 }}>
                  {result.metadata.retrieved_clips.map((c, i) => (
                    <li key={c.clip_id}>
                      <code>{c.clip_id}</code> · sim {c.cosine_similarity} · w{" "}
                      {result.metadata.blend_weights[i]}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <button
              type="button"
              className="gen-secondary-btn"
              disabled={!isDone || !result?.audio_b64}
              onClick={() =>
                downloadDataUrl(
                  `data:audio/wav;base64,${result.audio_b64}`,
                  `layer_a_bed_${tag}.wav`,
                )
              }
            >
              ↓ Download WAV
            </button>

            <button
              type="button"
              className="gen-secondary-btn"
              disabled={!isDone || !result?.image_b64}
              onClick={() =>
                downloadDataUrl(
                  `data:image/png;base64,${result.image_b64}`,
                  `layer_a_spec_${tag}.png`,
                )
              }
            >
              ↓ Download Spectrogram (PNG)
            </button>

            <button
              type="button"
              className="gen-secondary-btn"
              disabled={!isDone}
              onClick={downloadJson}
            >
              ↓ Download Metadata (JSON)
            </button>
          </div>
        </aside>
      </div>
    </section>
  );
}

function LabeledSelect({ label, value, options, onChange }) {
  return (
    <label className="layer-a-field">
      <span>{label}</span>
      <select
        className="layer-a-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}

function LabeledNumber({ label, value, min, max, step = 1, onChange }) {
  return (
    <label className="layer-a-field">
      <span>{label}</span>
      <input
        className="layer-a-input"
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function ReviewSection({ title, children }) {
  return (
    <section>
      <h3
        style={{
          margin: "0 0 8px",
          fontSize: 12,
          letterSpacing: 1.2,
          textTransform: "uppercase",
          opacity: 0.75,
        }}
      >
        {title}
      </h3>
      {children}
    </section>
  );
}

