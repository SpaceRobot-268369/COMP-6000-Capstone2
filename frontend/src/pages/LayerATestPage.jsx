import { useEffect, useState } from "react";
import { generateLayerA } from "../lib/api.js";

const DEFAULT_PARAMS = {
  seed: 42,
};

const FIXED_LAYER_A_PROMPT =
  "quiet spring night ambient soundscape, Bowra dry woodland, Australia, distant environmental bed, no foreground events, no music, no machinery";

const CURRENT_CHECKPOINT = "audioldm2-lora-raw-smoke";

export default function LayerATestPage() {
  const [params,   setParams]   = useState({ ...DEFAULT_PARAMS });
  const [status,   setStatus]   = useState("idle"); // idle | loading | done | error
  const [result,   setResult]   = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (status === "loading") {
      const startedAt = Date.now();
      setProgress(6);
      const timer = window.setInterval(() => {
        const elapsedS = (Date.now() - startedAt) / 1000;
        const next = Math.min(92, 6 + (1 - Math.exp(-elapsedS / 18)) * 88);
        setProgress((current) => Math.max(current, Math.round(next)));
      }, 500);
      return () => window.clearInterval(timer);
    }

    if (status === "done") {
      setProgress(100);
      return undefined;
    }

    setProgress(0);
    return undefined;
  }, [status]);

  function update(key, value) {
    setParams((p) => ({ ...p, [key]: value }));
  }

  async function handleRun() {
    setStatus("loading");
    setErrorMsg("");
    setResult(null);
    try {
      const data = await generateLayerA({
        seed: Number(params.seed) || 42,
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
    downloadDataUrl(url, `layer_a_audioldm2_seed${params.seed}_metadata.json`);
    URL.revokeObjectURL(url);
  }

  const isLoading = status === "loading";
  const isDone    = status === "done";
  const tag       = `audioldm2_seed${params.seed || 42}`;
  const progressText = getProgressText(progress, status);

  return (
    <section className="generation-page">
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">DEVELOPER TOOLS</p>
          <span>Layer A — Ambient AI Test</span>
        </div>
      </header>

      <div className="generation-grid layer-a-grid">
        {/* ── Left: input parameters ── */}
        <aside className="panel generation-sidebar-card">
          <div className="generation-card-head">
            <h2>Smoke Model</h2>
            <p>Fixed AudioLDM2 Layer A run</p>
          </div>

          <div className="generation-sidebar-body" style={{ display: "grid", gap: 14 }}>
            <div className="gen-info-block">
              <p>Checkpoint</p>
              <code>{CURRENT_CHECKPOINT}</code>
            </div>
            <div className="gen-info-block">
              <p>Fixed prompt</p>
              <code>{FIXED_LAYER_A_PROMPT}</code>
            </div>
            <LabeledNumber
              label="Seed"
              value={params.seed}
              min={0}
              max={2147483647}
              hint="Seed controls the random starting noise. Use any whole number from 0 to 2,147,483,647; same seed repeats the same variation with the same model settings."
              onChange={(v) => update("seed", v)}
            />

            <button
              type="button"
              className="gen-primary-btn"
              onClick={handleRun}
              disabled={isLoading}
              style={{ marginTop: 4 }}
            >
              {isLoading ? "Generating..." : "Generate Layer A"}
            </button>

            {(isLoading || isDone) && (
              <ProgressBlock progress={progress} label={progressText} />
            )}

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
                <p style={{ fontSize: 13, letterSpacing: 0 }}>
                  Generate the fixed Layer A smoke model to view outputs
                </p>
              </div>
            )}

            {isLoading && (
              <div className="gen-computing-overlay">
                <div className="layer-a-processing">
                  <div className="gen-computing-ring" />
                  <ProgressBlock progress={progress} label={progressText} />
                </div>
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
                      {result.sample_rate} Hz · {result.duration_s.toFixed(1)} s · seed{" "}
                      {result.metadata?.seed ?? params.seed}
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
            <p>WAV, mel-spectrogram, metadata</p>
          </div>

          <div className="generation-output-body">
            <article className="gen-file-card">
              <div className="gen-file-head">
                <div className="gen-file-icon">▤</div>
                <div>
                  <span>Run tag</span>
                  <strong>{isDone ? tag.toUpperCase() : "-"}</strong>
                </div>
              </div>
            </article>

            {isDone && result?.metadata?.prompt_locked && (
              <p className="mock-badge">
                Fixed prompt active - user prompts disabled
              </p>
            )}

            {isDone && result?.metadata?.audio && (
              <div className="gen-info-block">
                <p>Audio stats</p>
                <code>
                  RMS {result.metadata.audio.rms.toFixed(4)} · peak{" "}
                  {result.metadata.audio.peak.toFixed(4)}
                </code>
              </div>
            )}

            <button
              type="button"
              className="gen-secondary-btn"
              disabled={!isDone || !result?.audio_b64}
              onClick={() =>
                downloadDataUrl(
                  `data:audio/wav;base64,${result.audio_b64}`,
                  `layer_a_${tag}.wav`,
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
                  `layer_a_${tag}_spectrogram.png`,
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

function LabeledNumber({ label, value, min, max, step = 1, hint, onChange }) {
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
      {hint && <small>{hint}</small>}
    </label>
  );
}

function getProgressText(progress, status) {
  if (status === "done") return "Complete";
  if (progress < 18) return "Preparing fixed prompt";
  if (progress < 38) return "Loading smoke LoRA";
  if (progress < 72) return "Denoising ambient bed";
  if (progress < 94) return "Rendering WAV and spectrogram";
  return "Finalizing";
}

function ProgressBlock({ progress, label }) {
  return (
    <div className="gen-progress-block layer-a-progress" aria-live="polite">
      <div className="gen-progress-line">
        <strong>{Math.round(progress)}%</strong>
        <p>{label}</p>
      </div>
      <div
        className="gen-progress-track"
        role="progressbar"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={Math.round(progress)}
        aria-label="Layer A generation progress"
      >
        <i style={{ width: `${Math.max(4, Math.min(100, progress))}%` }} />
      </div>
    </div>
  );
}

function ReviewSection({ title, children }) {
  return (
    <section>
      <h3
        style={{
          margin: "0 0 8px",
          fontSize: 12,
          letterSpacing: 0,
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
