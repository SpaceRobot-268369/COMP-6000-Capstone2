import { useEffect, useMemo, useState } from "react";
import {
  fetchAttemptSamples,
  fetchLayerRegistry,
  generateAttempt,
  sampleWavUrl,
} from "../lib/api.js";

const DEFAULT_SEED = 42;

export default function LayerATestPage() {
  // Registry state
  const [registry, setRegistry] = useState(null);
  const [regError, setRegError] = useState("");
  const [layerId,   setLayerId]   = useState("");
  const [attemptId, setAttemptId] = useState("");

  // Generation state
  const [seed,     setSeed]     = useState(DEFAULT_SEED);
  const [status,   setStatus]   = useState("idle");   // idle | loading | done | error
  const [result,   setResult]   = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [progress, setProgress] = useState(0);

  // Cached samples (loaded whenever the attempt changes)
  const [samples,    setSamples]    = useState(null);  // {reference:[…], showcase:[…], canonical_seed}
  const [samplesErr, setSamplesErr] = useState("");

  // Load the layer registry once on mount.
  useEffect(() => {
    fetchLayerRegistry()
      .then((doc) => {
        setRegistry(doc);
        const firstLayer = doc.layers?.[0];
        if (firstLayer) {
          setLayerId(firstLayer.id);
          setAttemptId(firstLayer.default || firstLayer.attempts?.[0]?.id || "");
        }
      })
      .catch((e) => setRegError(e.message));
  }, []);

  // When the layer changes, snap the attempt to that layer's default.
  useEffect(() => {
    if (!registry || !layerId) return;
    const layer = registry.layers.find((l) => l.id === layerId);
    if (!layer) return;
    if (!layer.attempts.some((a) => a.id === attemptId)) {
      setAttemptId(layer.default || layer.attempts[0]?.id || "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layerId, registry]);

  // Whenever (layer, attempt) changes, refresh the cached-samples panel and
  // sync the seed input to that attempt's canonical seed.
  useEffect(() => {
    if (!layerId || !attemptId) return;
    setSamples(null);
    setSamplesErr("");
    fetchAttemptSamples(layerId, attemptId)
      .then((doc) => {
        setSamples(doc);
        if (Number.isInteger(doc?.canonical_seed)) setSeed(doc.canonical_seed);
      })
      .catch((e) => setSamplesErr(e.message));
  }, [layerId, attemptId]);

  const currentLayer = useMemo(
    () => registry?.layers.find((l) => l.id === layerId),
    [registry, layerId],
  );
  const currentAttempt = useMemo(
    () => currentLayer?.attempts.find((a) => a.id === attemptId),
    [currentLayer, attemptId],
  );

  // Fake progress ticker.
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

  async function handleRun() {
    if (!layerId || !attemptId) return;
    setStatus("loading");
    setErrorMsg("");
    setResult(null);
    try {
      const data = await generateAttempt(layerId, attemptId, {
        seed: Number(seed) || DEFAULT_SEED,
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
    downloadDataUrl(url, `${tag}_metadata.json`);
    URL.revokeObjectURL(url);
  }

  const isLoading = status === "loading";
  const isDone    = status === "done";
  const tag       = `${layerId}__${attemptId}__seed${seed || DEFAULT_SEED}`;
  const progressText = getProgressText(progress, status);

  if (regError) {
    return (
      <section className="generation-page">
        <header className="generation-topbar">
          <div className="generation-brandline">
            <p className="eyebrow">DEVELOPER TOOLS</p>
            <span>Layer / Attempt Dev Test</span>
          </div>
        </header>
        <p className="analysis-error">Failed to load layer registry: {regError}</p>
      </section>
    );
  }

  if (!registry) {
    return (
      <section className="generation-page">
        <header className="generation-topbar">
          <div className="generation-brandline">
            <p className="eyebrow">DEVELOPER TOOLS</p>
            <span>Layer / Attempt Dev Test</span>
          </div>
        </header>
        <p>Loading registry…</p>
      </section>
    );
  }

  return (
    <section className="generation-page">
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">DEVELOPER TOOLS</p>
          <span>Layer / Attempt Dev Test</span>
        </div>
      </header>

      <div className="generation-grid layer-a-grid">
        {/* ── Left: input parameters ── */}
        <aside className="panel generation-sidebar-card">
          <div className="generation-card-head">
            <h2>{currentLayer?.label || layerId}</h2>
            <p>{currentAttempt?.label || attemptId}</p>
          </div>

          <div className="generation-sidebar-body" style={{ display: "grid", gap: 14 }}>
            <LabeledSelect
              label="Layer"
              value={layerId}
              onChange={setLayerId}
              options={registry.layers.map((l) => ({ value: l.id, label: l.label }))}
            />

            <LabeledSelect
              label="Attempt"
              value={attemptId}
              onChange={setAttemptId}
              options={(currentLayer?.attempts || []).map((a) => ({
                value: a.id,
                label: `${a.label}  (${a.stage}, ${a.status})`,
              }))}
            />

            <div className="gen-info-block">
              <p>Attempt ID</p>
              <code>{attemptId}</code>
            </div>

            <LabeledNumber
              label="Seed"
              value={seed}
              min={0}
              max={2147483647}
              hint="Seed controls the random starting noise. Same seed + same attempt = same audio."
              onChange={setSeed}
            />

            <button
              type="button"
              className="gen-primary-btn"
              onClick={handleRun}
              disabled={isLoading || !attemptId}
              style={{ marginTop: 4 }}
            >
              {isLoading ? "Generating..." : "Generate"}
            </button>

            {(isLoading || isDone) && (
              <ProgressBlock progress={progress} label={progressText} />
            )}

            {errorMsg && (
              <p className="analysis-error" style={{ marginTop: 8 }}>{errorMsg}</p>
            )}
          </div>
        </aside>

        {/* ── Centre: viewer ── */}
        <main className="panel generation-canvas-card">
          <div className="generation-canvas"
               style={{ display: "block", padding: 20, overflow: "auto" }}>

            {/* Cached samples (no model load) */}
            <CachedSamples
              layerId={layerId}
              attemptId={attemptId}
              samples={samples}
              error={samplesErr}
            />

            {!isDone && !isLoading && !samples?.reference?.length && !samples?.showcase?.length && (
              <div style={{ padding: "24px 0", opacity: 0.6 }}>
                <p style={{ fontSize: 13 }}>
                  No cached samples on disk yet. Click Generate to run the handler.
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
              <div className="layer-a-result-stack">
                {result?.audio_b64 && (
                  <ReviewSection title="♪ Audio">
                    <audio controls
                           src={`data:audio/wav;base64,${result.audio_b64}`}
                           style={{ width: "100%" }} />
                    <p style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
                      {result.sample_rate} Hz · {result.duration_s?.toFixed?.(1) ?? result.duration_s} s · seed{" "}
                      {result.metadata?.seed ?? seed}
                    </p>
                  </ReviewSection>
                )}

                {result?.image_b64 && (
                  <ReviewSection title="▤ Mel-Spectrogram">
                    <img src={`data:image/png;base64,${result.image_b64}`}
                         alt="Mel-spectrogram"
                         className="gen-spectrogram-img layer-a-spectrogram-img" />
                  </ReviewSection>
                )}

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

        {/* ── Right: outputs ── */}
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
                  <strong>{isDone ? tag : "-"}</strong>
                </div>
              </div>
            </article>

            {isDone && result?.metadata?.prompt_locked && (
              <p className="mock-badge">
                Fixed prompt active — user prompts disabled
              </p>
            )}

            {isDone && result?.metadata?.audio && (
              <div className="gen-info-block">
                <p>Audio stats</p>
                <code>
                  RMS {result.metadata.audio.rms?.toFixed?.(4)} · peak{" "}
                  {result.metadata.audio.peak?.toFixed?.(4)}
                </code>
              </div>
            )}

            <button type="button" className="gen-secondary-btn"
                    disabled={!isDone || !result?.audio_b64}
                    onClick={() => downloadDataUrl(
                      `data:audio/wav;base64,${result.audio_b64}`,
                      `${tag}.wav`,
                    )}>
              ↓ Download WAV
            </button>

            <button type="button" className="gen-secondary-btn"
                    disabled={!isDone || !result?.image_b64}
                    onClick={() => downloadDataUrl(
                      `data:image/png;base64,${result.image_b64}`,
                      `${tag}_spectrogram.png`,
                    )}>
              ↓ Download Spectrogram (PNG)
            </button>

            <button type="button" className="gen-secondary-btn"
                    disabled={!isDone}
                    onClick={downloadJson}>
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
      <input className="layer-a-input" type="number"
             value={value} min={min} max={max} step={step}
             onChange={(e) => onChange(e.target.value)} />
      {hint && <small>{hint}</small>}
    </label>
  );
}

function LabeledSelect({ label, value, options, onChange }) {
  return (
    <label className="layer-a-field">
      <span>{label}</span>
      <select className="layer-a-input"
              value={value}
              onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

function getProgressText(progress, status) {
  if (status === "done") return "Complete";
  if (progress < 18) return "Preparing";
  if (progress < 38) return "Loading model / LoRA";
  if (progress < 72) return "Generating audio";
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
      <div className="gen-progress-track"
           role="progressbar"
           aria-valuemin="0" aria-valuemax="100"
           aria-valuenow={Math.round(progress)}
           aria-label="Generation progress">
        <i style={{ width: `${Math.max(4, Math.min(100, progress))}%` }} />
      </div>
    </div>
  );
}

function CachedSamples({ layerId, attemptId, samples, error }) {
  if (error) {
    return (
      <section style={{ marginBottom: 16 }}>
        <h3 style={cachedHeaderStyle}>⚡ Cached samples</h3>
        <p className="analysis-error">{error}</p>
      </section>
    );
  }
  if (!samples) return null;

  const groups = [
    { tier: "reference", entries: samples.reference || [] },
    { tier: "showcase",  entries: samples.showcase  || [] },
  ].filter((g) => g.entries.length > 0);

  if (!groups.length) return null;

  return (
    <section style={{ marginBottom: 24 }}>
      <h3 style={cachedHeaderStyle}>
        ⚡ Cached samples
        <small style={{ marginLeft: 8, opacity: 0.6, fontSize: 11 }}>
          canonical seed {samples.canonical_seed} · no model load
        </small>
      </h3>

      {groups.map((g) => (
        <div key={g.tier} style={{ marginTop: 12 }}>
          <p style={{ margin: "0 0 6px", fontSize: 11, opacity: 0.7,
                      textTransform: "uppercase", letterSpacing: 0.4 }}>
            {g.tier}
          </p>
          <div style={{ display: "grid", gap: 12 }}>
            {g.entries.map((s) => (
              <SampleCard
                key={`${g.tier}/${s.stem}`}
                tier={g.tier}
                sample={s}
                wavSrc={s.has_wav ? sampleWavUrl(layerId, attemptId, g.tier, s.stem) : null}
              />
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}

const cachedHeaderStyle = {
  margin: "0 0 8px",
  fontSize: 12,
  letterSpacing: 0,
  textTransform: "uppercase",
  opacity: 0.85,
};

function SampleCard({ tier, sample, wavSrc }) {
  return (
    <article style={{
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 8, padding: 12, display: "grid", gap: 8,
    }}>
      <p style={{ margin: 0, fontSize: 12, opacity: 0.85 }}>
        <code>{sample.stem}</code>
        {!sample.png_b64 && sample.has_png && (
          <small style={{ marginLeft: 8, opacity: 0.6 }}>
            (PNG is DVC-tracked, run <code>dvc pull</code> to render)
          </small>
        )}
      </p>

      {sample.png_b64 && (
        <img
          src={`data:image/png;base64,${sample.png_b64}`}
          alt={`${tier} sample ${sample.stem}`}
          style={{ width: "100%", maxHeight: 220, objectFit: "contain",
                   background: "#000", borderRadius: 4 }}
        />
      )}

      {wavSrc && (
        <audio controls src={wavSrc} style={{ width: "100%" }}>
          Your browser doesn't support audio playback.
        </audio>
      )}
      {!sample.has_wav && (
        <small style={{ opacity: 0.6 }}>WAV not present locally — generate or `dvc pull`.</small>
      )}

      {sample.metadata && (
        <details>
          <summary style={{ fontSize: 11, opacity: 0.7, cursor: "pointer" }}>
            metadata ({sample.metadata.handler_git_sha ?? "no git sha"})
          </summary>
          <pre style={{ fontSize: 10, marginTop: 6, maxHeight: 200, overflow: "auto" }}>
            {JSON.stringify(sample.metadata, null, 2)}
          </pre>
        </details>
      )}
    </article>
  );
}

function ReviewSection({ title, children }) {
  return (
    <section>
      <h3 style={{ margin: "0 0 8px", fontSize: 12, letterSpacing: 0,
                   textTransform: "uppercase", opacity: 0.75 }}>
        {title}
      </h3>
      {children}
    </section>
  );
}
