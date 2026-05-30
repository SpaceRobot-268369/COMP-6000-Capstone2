import { useEffect, useMemo, useState } from "react";
import {
  fetchAttemptSamples,
  fetchLayerRegistry,
  generateAttempt,
  sampleWavUrl,
} from "../lib/api.js";

const DEFAULT_SEED = 42;

const GENERATION_LAYER_IDS = ["layer_a", "layer_b", "layer_c", "layer_d"];
const ANALYSIS_LAYER_IDS   = ["layer_e"];

export default function LayerATestPage({
  mode = "generation",
  includeLayers,
  eyebrow = "DEVELOPER TOOLS",
  title = "Layer / Attempt Dev Test",
} = {}) {
  const allowedLayerIds = useMemo(() => {
    if (includeLayers && includeLayers.length) return includeLayers;
    return mode === "analysis" ? ANALYSIS_LAYER_IDS : GENERATION_LAYER_IDS;
  }, [mode, includeLayers]);

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
  const [samples,    setSamples]    = useState(null);  // {expected:[…], showcase:[…], canonical_seed}
  const [samplesErr, setSamplesErr] = useState("");
  const [expectedKey, setExpectedKey] = useState(""); // "<tier>/<stem>"

  // Load the layer registry once on mount.
  useEffect(() => {
    fetchLayerRegistry()
      .then((doc) => {
        const filtered = {
          ...doc,
          layers: (doc.layers || []).filter((l) => allowedLayerIds.includes(l.id)),
        };
        setRegistry(filtered);
        const firstLayer = filtered.layers?.[0];
        if (firstLayer) {
          setLayerId(firstLayer.id);
          setAttemptId(firstLayer.default || firstLayer.attempts?.[0]?.id || "");
        }
      })
      .catch((e) => setRegError(e.message));
  }, [allowedLayerIds]);

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
    setExpectedKey("");
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
  const usesSeed = currentAttempt?.uses_seed === true;

  // Flatten cached samples in display order (expected first, then showcase).
  const expectedEntries = useMemo(() => {
    if (!samples) return [];
    const tiers = [
      { tier: "expected", entries: samples.expected || [] },
      { tier: "showcase", entries: samples.showcase || [] },
    ];
    return tiers.flatMap((t) =>
      t.entries.map((s) => ({ tier: t.tier, sample: s, key: `${t.tier}/${s.stem}` })),
    );
  }, [samples]);

  // Auto-select the first expected sample when entries change.
  useEffect(() => {
    if (!expectedEntries.length) {
      setExpectedKey("");
      return;
    }
    if (!expectedEntries.some((e) => e.key === expectedKey)) {
      setExpectedKey(expectedEntries[0].key);
    }
  }, [expectedEntries, expectedKey]);

  const expectedSelected = useMemo(
    () => expectedEntries.find((e) => e.key === expectedKey) || null,
    [expectedEntries, expectedKey],
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
      const data = await generateAttempt(
        layerId,
        attemptId,
        usesSeed ? { seed: Number(seed) || DEFAULT_SEED } : {},
      );
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

  const registryReady = Boolean(registry);
  const layerOptions = registryReady
    ? registry.layers.map((l) => ({ value: l.id, label: l.label }))
    : allowedLayerIds.map((id) => ({ value: id, label: id }));
  const attemptOptions = (currentLayer?.attempts || []).map((a) => ({
    value: a.id,
    label: `${a.label}  (${a.stage}, ${a.status})${a.available === false ? " — unavailable" : ""}`,
  }));
  const registryBanner = regError
    ? `Failed to load layer registry: ${regError}. Controls shown for preview only — AI server unreachable.`
    : !registryReady
      ? "Loading registry…"
      : "";

  return (
    <section className="generation-page">
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">{eyebrow}</p>
          <span>{title}</span>
        </div>
      </header>

      {registryBanner && (
        <p className={regError ? "analysis-error" : ""}>{registryBanner}</p>
      )}

      <div className="dev-controls-row">
        {/* ── Top: controls (own row) ── */}
        <aside className="panel generation-sidebar-card">
          <div className="generation-card-head">
            <h2>
              {currentLayer?.label || layerId}
              {currentAttempt && currentAttempt.available === false && (
                <span className="dev-unavailable-pill" title={currentAttempt.unavailable_reason || ""}>
                  Unavailable
                </span>
              )}
            </h2>
            <p>{currentAttempt?.label || attemptId}</p>
          </div>

          <div className="dev-controls-body">
            <div className="dev-controls-form">
              <LabeledSelect
                label="Layer"
                value={layerId}
                onChange={setLayerId}
                options={layerOptions}
              />

              <LabeledSelect
                label="Model / Attempt"
                value={attemptId}
                onChange={setAttemptId}
                options={attemptOptions}
              />

              <LabeledNumber
                label="Seed"
                value={seed}
                min={0}
                max={2147483647}
                hint={
                  usesSeed
                    ? "Same seed + same attempt = same audio."
                    : "This model does not use a seed."
                }
                onChange={setSeed}
                disabled={!usesSeed}
              />

              <div className="dev-controls-action">
                <button
                  type="button"
                  className="gen-primary-btn"
                  onClick={handleRun}
                  disabled={isLoading || !attemptId || !registryReady || currentAttempt?.available === false}
                  title={
                    currentAttempt?.available === false
                      ? currentAttempt?.unavailable_reason || "Model weights unavailable"
                      : undefined
                  }
                >
                  {isLoading ? "Generating..." : "Generate"}
                </button>
              </div>
            </div>

            {currentAttempt && currentAttempt.available === false && (
              <div className="dev-availability-warn" role="alert">
                <p className="dev-availability-title">⚠ Model weights unavailable</p>
                <p className="dev-availability-reason">
                  {currentAttempt.unavailable_reason ||
                    "Required weight files are not on disk."}
                </p>
                {currentAttempt.missing_files?.length > 0 && currentAttempt.checkpoint && (
                  <pre className="dev-availability-cmd">
{`dvc pull \\\n  ${currentAttempt.missing_files
  .map((f) => `${currentAttempt.checkpoint}/${f}`)
  .join(" \\\n  ")}`}
                  </pre>
                )}
              </div>
            )}

            {(isLoading || isDone) && (
              <ProgressBlock progress={progress} label={progressText} />
            )}

            {errorMsg && <p className="analysis-error">{errorMsg}</p>}

            <div className="dev-controls-meta">
              <div className="gen-info-block">
                <p>Attempt ID</p>
                <code>{attemptId || "—"}</code>
              </div>

              <div className="gen-info-block">
                <p>Run tag</p>
                <code>{isDone ? tag : "—"}</code>
              </div>

              <div className="gen-info-block">
                <p>Audio stats</p>
                <code>
                  {isDone && result?.metadata?.audio
                    ? `RMS ${result.metadata.audio.rms?.toFixed?.(4)} · peak ${result.metadata.audio.peak?.toFixed?.(4)}`
                    : "—"}
                </code>
              </div>
            </div>

            {isDone && result?.metadata?.prompt_locked && (
              <p className="mock-badge">
                Fixed prompt active — user prompts disabled
              </p>
            )}
          </div>
        </aside>

      </div>

      <div className="dev-results-row">
        {/* ── Left: expected results (cached) ── */}
        <main className="panel dev-result-card">
          <div className="generation-card-head">
            <h2>Expected Results</h2>
            <p>
              {samples?.canonical_seed != null
                ? `Cached samples · canonical seed ${samples.canonical_seed}`
                : "Cached expected / showcase samples"}
            </p>
          </div>

          <div className="dev-result-body">
            {samplesErr && <p className="analysis-error">{samplesErr}</p>}

            {expectedEntries.length > 1 && (
              <div className="dev-sample-tabs" role="tablist" aria-label="Cached samples">
                {expectedEntries.map((e) => (
                  <button
                    key={e.key}
                    type="button"
                    role="tab"
                    aria-selected={e.key === expectedKey}
                    className={`dev-sample-tab${e.key === expectedKey ? " active" : ""}`}
                    onClick={() => setExpectedKey(e.key)}
                  >
                    <span className="dev-sample-tab-tier">{e.tier}</span>
                    <span className="dev-sample-tab-stem">{e.sample.stem}</span>
                  </button>
                ))}
              </div>
            )}

            <ExpectedSample
              layerId={layerId}
              attemptId={attemptId}
              tier={expectedSelected?.tier}
              sample={expectedSelected?.sample}
              loading={!samplesErr && !samples}
              empty={!samplesErr && samples && expectedEntries.length === 0}
            />
          </div>
        </main>

        {/* ── Right: generated results (live run) ── */}
        <aside className="panel dev-result-card">
          <div className="generation-card-head">
            <h2>Generated Results</h2>
            <p>Live output from the latest run</p>
          </div>

          <div className="dev-result-body">
            <ReviewSection title="♪ Audio">
              {isDone && result?.audio_b64 ? (
                <>
                  <audio controls
                         src={`data:audio/wav;base64,${result.audio_b64}`}
                         style={{ width: "100%" }} />
                  <p style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
                    {result.sample_rate} Hz · {result.duration_s?.toFixed?.(1) ?? result.duration_s} s · seed{" "}
                    {result.metadata?.seed ?? seed}
                  </p>
                </>
              ) : (
                <Placeholder kind="audio" loading={isLoading}>
                  {isLoading ? "Generating audio…" : "Click Generate to run the handler."}
                </Placeholder>
              )}
            </ReviewSection>

            <ReviewSection title="▤ Mel-Spectrogram">
              {isDone && result?.image_b64 ? (
                <img src={`data:image/png;base64,${result.image_b64}`}
                     alt="Mel-spectrogram"
                     className="gen-spectrogram-img layer-a-spectrogram-img" />
              ) : (
                <Placeholder kind="image" loading={isLoading}>
                  {isLoading ? "Rendering spectrogram…" : "Spectrogram appears after generation."}
                </Placeholder>
              )}
            </ReviewSection>

            <ReviewSection title="{ } Metadata">
              {isDone && result?.metadata ? (
                <pre className="layer-a-json">
                  {JSON.stringify(result.metadata, null, 2)}
                </pre>
              ) : (
                <Placeholder kind="json" loading={isLoading}>
                  {isLoading ? "Collecting metadata…" : "Metadata appears after generation."}
                </Placeholder>
              )}
            </ReviewSection>

            {isLoading && (
              <ProgressBlock progress={progress} label={progressText} />
            )}

            <div className="dev-download-row">
              <button type="button" className="gen-secondary-btn"
                      disabled={!isDone || !result?.audio_b64}
                      onClick={() => downloadDataUrl(
                        `data:audio/wav;base64,${result.audio_b64}`,
                        `${tag}.wav`,
                      )}>
                ↓ WAV
              </button>
              <button type="button" className="gen-secondary-btn"
                      disabled={!isDone || !result?.image_b64}
                      onClick={() => downloadDataUrl(
                        `data:image/png;base64,${result.image_b64}`,
                        `${tag}_spectrogram.png`,
                      )}>
                ↓ Spectrogram
              </button>
              <button type="button" className="gen-secondary-btn"
                      disabled={!isDone || !result?.metadata}
                      onClick={downloadJson}>
                ↓ Metadata
              </button>
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}

function ExpectedSample({ layerId, attemptId, tier, sample, loading, empty }) {
  const hasSample = Boolean(sample);
  const wavSrc = hasSample && sample.has_wav
    ? sampleWavUrl(layerId, attemptId, tier, sample.stem)
    : null;

  const audioCaption = loading
    ? "Loading cached samples…"
    : empty
      ? "No cached samples on disk for this attempt yet."
      : !hasSample
        ? "Select a cached sample to preview."
        : "WAV not present locally — generate or run dvc pull.";

  const spectrogramCaption = loading
    ? "Loading…"
    : empty
      ? "No cached spectrogram available."
      : !hasSample
        ? "Spectrogram appears once a sample is selected."
        : sample.has_png
          ? "PNG is DVC-tracked — run dvc pull to render."
          : "No spectrogram on disk.";

  const metadataCaption = loading
    ? "Loading…"
    : empty
      ? "No metadata cached for this attempt yet."
      : !hasSample
        ? "Metadata appears once a sample is selected."
        : "No metadata on disk for this sample.";

  return (
    <>
      <ReviewSection title="♪ Audio">
        {wavSrc ? (
          <>
            <audio controls src={wavSrc} style={{ width: "100%" }}>
              Your browser doesn't support audio playback.
            </audio>
            <p style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
              <code>{sample.stem}</code> · {tier}
            </p>
          </>
        ) : (
          <Placeholder kind="audio" loading={loading}>{audioCaption}</Placeholder>
        )}
      </ReviewSection>

      <ReviewSection title="▤ Mel-Spectrogram">
        {hasSample && sample.png_b64 ? (
          <img
            src={`data:image/png;base64,${sample.png_b64}`}
            alt={`${tier} sample ${sample.stem}`}
            className="gen-spectrogram-img layer-a-spectrogram-img"
          />
        ) : (
          <Placeholder kind="image" loading={loading}>{spectrogramCaption}</Placeholder>
        )}
      </ReviewSection>

      <ReviewSection title="{ } Metadata">
        {hasSample && sample.metadata ? (
          <pre className="layer-a-json">
            {JSON.stringify(sample.metadata, null, 2)}
          </pre>
        ) : (
          <Placeholder kind="json" loading={loading}>{metadataCaption}</Placeholder>
        )}
      </ReviewSection>
    </>
  );
}

function Placeholder({ kind, loading, children }) {
  return (
    <div className={`dev-placeholder dev-placeholder-${kind}${loading ? " is-loading" : ""}`}>
      <div className="dev-placeholder-art" aria-hidden="true">
        {kind === "audio" && (
          <div className="dev-placeholder-waveform">
            {Array.from({ length: 28 }).map((_, i) => (
              <span key={i} style={{ ["--i"]: i }} />
            ))}
          </div>
        )}
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

function LabeledNumber({ label, value, min, max, step = 1, hint, onChange, disabled = false }) {
  return (
    <label className={`layer-a-field${disabled ? " is-disabled" : ""}`}>
      <span>{label}</span>
      <input className="layer-a-input" type="number"
             value={value} min={min} max={max} step={step}
             disabled={disabled}
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

function ReviewSection({ title, children }) {
  return (
    <section className="dev-review-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}
