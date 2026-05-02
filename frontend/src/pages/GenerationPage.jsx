import { useState } from "react";
import AudioPlayer from "../components/AudioPlayer.jsx";
import EnvControls, { DEFAULT_CONDITIONS, monthLabel } from "../components/EnvControls.jsx";
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
    a.download = `soundscape_${monthLabel(conditions.month)}_${conditions.sample_bin}.wav`;
    a.click();
  }

  const isGenerating = status === "generating";
  const isDone       = status === "done";
  const audioMime    = result?.audio_mime || "audio/wav";
  const audioExt     = result?.audio_ext || "wav";
  const audioSrc     = result?.audio_b64 ? `data:${audioMime};base64,${result.audio_b64}` : null;
  const selectedClip = result?.selected?.clip_path;
  const isLayerA     = result?.mode === "layer_a_ambient_bed";
  const conditionMonth = monthLabel(conditions.month);
  const selectedMonth  = result?.selected?.month ? monthLabel(result.selected.month) : conditionMonth;
  const weather = result?.weather;
  const weatherLayers = weather?.layers;
  const activeWeather = weatherLayers
    ? Object.entries(weatherLayers).filter(([, layer]) => layer?.enabled)
    : [];

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
                  <strong>27.987°S 145.609°E</strong>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* ── Centre: canvas / status ── */}
        <main className="panel generation-canvas-card">
          <div className="generation-canvas">
            {isDone && result?.image_b64 ? (
              <img
                src={`data:image/png;base64,${result.image_b64}`}
                alt="Generated mel-spectrogram"
                className="gen-spectrogram-img"
              />
            ) : (
              <>
                <div className="gen-wave gen-wave-back" />
                <div className="gen-wave gen-wave-mid" />
                <div className="gen-wave gen-wave-front" />
              </>
            )}

            {isGenerating && (
              <div className="gen-computing-overlay">
                <div className="gen-computing-ring" />
                <p>Retrieving ambient bed…</p>
              </div>
            )}
          </div>

          <div className="generation-caption">
            <h1>{isDone && isLayerA ? "Ambient Bed Retrieval" : "Latent Soundscape Synthesis"}</h1>
            <div className="generation-caption-row">
              <i />
              <p>
                {isGenerating
                  ? "Retrieving ambient bed…"
                  : isDone && isLayerA
                    ? `Layer A ambient bed · ${selectedMonth} · ${result.selected?.sample_bin}`
                    : isDone
                      ? `${conditionMonth} · ${conditions.sample_bin} · ${conditions.temperature_c}°C`
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
                <p>{isGenerating ? "Retrieving…" : isDone ? "Done" : "Awaiting input"}</p>
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
                      ? `${conditionMonth.toUpperCase()}_${conditions.sample_bin.toUpperCase()}`
                      : "—"}
                  </strong>
                </div>
              </div>
              <OutputBars active={isDone} />
            </article>

            {isDone && result?.mock && (
              <p className="mock-badge">Demo mode — connect model server for real audio</p>
            )}

            {isDone && selectedClip && (
              <p className="mock-badge">
                Layer A selected: {selectedClip.split("/").slice(-2).join("/")}
              </p>
            )}

            {isDone && weather && (
              <div className="metrics-proxy-note">
                <strong>Layer B weather: {weather.status}</strong>
                {activeWeather.length > 0 ? (
                  <ul className="gen-weather-list">
                    {activeWeather.map(([kind, layer]) => (
                      <li key={kind}>
                        {kind}: {layer.target_intensity}
                        {" · "}
                        confidence {Math.round((layer.confidence ?? 0) * 100)}%
                        {" · "}
                        {layer.gain_db} dB
                      </li>
                    ))}
                  </ul>
                ) : (
                  <span> · no wind/rain layer selected</span>
                )}
              </div>
            )}

            {isDone && result?.explanation && (
              <p className="metrics-proxy-note">{result.explanation}</p>
            )}

            <button
              type="button"
              className="gen-primary-btn"
              onClick={handleGenerate}
              disabled={isGenerating}
            >
              {isGenerating ? "Generating…" : "▣ Generate Soundscape"}
            </button>

            {isDone && result?.image_b64 && (
              <button
                type="button"
                className="gen-secondary-btn"
                onClick={() => {
                  const a = document.createElement("a");
                  a.href = `data:image/png;base64,${result.image_b64}`;
                  a.download = `spectrogram_${conditionMonth}_${conditions.sample_bin}.png`;
                  a.click();
                }}
              >
                ↓ Download Spectrogram
              </button>
            )}

            {isDone && result?.audio_b64 && (
              <button
                type="button"
                className="gen-secondary-btn"
                onClick={() => {
                  const a = document.createElement("a");
                  a.href = audioSrc;
                  a.download = `soundscape_${conditionMonth}_${conditions.sample_bin}.${audioExt}`;
                  a.click();
                }}
              >
                ↓ Download Audio (.{audioExt})
              </button>
            )}
          </div>
        </aside>
      </div>

      <AudioPlayer
        src={audioSrc}
        label={isDone ? `${conditionMonth} · ${conditions.sample_bin}` : "Generation Output"}
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
