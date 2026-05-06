import { useEffect, useRef, useState } from "react";
import AudioPlayer from "../components/AudioPlayer.jsx";
import SpectrogramCanvas from "../components/SpectrogramCanvas.jsx";
import { analyseAudio } from "../lib/api.js";
import { analyseAudioBasicFallback } from "../lib/basicAnalysis.js";

const apiBase = import.meta.env.VITE_API_URL || "";

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

function monthDisplay(estimated) {
  const rawMonth = Number(estimated?.month);
  if (Number.isFinite(rawMonth) && rawMonth >= 1 && rawMonth <= 12) {
    return MONTH_NAMES[Math.round(rawMonth) - 1];
  }

  const range = String(estimated?.month_range || "");
  return range
    .split("-")
    .map(part => part ? part[0].toUpperCase() + part.slice(1).toLowerCase() : part)
    .join("-");
}

// ---------------------------------------------------------------------------
// Latent vector stats (secondary display only)
// ---------------------------------------------------------------------------
function latentStats(latent) {
  const n    = latent.length;
  const norm = Math.sqrt(latent.reduce((s, v) => s + v * v, 0));
  const active = latent.filter(v => Math.abs(v) > 0.5).length;
  const diversity = Math.min(active / (n * 0.6), 1.0);
  return {
    norm:       parseFloat(norm.toFixed(2)),
    active_dims: active,
    latent_dim:  n,
    diversity:   parseFloat(diversity.toFixed(3)),
    complexity:  diversity > 0.65 ? "Rich" : diversity > 0.35 ? "Moderate" : "Sparse",
  };
}

function acousticStats(features = {}) {
  const windProxy = Number(features.wind_texture_proxy ?? 0);
  const rainProxy = Number(features.rain_texture_proxy ?? 0);
  const highRatio = Number(features.high_energy_ratio ?? 0);
  const activity = Math.min(Math.max((windProxy + rainProxy + highRatio) / 1.8, 0), 1);
  return {
    mode: "fallback",
    complexity: activity > 0.62 ? "Active" : activity > 0.32 ? "Moderate" : "Sparse",
    diversity: parseFloat(activity.toFixed(3)),
    norm: Number(features.rms_db ?? 0),
    active_dims: null,
    latent_dim: 0,
  };
}

function analysisStats(data) {
  if (Array.isArray(data?.latent) && data.latent.length > 0) {
    return { ...latentStats(data.latent), mode: "latent" };
  }
  return acousticStats(data?.acoustic_features);
}

// ---------------------------------------------------------------------------
// Env condition display config
// ---------------------------------------------------------------------------
const ENV_FIELDS = [
  { key: "temperature_c",       label: "Temperature",      unit: "°C",   icon: "◈" },
  { key: "humidity_pct",        label: "Humidity",         unit: "%",    icon: "◈" },
  { key: "wind_speed_ms",       label: "Wind Speed",       unit: "m/s",  icon: "◈" },
  { key: "wind_max_ms",         label: "Wind Gust",        unit: "m/s",  icon: "◈" },
  { key: "precipitation_mm",    label: "Precipitation",    unit: "mm",   icon: "◈" },
  { key: "days_since_rain",     label: "Days Since Rain",  unit: "days", icon: "◈" },
  { key: "solar_radiation_wm2", label: "Solar Radiation",  unit: "W/m²", icon: "◈" },
  { key: "surface_pressure_kpa",label: "Air Pressure",     unit: "kPa",  icon: "◈" },
  { key: "daylight_hours",      label: "Daylight Hours",   unit: "hrs",  icon: "◈" },
  { key: "hour_local",          label: "Time of Day",      unit: "hr",   icon: "◈",
    format: v => {
      const h = Math.round(v);
      const ampm = h < 12 ? "AM" : "PM";
      const hr   = h % 12 || 12;
      return `${hr}:00 ${ampm}`;
    }
  },
];

const BASIC_FEATURE_FIELDS = [
  { key: "rms_db", label: "RMS Loudness", unit: "dB" },
  { key: "spectral_centroid_hz", label: "Spectral Centroid", unit: "Hz" },
  { key: "low_high_energy_ratio", label: "Low / High Energy", unit: "" },
  { key: "transient_rate_per_sec", label: "Transient Rate", unit: "/s" },
  { key: "duration_sec", label: "Duration", unit: "sec" },
  { key: "sound_density", label: "Sound Density", unit: "" },
  { key: "brightness", label: "Brightness", unit: "" },
  { key: "brightness_label", label: "Brightness Label", unit: "" },
  { key: "activity_score", label: "Activity Score", unit: "" },
];

export default function HomePage() {
  const fileInputRef = useRef(null);

  const [health,     setHealth]     = useState({ loading: true });
  const [file,       setFile]       = useState(null);
  const [audioUrl,   setAudioUrl]   = useState(null);
  const [status,     setStatus]     = useState("idle");
  const [estimated,  setEstimated]  = useState(null);   // estimated_conditions from server
  const [stats,      setStats]      = useState(null);   // latent stats
  const [rawLatent,  setRawLatent]  = useState(null);
  const [errorMsg,   setErrorMsg]   = useState("");
  const [dragging,   setDragging]   = useState(false);

  useEffect(() => {
    async function loadHealth() {
      try {
        const [backendRes, aiRes] = await Promise.allSettled([
          fetch(`${apiBase}/api/health`).then(r => r.json()),
          fetch(`${apiBase}/api/ai/health`).then(r => r.json()),
        ]);

        const backendHealth = backendRes.status === "fulfilled"
          ? backendRes.value
          : { ok: false, message: String(backendRes.reason) };
        const aiHealth = aiRes.status === "fulfilled"
          ? aiRes.value
          : { ok: false, message: String(aiRes.reason) };

        setHealth({ ...backendHealth, ai: aiHealth });
      } catch (e) {
        setHealth({ ok: false, ai: { ok: false }, message: String(e) });
      }
    }

    loadHealth();
  }, []);

  const healthText = health.loading
    ? "Scanning node telemetry…"
    : health.ok
      ? "Operational"
      : "Attention required";
  const aiHealth = health.ai ?? {};
  const checkpointPath = aiHealth.checkpoint || "acoustic_ai/checkpoints/best.pt";
  const checkpointReady = Boolean(aiHealth.ok && aiHealth.exists);
  const aiServerReady = Boolean(aiHealth.ok);
  const fallbackAvailable = Boolean(aiHealth.analysis_modes?.basic_audio_fallback ?? aiServerReady);
  const analysisCanRun = aiServerReady && (checkpointReady || fallbackAvailable);
  const analysisUnavailableReason = health.loading
    ? "Checking model checkpoint…"
    : checkpointReady
      ? ""
      : aiHealth.message
        ? `AI server unavailable: ${aiHealth.message}`
        : `Basic Analysis Mode (No model loaded). Advanced latent analysis unavailable.`;
  const analysisHealthText = health.loading
    ? "checking"
    : checkpointReady
      ? "ready"
      : aiServerReady && fallbackAvailable
        ? "basic mode"
        : "unavailable";

  function acceptFile(f) {
    if (!f) return;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setFile(f);
    setAudioUrl(URL.createObjectURL(f));
    setEstimated(null);
    setStats(null);
    setRawLatent(null);
    setStatus("idle");
    setErrorMsg("");
  }

  function onFileChange(e)  { acceptFile(e.target.files?.[0] ?? null); }
  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    acceptFile(e.dataTransfer.files?.[0] ?? null);
  }

  async function runAnalysis() {
    if (!file) return;
    if (!analysisCanRun) {
      setErrorMsg(analysisUnavailableReason);
      setStatus("error");
      return;
    }
    setStatus("analysing");
    setErrorMsg("");
    try {
      const data = await analyseAudio(file);
      setRawLatent(data);
      setStats(analysisStats(data));
      setEstimated(data.estimated_conditions ?? null);
      setStatus("done");
    } catch (err) {
      const is401 = err.message.includes("401") || err.message.toLowerCase().includes("authenticated");
      const checkpointMissing = err.message.toLowerCase().includes("checkpoint");
      if (!is401 && checkpointMissing) {
        try {
          const data = await analyseAudioBasicFallback(file);
          setRawLatent(data);
          setStats(analysisStats(data));
          setEstimated(data.estimated_conditions ?? null);
          setErrorMsg("");
          setStatus("done");
          return;
        } catch (fallbackErr) {
          setErrorMsg(`Backend analysis failed and browser fallback failed: ${fallbackErr.message}`);
          setStatus("error");
          return;
        }
      }
      setErrorMsg(is401 ? "Login required — sign in to run analysis." : err.message);
      setStatus("error");
    }
  }

  const isAnalysing = status === "analysing";
  const isDone      = status === "done";
  const estimatedMonth = monthDisplay(estimated);

  return (
    <section className="dashboard-page">
      <header className="topbar">
        <div>
          <p className="eyebrow">SONIC LABORATORY</p>
          <h1>ANALYSIS PIPELINE</h1>
          <div className="status-line">
            <span className="status-accent" />
            <p>
              System status: {healthText}
              {" / "}
              AI analysis: {analysisHealthText}
              {" / "}
              {apiBase || "Local node"}
            </p>
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
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
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
                  disabled={isAnalysing || !analysisCanRun}
                  title={analysisCanRun ? undefined : analysisUnavailableReason}
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

              {!checkpointReady && analysisCanRun && <p className="analysis-error">{analysisUnavailableReason}</p>}
              {errorMsg && <p className="analysis-error">{errorMsg}</p>}
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
          <WaveBars file={file} active={Boolean(file)} />
        </section>

        {/* ── Estimated Environmental Conditions ── */}
        <section className="panel metrics-panel">
          <div className="panel-heading">
            <h3>Estimated Environmental Conditions</h3>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {isDone && estimated && (
                <span className="pipeline-badge pipeline-badge--proxy">
                  {Math.round((estimated.confidence ?? 0) * 100)}% confidence
                </span>
              )}
              <p>
                {isDone
                  ? rawLatent?.analysis_mode === "basic_fallback"
                    ? "Heuristic audio-feature estimation"
                    : "Nearest-neighbour inference from latent space"
                  : "—"}
              </p>
            </div>
          </div>

          {isDone && estimated && Object.keys(estimated).length > 0 ? (
            <>
              {/* Month + time of day badges */}
              <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
                {estimatedMonth && (
                  <span className="pipeline-badge pipeline-badge--live">
                    Month: {estimatedMonth}
                  </span>
                )}
                {estimated.sample_bin && (
                  <span className="pipeline-badge pipeline-badge--live" style={{ textTransform: "capitalize" }}>
                    {estimated.sample_bin}
                  </span>
                )}
              </div>

              {/* Numeric env fields */}
              <div className="metric-list">
                {ENV_FIELDS.map(({ key, label, unit, format }) => {
                  const raw = estimated[key];
                  if (raw === undefined) return null;
                  const display = format ? format(raw) : `${raw} ${unit}`;
                  return (
                    <div key={key} className="metric-row" style={{ alignItems: "center" }}>
                      <span className="metric-label-row" style={{ width: "100%", justifyContent: "space-between" }}>
                        <span style={{ color: "var(--text-secondary, #888)", fontSize: "0.78rem" }}>{label}</span>
                        <strong style={{ fontSize: "0.88rem" }}>{display}</strong>
                      </span>
                    </div>
                  );
                })}
              </div>

              <p className="metrics-proxy-note" style={{ marginTop: 12 }}>
                {rawLatent?.analysis_mode === "basic_fallback"
                  ? "Basic Analysis Mode: direct audio features plus explainable wind/rain/activity heuristics. Advanced latent analysis unavailable."
                  : "Inferred from top-5 nearest clips in learned latent space."}
                Species-specific analysis remains limited by sparse annotations.
              </p>
            </>
          ) : isDone ? (
            <p className="metrics-proxy-note">
              latent_clips.npy not found — re-run <code>precompute_latents.py</code> to enable env estimation.
            </p>
          ) : !checkpointReady && analysisCanRun ? (
            <p className="metrics-proxy-note" style={{ marginTop: 12 }}>
              VAE analysis requires <code>best.pt</code>; basic audio-feature fallback can still run.
            </p>
          ) : (
            <p className="metrics-proxy-note" style={{ marginTop: 12 }}>
              Upload an audio file and run analysis to estimate environmental conditions.
            </p>
          )}
        </section>

        {isDone && rawLatent?.acoustic_features && (
          <section className="panel metrics-panel">
            <div className="panel-heading">
              <h3>Basic Audio Features</h3>
              <p>{rawLatent.analysis_mode === "basic_fallback" ? "No model loaded" : "Always available"}</p>
            </div>
            <div className="metric-list">
              {BASIC_FEATURE_FIELDS.map(({ key, label, unit }) => {
                const raw = rawLatent.acoustic_features?.[key];
                if (raw === undefined) return null;
                const value = typeof raw === "number" ? raw.toFixed(key.includes("ratio") || key.includes("density") || key.includes("brightness") || key.includes("score") ? 3 : 2) : raw;
                return (
                  <div key={key} className="metric-row" style={{ alignItems: "center" }}>
                    <span className="metric-label-row" style={{ width: "100%", justifyContent: "space-between" }}>
                      <span style={{ color: "var(--text-secondary, #888)", fontSize: "0.78rem" }}>{label}</span>
                      <strong style={{ fontSize: "0.88rem" }}>{value}{unit ? ` ${unit}` : ""}</strong>
                    </span>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {isDone && rawLatent?.heuristic_environment && (
          <section className="panel summary-panel">
            <div className="panel-heading">
              <h3>Heuristic Environment</h3>
              <p>Explainable fallback</p>
            </div>
            <div className="metric-list">
              {["wind", "rain", "activity", "time_of_day_hint"].map(key => {
                const item = rawLatent.heuristic_environment?.[key];
                if (!item) return null;
                const label = key === "time_of_day_hint" ? "Time-of-day hint" : key[0].toUpperCase() + key.slice(1);
                const level = item.level || item.label || "—";
                const confidence = Math.round((item.confidence ?? 0) * 100);
                return (
                  <div key={key} className="metric-row" style={{ display: "block" }}>
                    <span className="metric-label-row" style={{ width: "100%", justifyContent: "space-between" }}>
                      <span style={{ color: "var(--text-secondary, #888)", fontSize: "0.78rem" }}>{label}</span>
                      <strong style={{ fontSize: "0.88rem", textTransform: "capitalize" }}>{level} · {confidence}%</strong>
                    </span>
                    <p className="metrics-proxy-note" style={{ marginTop: 6 }}>{item.explanation}</p>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* ── Stat cards ── */}
        <section className="stats-split">
          <article className="panel stat-card">
            <p className="stat-label">Soundscape Structure</p>
            <strong>{stats ? stats.complexity : "—"}</strong>
            <span>
              {stats
                ? stats.mode === "latent"
                  ? `${stats.active_dims} / ${stats.latent_dim} active latent dims`
                  : "Basic acoustic activity proxy"
                : "Upload audio to analyse"}
            </span>
          </article>
          <article className="panel stat-card">
            <p className="stat-label">{stats?.mode === "fallback" ? "Acoustic Activity" : "Latent Diversity"}</p>
            <strong>{stats ? stats.diversity.toFixed(3) : "—"}</strong>
            <span>{stats ? (stats.mode === "fallback" ? `RMS = ${stats.norm} dB` : `‖z‖ = ${stats.norm}`) : "Awaiting input"}</span>
          </article>
        </section>

        {/* ── Neural Summary ── */}
        <section className="panel summary-panel">
          <div className="panel-heading">
            <h3>Neural Summary</h3>
            <p>
              {isDone
                ? rawLatent?.analysis_mode === "basic_fallback"
                  ? "basic analysis"
                  : `${rawLatent?.latent_dim ?? 256}-dim latent`
                : "—"}
            </p>
          </div>
          <p className="summary-text">
            {isDone && rawLatent?.analysis_mode === "basic_fallback"
              ? `Basic Analysis Mode ran without a loaded VAE checkpoint. ` +
                `It extracted loudness, frequency balance, transient density, and spectral texture, then estimated ` +
                `${rawLatent?.heuristic_environment?.wind?.level ?? "unknown"} wind, ` +
                `${rawLatent?.heuristic_environment?.rain?.level ?? "unknown"} rain, and ` +
                `${rawLatent?.heuristic_environment?.activity?.level ?? "unknown"} biological activity. ` +
                `Advanced latent analysis remains unavailable until best.pt is restored.`
              : isDone && estimated && Object.keys(estimated).length > 0
              ? `Module A encoded the clip into a 256-dim latent vector (‖z‖ = ${stats?.norm ?? "—"}). ` +
                `Nearest-neighbour lookup estimated ${estimatedMonth || "month-unavailable"} ${estimated.sample_bin} conditions: ` +
                `${estimated.temperature_c}°C, ${estimated.humidity_pct}% humidity, ` +
                `${estimated.wind_speed_ms} m/s wind. ` +
                `Confidence: ${Math.round((estimated.confidence ?? 0) * 100)}%. ` +
                `Species-specific analysis remains limited; generation uses retrieval-based biological events.`
              : isDone
              ? `Module A encoded the clip into a ${stats?.latent_dim ?? 256}-dim latent vector (‖z‖ = ${stats?.norm ?? "—"}). ` +
                `Run precompute_latents.py to enable environmental condition inference.`
              : "Upload an audio file and run analysis to generate a neural summary."}
          </p>
        </section>

        {/* ── Pipeline Status ── */}
        <section className="panel pipeline-status-panel">
          <div className="panel-heading">
            <h3>Pipeline Status</h3>
            <p>What's working</p>
          </div>
          <div className="pipeline-stage-list">
            <PipelineStage
              state="live"
              label="File upload + browser spectrogram"
              detail="Client-side — no server needed"
            />
            <PipelineStage
              state="live"
              label="Audio player"
              detail="Plays uploaded file immediately"
            />
            <PipelineStage
              state={checkpointReady ? "live" : "blocked"}
              label="VAE encode → latent vector (256-d)"
              detail={
                checkpointReady
                  ? `Module A · VAE · ${rawLatent ? `last run: ${rawLatent.latent_dim}-dim` : "best.pt ready"}`
                  : "Advanced latent analysis unavailable"
              }
            />
            <PipelineStage
              state={
                !checkpointReady && !analysisCanRun
                  ? "blocked"
                  : isDone && estimated && Object.keys(estimated).length > 0
                    ? "live"
                    : "proxy"
              }
              label="Environmental condition inference"
              detail={
                !checkpointReady
                  ? "Basic Analysis Mode estimates wind, rain, activity, and time hint from audio features"
                  : isDone && estimated && Object.keys(estimated).length > 0
                  ? `Top-5 nearest neighbours · confidence ${Math.round((estimated.confidence ?? 0) * 100)}%`
                  : "Requires latent_clips.npy — run precompute_latents.py"
              }
            />
            <PipelineStage
              state="pending"
              label="Species-specific analysis labels"
              detail="Sparse annotations — classifier not trained for analysis mode"
            />
            <PipelineStage
              state={analysisCanRun ? "live" : "blocked"}
              label="Analysis summary output"
              detail={checkpointReady ? "Latent statistics + environmental estimate when available" : "Basic Analysis Mode (No model loaded)"}
            />
          </div>
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

// ── Sub-components ─────────────────────────────────────────────────────────

function WaveBars({ file, active }) {
  const [bars, setBars] = useState(() => Array.from({ length: 36 }, () => 4));

  useEffect(() => {
    if (!file) {
      setBars(Array.from({ length: 36 }, () => 4));
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;
        const audioCtx = new AudioCtx();
        const audioBuffer = await audioCtx.decodeAudioData(await file.arrayBuffer());
        await audioCtx.close();
        if (cancelled) return;

        const samples = audioBuffer.getChannelData(0);
        const count = 36;
        const block = Math.max(1, Math.floor(samples.length / count));
        const rms = Array.from({ length: count }, (_, i) => {
          const start = i * block;
          const end = Math.min(samples.length, start + block);
          let sum = 0;
          for (let j = start; j < end; j += 1) sum += samples[j] * samples[j];
          return Math.sqrt(sum / Math.max(1, end - start));
        });
        const max = Math.max(...rms, 1e-6);
        setBars(rms.map(value => 6 + Math.round((value / max) ** 0.65 * 58)));
      } catch (err) {
        if (!cancelled) setBars(Array.from({ length: 36 }, () => 4));
      }
    })();

    return () => { cancelled = true; };
  }, [file]);

  return (
    <div className="wave-bars" aria-label="RMS energy envelope">
      {bars.map((height, i) => (
        <span
          key={i}
          className="wave-bar"
          style={{ height: `${active ? height : 4}px`, transition: "height 0.6s ease" }}
        />
      ))}
    </div>
  );
}

const STAGE_META = {
  live:    { icon: "✓", cls: "pipeline-stage--live",    label: "LIVE"    },
  proxy:   { icon: "◑", cls: "pipeline-stage--proxy",   label: "PROXY"   },
  blocked: { icon: "⊘", cls: "pipeline-stage--blocked", label: "BLOCKED" },
  pending: { icon: "○", cls: "pipeline-stage--pending", label: "PENDING" },
};

function PipelineStage({ state, label, detail }) {
  const meta = STAGE_META[state] ?? STAGE_META.pending;
  return (
    <div className={`pipeline-stage ${meta.cls}`}>
      <span className="pipeline-stage-icon">{meta.icon}</span>
      <div className="pipeline-stage-body">
        <span className="pipeline-stage-label">{label}</span>
        <span className="pipeline-stage-detail">{detail}</span>
      </div>
      <span className="pipeline-stage-pill">{meta.label}</span>
    </div>
  );
}
