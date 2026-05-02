import { useEffect, useRef, useState } from "react";
import AudioPlayer from "../components/AudioPlayer.jsx";
import SpectrogramCanvas from "../components/SpectrogramCanvas.jsx";
import { analyseAudio } from "../lib/api.js";

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
    fetch(`${apiBase}/api/health`)
      .then(r => r.json())
      .then(d => setHealth(d))
      .catch(e => setHealth({ ok: false, message: String(e) }));
  }, []);

  const healthText = health.loading
    ? "Scanning node telemetry…"
    : health.ok
      ? "Operational"
      : "Attention required";

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
    setStatus("analysing");
    setErrorMsg("");
    try {
      const data = await analyseAudio(file);
      setRawLatent(data);
      setStats(latentStats(data.latent));
      setEstimated(data.estimated_conditions ?? null);
      setStatus("done");
    } catch (err) {
      const is401 = err.message.includes("401") || err.message.toLowerCase().includes("authenticated");
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
          <WaveBars active={isDone} />
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
              <p>{isDone ? "Nearest-neighbour inference from latent space" : "—"}</p>
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
                Inferred from top-5 nearest clips in learned latent space.
                Species labels pending Module B (Stage 3).
              </p>
            </>
          ) : isDone ? (
            <p className="metrics-proxy-note">
              latent_clips.npy not found — re-run <code>precompute_latents.py</code> to enable env estimation.
            </p>
          ) : (
            <p className="metrics-proxy-note" style={{ marginTop: 12 }}>
              Upload an audio file and run analysis to estimate environmental conditions.
            </p>
          )}
        </section>

        {/* ── Stat cards ── */}
        <section className="stats-split">
          <article className="panel stat-card">
            <p className="stat-label">Soundscape Structure</p>
            <strong>{stats ? stats.complexity : "—"}</strong>
            <span>{stats ? `${stats.active_dims} / ${stats.latent_dim} active latent dims` : "Upload audio to analyse"}</span>
          </article>
          <article className="panel stat-card">
            <p className="stat-label">Latent Diversity</p>
            <strong>{stats ? stats.diversity.toFixed(3) : "—"}</strong>
            <span>{stats ? `‖z‖ = ${stats.norm}` : "Awaiting input"}</span>
          </article>
        </section>

        {/* ── Neural Summary ── */}
        <section className="panel summary-panel">
          <div className="panel-heading">
            <h3>Neural Summary</h3>
            <p>{isDone ? `${rawLatent?.latent_dim ?? 256}-dim latent` : "—"}</p>
          </div>
          <p className="summary-text">
            {isDone && estimated && Object.keys(estimated).length > 0
              ? `Module A encoded the clip into a 256-dim latent vector (‖z‖ = ${stats?.norm ?? "—"}). ` +
                `Nearest-neighbour lookup estimated ${estimatedMonth || "month-unavailable"} ${estimated.sample_bin} conditions: ` +
                `${estimated.temperature_c}°C, ${estimated.humidity_pct}% humidity, ` +
                `${estimated.wind_speed_ms} m/s wind. ` +
                `Confidence: ${Math.round((estimated.confidence ?? 0) * 100)}%. ` +
                `Semantic species labels pending Module B training.`
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
              state="live"
              label="VAE encode → latent vector (256-d)"
              detail={`Module A · VAE · best.pt · ${rawLatent ? `last run: ${rawLatent.latent_dim}-dim` : "ready"}`}
            />
            <PipelineStage
              state={isDone && estimated && Object.keys(estimated).length > 0 ? "live" : "proxy"}
              label="Environmental condition inference"
              detail={
                isDone && estimated && Object.keys(estimated).length > 0
                  ? `Top-5 nearest neighbours · confidence ${Math.round((estimated.confidence ?? 0) * 100)}%`
                  : "Requires latent_clips.npy — run precompute_latents.py"
              }
            />
            <PipelineStage
              state="pending"
              label="Bioacoustic species labels"
              detail="Module B · ecological classifier · Stage 3"
            />
            <PipelineStage
              state="pending"
              label="Audio reconstruction output"
              detail="Module C · conditioned generator · Stage 3"
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

function WaveBars({ active }) {
  const base = [10, 24, 44, 62, 51, 18, 37, 55, 29, 8, 40, 59, 32, 15, 48, 24, 11, 31, 57, 63, 54, 19, 29, 55, 36, 7, 24];
  return (
    <div className="wave-bars" aria-hidden="true">
      {base.map((height, i) => (
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
