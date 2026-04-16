import { useEffect, useState } from "react";

const apiBase = import.meta.env.VITE_API_URL || "";
const bars = [10, 24, 44, 62, 51, 18, 37, 55, 29, 8, 40, 59, 32, 15, 48, 24, 11, 31, 57, 63, 54, 19, 29, 55, 36, 7, 24];
const speciesMetrics = [
  { label: "Avian biophony", value: 82, tone: "cyan" },
  { label: "Anthropogenic noise", value: 12, tone: "rose" },
  { label: "Insect geophony", value: 44, tone: "mint" },
];

export default function HomePage() {
  const [health, setHealth] = useState({ loading: true });

  useEffect(() => {
    fetch(`${apiBase}/api/health`)
      .then((response) => response.json())
      .then((data) => setHealth(data))
      .catch((error) => setHealth({ ok: false, message: String(error) }));
  }, []);

  const healthText = health.loading
    ? "Scanning node telemetry..."
    : health.ok
      ? `Operational${health.message ? ` • ${health.message}` : ""}`
      : `Attention required${health.message ? ` • ${health.message}` : ""}`;

  return (
    <section className="dashboard-page">
      <header className="topbar">
        <div>
          <p className="eyebrow">SONIC LABORATORY</p>
          <h1>ANALYSIS PIPELINE</h1>
          <div className="status-line">
            <span className="status-accent" />
            <p>
              System status: {healthText} / {apiBase || "Local node"}
            </p>
          </div>
        </div>

        <div className="topbar-tools">
          <label className="search-box">
            <span>⌕</span>
            <input type="text" placeholder="Search parameters..." />
          </label>
          <button type="button" className="icon-button" aria-label="Settings">
            ⚙
          </button>
          <button type="button" className="icon-button" aria-label="Alerts">
            ◔
          </button>
          <button type="button" className="icon-button" aria-label="Help">
            ?
          </button>
        </div>
      </header>

      <div className="content-grid">
        <section className="hero-upload panel panel-hero">
          <div className="upload-icon">⇪</div>
          <h2>DROP AUDIO FOR DEEP ANALYSIS</h2>
          <p>FLAC, WAV, MP3 up to 2GB • multi-channel supported</p>
        </section>

        <section className="panel spectral-panel">
          <div className="panel-heading">
            <h3>Spectral Mapping</h3>
            <p>Resolution: 96kHz / 24-bit</p>
          </div>

          <div className="spectrogram">
            <div className="scan-line" />
          </div>

          <div className="wave-bars" aria-hidden="true">
            {bars.map((height, index) => (
              <span
                key={`${height}-${index}`}
                className="wave-bar"
                style={{ height: `${height}px` }}
              />
            ))}
          </div>
        </section>

        <section className="panel metrics-panel">
          <div className="panel-heading">
            <h3>Species Activity Indicators</h3>
            <p>i</p>
          </div>

          <div className="metric-list">
            {speciesMetrics.map((metric) => (
              <div key={metric.label} className="metric-row">
                <div className="metric-label-row">
                  <span>{metric.label}</span>
                  <span>{metric.value}%</span>
                </div>
                <div className="metric-track">
                  <span
                    className={`metric-fill metric-${metric.tone}`}
                    style={{ width: `${metric.value}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="stats-split">
          <article className="panel stat-card">
            <p className="stat-label">Acoustic Diversity</p>
            <strong>0.742</strong>
            <span>High variance detected</span>
          </article>

          <article className="panel stat-card">
            <p className="stat-label">Soundscape Structure</p>
            <strong>Complex</strong>
            <span>4 distinct layers identified</span>
          </article>
        </section>

        <section className="panel summary-panel">
          <div className="panel-heading">
            <h3>Neural Summary Description</h3>
            <p>Peak volume</p>
          </div>
          <p className="summary-text">
            Anthropogenic interference is minimal. The soundscape maintains a dense
            structural integrity with persistent mid-band activity, suggesting a stable
            habitat with layered biological signatures.
          </p>
        </section>
      </div>

      <footer className="player-bar panel">
        <button type="button" className="play-button" aria-label="Play audio">
          ▶
        </button>
        <div className="player-copy">
          <strong>MEDIA CONTROLLER</strong>
          <p>Session_Alpha_04.wav • 00:45 / 03:22</p>
        </div>
        <div className="player-levels" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
        </div>
      </footer>
    </section>
  );
}
