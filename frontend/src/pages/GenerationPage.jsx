const environmentSliders = [
  { label: "Temperature", value: "+5°C", progress: "64%" },
  { label: "Wind Speed", value: "2.0x", progress: "42%" },
  { label: "Atmosphere Density", value: "High", progress: "82%" },
];

const renderBars = [22, 44, 68, 31, 83, 53, 18, 40, 66, 75, 29, 55];

export default function GenerationPage() {
  return (
    <section className="generation-page">
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">SONIC LABORATORY</p>
          <span>Speculative pipeline v2.4</span>
        </div>

        <label className="search-box generation-search">
          <span>⌕</span>
          <input type="text" placeholder="Search models..." />
        </label>
      </header>

      <div className="generation-grid">
        <aside className="panel generation-sidebar-card">
          <div className="generation-card-head">
            <h2>Condition Encoding</h2>
            <p>Environmental variables</p>
          </div>

          <div className="generation-sidebar-body">
            {environmentSliders.map((item) => (
              <div key={item.label} className="gen-control">
                <div className="gen-control-head">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
                <div className="gen-slider">
                  <i style={{ insetInlineEnd: `calc(${item.progress} - 12px)` }} />
                </div>
              </div>
            ))}

            <div className="gen-info-block">
              <p>Site information</p>
              <div className="gen-site-card">
                <div>
                  <span>Location</span>
                  <strong>Neo-Tokyo Sector 7</strong>
                </div>
                <div>
                  <span>Acoustic type</span>
                  <strong>Metallic Void</strong>
                </div>
              </div>
            </div>

            <div className="gen-info-block">
              <p>Reference audio</p>
              <div className="gen-dropzone">
                <span>⇪</span>
                <strong>Drop .wav or .aiff</strong>
              </div>
            </div>
          </div>
        </aside>

        <main className="panel generation-canvas-card">
          <div className="generation-canvas">
            <div className="gen-wave gen-wave-back" />
            <div className="gen-wave gen-wave-mid" />
            <div className="gen-wave gen-wave-front" />
            <span className="gen-node gen-node-one">● Frequency Node 0.44Hz</span>
            <span className="gen-node gen-node-two">● Latent Vector 2.89</span>
          </div>

          <div className="generation-caption">
            <h1>Latent Soundscape Synthesis</h1>
            <div className="generation-caption-row">
              <i />
              <p>Processing real-time stream</p>
            </div>
          </div>
        </main>

        <aside className="panel generation-output-card">
          <div className="generation-card-head">
            <h2>Output Reconstruction</h2>
            <p>Generated artifact</p>
          </div>

          <div className="generation-output-body">
            <div className="gen-progress-block">
              <span>Progress</span>
              <div className="gen-progress-line">
                <strong>82%</strong>
                <p>Reconstructing...</p>
              </div>
              <div className="gen-progress-track">
                <i />
              </div>
            </div>

            <article className="gen-file-card">
              <div className="gen-file-head">
                <div className="gen-file-icon">▤</div>
                <div>
                  <span>Filename</span>
                  <strong>SPEC_001_S7_METALLIC</strong>
                </div>
              </div>
              <div className="gen-render-bars" aria-hidden="true">
                {renderBars.map((bar, index) => (
                  <i key={`${bar}-${index}`} style={{ height: `${bar}px` }} />
                ))}
              </div>
            </article>

            <button type="button" className="gen-primary-btn">
              ▣ Save soundscape
            </button>
            <button type="button" className="gen-secondary-btn">
              Export metadata
            </button>
          </div>
        </aside>
      </div>

      <footer className="panel generation-player">
        <div className="generation-player-left">
          <button type="button" aria-label="Previous">
            ◀
          </button>
          <button type="button" className="active" aria-label="Play">
            ▶
          </button>
          <button type="button" aria-label="Next">
            ▶
          </button>
          <div className="generation-player-copy">
            <strong>Media Controller</strong>
            <p>Monitoring active feed</p>
          </div>
        </div>

        <div className="generation-player-center">
          <span>02:44</span>
          <div className="generation-player-track">
            <i />
          </div>
          <span>04:00</span>
        </div>

        <div className="generation-player-right">
          <span>⌁</span>
          <div className="generation-volume-track">
            <i />
          </div>
          <button type="button" aria-label="Stats">
            ▥
          </button>
          <button type="button" aria-label="Fullscreen">
            ⤢
          </button>
        </div>
      </footer>
    </section>
  );
}
