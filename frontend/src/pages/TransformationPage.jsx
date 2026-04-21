const sourceTags = [
  { label: "Birds", tone: "cyan" },
  { label: "Wind", tone: "mint" },
  { label: "Anthropogenic noise", tone: "rose" },
];

const seasons = ["Spring", "Early Summer", "Late Summer", "Autumn"];
const inputWave = [20, 42, 68, 36, 82, 28, 56, 74, 34, 62, 25, 50, 70, 30, 58, 76, 39, 61, 31, 47, 66, 41, 59, 24, 52, 72];
const outputWave = [18, 35, 61, 27, 73, 31, 44, 80, 38, 67, 22, 54, 86, 33, 49, 78, 43, 71, 29, 57, 83, 36, 64, 25, 46, 75];

export default function TransformationPage() {
  return (
    <section className="transform-page">
      <header className="transform-topbar">
        <p className="eyebrow">SONIC LABORATORY</p>

        <div className="transform-tools">
          <label className="search-box transform-search">
            <span>⌕</span>
            <input type="text" placeholder="Search experiments..." />
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

      <div className="transform-header-row">
        <div>
          <p className="eyebrow transform-section-label">Source Input</p>
          <h1>Original Audio</h1>
        </div>
        <div className="transform-center-heading">
          <p className="eyebrow transform-section-label">Processor Hub</p>
          <h2>Transformation Controls</h2>
        </div>
        <div className="transform-right-heading">
          <p className="eyebrow transform-section-label">Target Generation</p>
          <h2>Modified Audio Output</h2>
        </div>
      </div>

      <div className="transform-grid">
        <section className="panel transform-column transform-source">
          <div className="transform-wave-frame">
            <div className="transform-waveform input" aria-hidden="true">
              {inputWave.map((bar, index) => (
                <span key={`input-${index}`} style={{ height: `${bar}%` }} />
              ))}
            </div>
            <div className="transform-badges">
              <span>44.1 KHZ</span>
              <span>24-BIT</span>
            </div>
          </div>

          <div className="transform-tag-head">
            <p>Acoustic Tags</p>
            <button type="button" aria-label="Add tag">
              +
            </button>
          </div>

          <div className="transform-tags">
            {sourceTags.map((tag) => (
              <span key={tag.label} className={`transform-tag ${tag.tone}`}>
                <i />
                {tag.label}
              </span>
            ))}
          </div>

          <div className="transform-source-player">
            <button type="button" className="transform-mini-play" aria-label="Play source audio">
              ▶
            </button>
            <div className="transform-mini-track">
              <span />
            </div>
            <p>02:14</p>
          </div>
        </section>

        <section className="panel transform-column transform-controls">
          <div className="control-block">
            <div className="control-heading">
              <p>Increase Wind Intensity</p>
              <strong>74%</strong>
            </div>
            <div className="control-slider">
              <span className="control-line" />
              <span className="control-knob" />
            </div>
          </div>

          <div className="control-block">
            <div className="control-heading">
              <p>Seasonal Simulation</p>
              <strong>Late Summer</strong>
            </div>
            <div className="season-grid">
              {seasons.map((season) => (
                <button
                  key={season}
                  type="button"
                  className={`season-pill${season === "Late Summer" ? " active" : ""}`}
                >
                  {season}
                </button>
              ))}
            </div>
          </div>

          <div className="control-block">
            <p className="control-label">Atmospheric Density</p>
            <div className="density-row">
              <span>☁</span>
              <div className="density-track">
                <span />
              </div>
              <span>☀</span>
            </div>
          </div>

          <div className="control-block upsampling-block">
            <div>
              <p className="control-label">Neural Upsampling</p>
              <small>Enable high-fidelity reconstruction</small>
            </div>
            <button type="button" className="toggle-switch active" aria-label="Enable neural upsampling">
              <span />
            </button>
          </div>

          <button type="button" className="transform-cta">
            ✦ Synthesize
          </button>
        </section>

        <section className="panel transform-column transform-output">
          <div className="transform-wave-frame">
            <div className="transform-live-pill">● Live render</div>
            <div className="transform-waveform output" aria-hidden="true">
              {outputWave.map((bar, index) => (
                <span key={`output-${index}`} style={{ height: `${bar}%` }} />
              ))}
            </div>
          </div>

          <div className="transform-meta-head">
            <p>Output Metadata</p>
          </div>

          <div className="transform-meta-grid">
            <article className="transform-meta-card">
              <span>Target ambience</span>
              <strong>Late Summer Boreal</strong>
            </article>
            <article className="transform-meta-card">
              <span>Noise reduction</span>
              <strong>-14.2 LUFS</strong>
            </article>
          </div>

          <div className="transform-output-footer">
            <div className="transform-output-actions">
              <button type="button" className="transform-icon-chip" aria-label="Download output">
                ↓
              </button>
              <button type="button" className="transform-icon-chip" aria-label="Share output">
                ↗
              </button>
            </div>

            <div className="transform-file-size">
              <span>Est. file size</span>
              <strong>42.8 MB</strong>
            </div>
          </div>
        </section>
      </div>

      <footer className="panel transport-bar">
        <div className="transport-session">
          <div className="transport-art">≋</div>
          <div>
            <strong>Session Active</strong>
            <p>AMAZONAS_RAINFOREST_042.WAV</p>
          </div>
        </div>

        <div className="transport-controls">
          <button type="button" aria-label="Shuffle">
            ⇱
          </button>
          <button type="button" aria-label="Previous">
            ◀
          </button>
          <button type="button" className="primary" aria-label="Play transport">
            ▶
          </button>
          <button type="button" aria-label="Next">
            ▶
          </button>
          <button type="button" aria-label="Repeat">
            ↻
          </button>
        </div>

        <div className="transport-timeline">
          <span>00:42</span>
          <div className="transport-track">
            <i />
          </div>
          <span>03:15</span>
        </div>

        <div className="transport-volume">
          <span>⌁</span>
          <div className="transport-volume-track">
            <i />
          </div>
          <button type="button" aria-label="Fullscreen">
            □
          </button>
        </div>
      </footer>
    </section>
  );
}
