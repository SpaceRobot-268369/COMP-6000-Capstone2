import { Link } from "react-router-dom";

const workflows = [
  {
    icon: "◫",
    title: "Analysis",
    to: "/",
    tagline: "Decode what's in a recording",
    description:
      "Upload any ecoacoustic recording. The model encodes it into a latent representation and returns spectral metrics — avian biophony, anthropogenic interference, insect geophony, acoustic diversity index, and soundscape complexity. The raw spectrogram is computed in-browser from the audio file without needing a server.",
    steps: ["Upload a WAV or FLAC file", "Model encodes audio → latent vector", "View spectrogram, metrics, summary"],
    status: "live",
  },
  {
    icon: "✦",
    title: "Generation",
    to: "/generation",
    tagline: "Synthesise a soundscape from conditions",
    description:
      "Set environmental conditions — temperature, humidity, wind speed, precipitation, month, and time of day — and the model generates a speculative soundscape that would plausibly occur under those conditions. Useful for exploring how climate change or monthly shifts might alter acoustic environments.",
    steps: ["Set environmental conditions", "Model maps conditions → latent z", "Decoder reconstructs spectrogram → audio"],
    status: "live",
    featured: true,
  },
  {
    icon: "≋",
    title: "Transformation",
    to: "/transformation",
    tagline: "Shift an existing recording to new conditions",
    description:
      "Upload a source recording then dial in a new set of environmental conditions. The model encodes the source audio, adjusts the latent representation toward the target conditions, and decodes a transformed soundscape. Hear what the same site might sound like in a different month or under a changing climate.",
    steps: ["Upload source audio", "Set target env conditions", "Encode → adjust z → decode → output audio"],
    status: "live",
  },
];

const dataStats = [
  { label: "Recordings sampled", value: "287" },
  { label: "Audio clips", value: "6,148" },
  { label: "Hours of audio", value: "~510 h" },
  { label: "Env features per clip", value: "29" },
  { label: "Site", value: "Bowra, QLD" },
  { label: "Years covered", value: "2019–2025" },
];

const pipeline = [
  { step: "01", label: "Raw Audio", detail: "FLAC recordings, 22 050 Hz mono" },
  { step: "02", label: "Mel Spectrogram", detail: "128 mel bins, 30 s crop per step" },
  { step: "03", label: "CNN Encoder", detail: "4-block conv → 512-dim embedding" },
  { step: "04", label: "Fusion MLP", detail: "audio emb + 29 env features → 256-dim latent z" },
  { step: "05", label: "Decoder", detail: "Transposed conv → reconstructed spectrogram" },
];

export default function AboutPage() {
  return (
    <section className="intro-page">
      {/* ── Hero ── */}
      <header className="intro-hero panel">
        <div className="intro-hero-text">
          <p className="eyebrow intro-eyebrow">SONIC LABORATORY — RESEARCH PROTOTYPE</p>
          <h1 className="intro-hero-title">Speculative Soundscape Generation</h1>
          <p className="intro-hero-lead">
            An AI system that learns relationships between environmental conditions and
            ecoacoustic recordings, then generates or transforms soundscapes under new
            ecological conditions. Built on 510 hours of field recordings from a single
            semi-arid site in Queensland, Australia.
          </p>
          <div className="intro-hero-links">
            <Link to="/" className="intro-cta-primary">Start Analysis →</Link>
            <Link to="/generation" className="intro-cta-ghost">Try Generation →</Link>
          </div>
        </div>
        <div className="intro-hero-art" aria-hidden="true">
          <div className="intro-wave intro-wave-a" />
          <div className="intro-wave intro-wave-b" />
          <div className="intro-wave intro-wave-c" />
          <span className="intro-node intro-node-1">● env conditioning</span>
          <span className="intro-node intro-node-2">● latent z = 256 dim</span>
        </div>
      </header>

      {/* ── Three workflows ── */}
      <section className="intro-section">
        <div className="intro-section-head">
          <p className="eyebrow">THREE MODES</p>
          <p className="intro-section-sub">Each mode uses the same trained model in a different direction.</p>
        </div>
        <div className="intro-workflow-grid">
          {workflows.map((w) => (
            <article key={w.title} className={`intro-workflow-card panel${w.featured ? " featured" : ""}`}>
              <div className="intro-wf-top">
                <div className="intro-wf-icon">{w.icon}</div>
                <span className={`intro-wf-status status-${w.status}`}>
                  {w.status === "live" ? "● Live" : "Coming soon"}
                </span>
              </div>
              <h2 className="intro-wf-title">{w.title}</h2>
              <p className="intro-wf-tagline">{w.tagline}</p>
              <p className="intro-wf-desc">{w.description}</p>
              <ol className="intro-wf-steps">
                {w.steps.map((s, i) => (
                  <li key={i}><span>{i + 1}</span>{s}</li>
                ))}
              </ol>
              <Link to={w.to} className={`intro-wf-link${w.featured ? " primary" : ""}`}>
                Open {w.title} →
              </Link>
            </article>
          ))}
        </div>
      </section>

      {/* ── AI pipeline ── */}
      <section className="intro-section">
        <div className="intro-section-head">
          <p className="eyebrow">MODEL ARCHITECTURE</p>
          <p className="intro-section-sub">Environmental Autoencoder — Module A of the three-stage AI plan.</p>
        </div>
        <div className="intro-pipeline panel">
          {pipeline.map((p, i) => (
            <div key={p.step} className="intro-pipe-step">
              <div className="intro-pipe-num">{p.step}</div>
              <div className="intro-pipe-body">
                <strong>{p.label}</strong>
                <p>{p.detail}</p>
              </div>
              {i < pipeline.length - 1 && <div className="intro-pipe-arrow">→</div>}
            </div>
          ))}
        </div>
      </section>

      {/* ── Dataset ── */}
      <section className="intro-section">
        <div className="intro-section-head">
          <p className="eyebrow">TRAINING DATASET</p>
          <p className="intro-section-sub">
            Site 257 — Bowra Wildlife Sanctuary, QLD (Australian Acoustic Observatory).
          </p>
        </div>
        <div className="intro-data-grid">
          {dataStats.map((s) => (
            <article key={s.label} className="panel intro-stat-card">
              <span>{s.label}</span>
              <strong>{s.value}</strong>
            </article>
          ))}
        </div>
      </section>
    </section>
  );
}
