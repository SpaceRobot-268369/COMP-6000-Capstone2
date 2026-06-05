import { Link } from "react-router-dom";

// Status → pill label + colour class (reuses .intro-wf-status colour classes)
const STATUS_META = {
  live:        { label: "● Live",        cls: "status-live" },
  partial:     { label: "◑ Partial",     cls: "status-partial" },
  smoke:       { label: "◑ Smoke test",  cls: "status-partial" },
  placeholder: { label: "○ Placeholder", cls: "status-coming" },
  planned:     { label: "○ Planned",     cls: "status-coming" },
};

const workflows = [
  {
    icon: "◫",
    title: "Analysis",
    to: "/generation",
    tagline: "Decode what's in a recording",
    description:
      "Upload any ecoacoustic recording. Three detector heads run in parallel on the raw mixture — ambient context (k-NN against the learned latent index), weather intensity, and species/event onsets — then aggregate into a single report. The raw spectrogram is computed in-browser from the audio file without needing a server.",
    steps: [
      "Upload a WAV or FLAC file",
      "Ambient · weather · event heads run in parallel",
      "View spectrogram, estimated conditions, summary",
    ],
    status: "live",
  },
  {
    icon: "✦",
    title: "Generation",
    to: "/analysis",
    tagline: "Synthesise a soundscape from conditions",
    description:
      "Set environmental conditions and the model composes a speculative soundscape layer by layer: an ambient bed (Layer A), weather sounds (Layer B), and species events (Layer C), combined by a mixer (Layer D). Today the ambient bed is fully generative; the remaining layers are being built out.",
    steps: [
      "Set environmental conditions",
      "Layer A generates ambient bed; B/C add weather + events",
      "Mixer combines layers → audio + explanation",
    ],
    status: "partial",
    featured: true,
  },
  {
    icon: "≋",
    title: "Transformation",
    to: "/transformation",
    tagline: "Shift an existing recording to new conditions",
    description:
      "Upload a source recording then dial in a new set of environmental conditions. The model encodes the source into the VAE latent space, adjusts it toward the target conditions, and decodes a transformed soundscape. Hear what the same site might sound like in a different season or under a changing climate.",
    steps: [
      "Upload source audio",
      "Set target env conditions",
      "Encode → adjust z → decode → output audio",
    ],
    status: "planned",
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

// Five-layer composition (A–E). Detail = plain role + model sub-line.
const layers = [
  { step: "A", label: "Ambient bed", detail: "AudioLDM2 LoRA · per-cell bank (16 season×diel)", status: "live" },
  { step: "B", label: "Weather", detail: "Curated wind/rain assets · parameter mixing", status: "placeholder" },
  { step: "C", label: "Events", detail: "AudioGen LoRA per species · 16 kHz native", status: "smoke" },
  { step: "D", label: "Mixer", detail: "Combine A+B+C → WAV + explanation JSON", status: "placeholder" },
  { step: "E", label: "Analysis", detail: "Ambient similarity + weather + event detectors", status: "partial" },
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
            ecological conditions. Built as a <strong>layered composition</strong> — ambient
            bed, weather, and events mixed separately rather than one generated waveform — on
            510 hours of field recordings from a single semi-arid site in Queensland, Australia.
          </p>
          <div className="intro-hero-links">
            <Link to="/generation" className="intro-cta-primary">Start Analysis →</Link>
            <Link to="/analysis" className="intro-cta-ghost">Try Generation →</Link>
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
          <p className="intro-section-sub">Each mode orchestrates the same five-layer model stack in a different direction.</p>
        </div>
        <div className="intro-workflow-grid">
          {workflows.map((w) => {
            const meta = STATUS_META[w.status] ?? STATUS_META.planned;
            return (
              <article key={w.title} className={`intro-workflow-card panel${w.featured ? " featured" : ""}`}>
                <div className="intro-wf-top">
                  <div className="intro-wf-icon">{w.icon}</div>
                  <span className={`intro-wf-status ${meta.cls}`}>{meta.label}</span>
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
            );
          })}
        </div>
      </section>

      {/* ── Layered architecture (A–E) ── */}
      <section className="intro-section">
        <div className="intro-section-head">
          <p className="eyebrow">MODEL ARCHITECTURE</p>
          <p className="intro-section-sub">Five-layer composition — frozen base models + LoRA adapters, not a single waveform.</p>
        </div>
        <div className="intro-pipeline panel">
          {layers.map((l) => {
            const meta = STATUS_META[l.status] ?? STATUS_META.placeholder;
            return (
              <div key={l.step} className="intro-pipe-step">
                <div className="intro-pipe-num">{l.step}</div>
                <div className="intro-pipe-body">
                  <strong>{l.label}</strong>
                  <p>{l.detail}</p>
                  <span className={`intro-wf-status ${meta.cls}`}>{meta.label}</span>
                </div>
              </div>
            );
          })}
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
