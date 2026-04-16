const systemStats = [
  { label: "Processing load", value: "42.8 TFLOPS" },
  { label: "Signal latency", value: "0.02ms" },
  { label: "Active threads", value: "14,202" },
];

const workflows = [
  {
    icon: "▣",
    title: "Analysis",
    copy:
      "Deconstruct recordings into granular data. Extract frequency maps, emotional vectors, and semantic metadata.",
    action: "Initialize scan",
  },
  {
    icon: "✦",
    title: "Generation",
    copy:
      "Synthesize infinite landscapes from variables. Create non-repeating audio based on biome and atmosphere parameters.",
    action: "New synthesis",
    featured: true,
  },
  {
    icon: "≋",
    title: "Transformation",
    copy:
      "Modify existing audio through spatialization and timbre morphing. Apply AI-driven acoustic re-texturing.",
    action: "Begin morph",
  },
];

const recentProjects = [
  {
    name: "Boreal Forest Alpha",
    type: "Generation",
    detail: "V54.2 Synthesis",
    modified: "12 Oct 2023",
    time: "14:22 GMT",
    profile: [18, 34, 52, 28, 64, 40, 22, 46],
  },
  {
    name: "Sub-Station Resonance",
    type: "Analysis",
    detail: "Spectral Mapping",
    modified: "11 Oct 2023",
    time: "09:15 GMT",
    profile: [10, 14, 22, 18, 29, 12, 15, 9],
  },
  {
    name: "Oceanic Depth Morph",
    type: "Transformation",
    detail: "Tidal Transform",
    modified: "09 Oct 2023",
    time: "21:40 GMT",
    profile: [14, 26, 30, 42, 50, 22, 28, 34],
  },
];

export default function AboutPage() {
  return (
    <section className="system-page">
      <header className="system-topbar">
        <div className="system-wordmark">
          <p>SONIC LABORATORY</p>
          <nav className="system-tabs" aria-label="Section navigation">
            <a href="#network" className="active">
              Network
            </a>
            <a href="#resources">Resources</a>
            <a href="#archive">Archive</a>
          </nav>
        </div>

        <div className="system-toolbar">
          <label className="system-search">
            <span>⌕</span>
            <input type="text" placeholder="Search signals..." />
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

      <section className="system-hero panel" id="network">
        <p className="eyebrow system-version">VERSION 4.2 // STABLE</p>
        <h1>AI Environmental Audio Engine</h1>
        <p className="system-lead">
          Unlock the architectural potential of sound. Our neural synthesis engine
          allows researchers and artists to analyze, generate, and transform
          acoustic environments with unprecedented precision.
        </p>

        <div className="system-stat-grid">
          {systemStats.map((stat) => (
            <article key={stat.label} className="system-stat panel">
              <span>{stat.label}</span>
              <strong>{stat.value}</strong>
            </article>
          ))}
        </div>
      </section>

      <section className="system-section" id="resources">
        <div className="system-section-heading">
          <p className="eyebrow">SELECT WORKFLOW</p>
          <span>Core pipelines</span>
        </div>

        <div className="workflow-grid">
          {workflows.map((workflow) => (
            <article
              key={workflow.title}
              className={`workflow-card panel${workflow.featured ? " featured" : ""}`}
            >
              <div className="workflow-icon">{workflow.icon}</div>
              <h2>{workflow.title}</h2>
              <p>{workflow.copy}</p>
              <button type="button" className="workflow-action">
                {workflow.action}
              </button>
            </article>
          ))}
        </div>
      </section>

      <section className="system-section" id="archive">
        <div className="system-section-heading">
          <p className="eyebrow">RECENT PROJECTS</p>
          <a href="#library">View library</a>
        </div>

        <div className="project-table panel">
          <div className="project-table-head">
            <span>Project identity</span>
            <span>Type</span>
            <span>Signal profile</span>
            <span>Last modified</span>
          </div>

          <div className="project-table-body">
            {recentProjects.map((project) => (
              <article key={project.name} className="project-row">
                <div className="project-primary">
                  <button type="button" className="project-play" aria-label={`Open ${project.name}`}>
                    ▶
                  </button>
                  <div>
                    <strong>{project.name}</strong>
                    <p>{project.detail}</p>
                  </div>
                </div>

                <div className="project-type">
                  <span>{project.type}</span>
                </div>

                <div className="project-profile" aria-hidden="true">
                  {project.profile.map((bar, index) => (
                    <i key={`${project.name}-${index}`} style={{ height: `${bar}px` }} />
                  ))}
                </div>

                <div className="project-meta">
                  <strong>{project.modified}</strong>
                  <p>{project.time}</p>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>
    </section>
  );
}
