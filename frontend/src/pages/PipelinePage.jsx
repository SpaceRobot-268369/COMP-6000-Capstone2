import { useState } from "react";

// Status → pill label + colour class (shares the intro-page status classes).
const STATUS_META = {
  live: { label: "● Live", cls: "status-live" },
  partial: { label: "◑ Partial", cls: "status-partial" },
  smoke: { label: "◑ Smoke test", cls: "status-partial" },
  placeholder: { label: "○ Placeholder", cls: "status-coming" },
};

// ── Generation: forward / additive. env request → A+B+C in parallel → mixer → out.
const GENERATION = {
  source: {
    icon: "✦",
    label: "Environmental request",
    sub: "seed · season × diel · weather params",
  },
  parallelHeading: "Three independent layers compose in parallel",
  parallelNote:
    "Each layer owns one acoustic role and must not bleed into another — mixing events into the bed double-counts them and breaks layer separation.",
  layers: [
    {
      step: "A",
      label: "Ambient bed",
      role: "Continuous site texture — insects, low-level tone. Carries no events.",
      model: "AudioLDM2 LoRA · per-cell bank (16 season×diel)",
      status: "live",
    },
    {
      step: "B",
      label: "Weather",
      role: "Wind / rain / thunder, mixed from curated assets by intensity.",
      model: "Curated assets · parameter mixing",
      status: "placeholder",
    },
    {
      step: "C",
      label: "Events",
      role: "Species calls made plausible for the requested time & season.",
      model: "AudioGen LoRA per species · 16 kHz",
      status: "smoke",
    },
  ],
  merge: {
    step: "D",
    label: "Mixer",
    role: "Stacks A+B+C — sample-rate match, gain staging, fades.",
    model: "Algorithmic combiner",
    status: "placeholder",
    modelLabel: "Method",
  },
  output: {
    icon: "♪",
    label: "Soundscape + explanation",
    sub: "WAV · spectrogram · per-layer reasoning JSON",
  },
};

// ── Analysis: NOT reverse generation. raw mixture → 3 parallel heads → fuse → report.
const ANALYSIS = {
  source: {
    icon: "◫",
    label: "Uploaded recording",
    sub: "any ecoacoustic WAV / FLAC — the raw mixture",
  },
  pre: {
    icon: "≡",
    label: "Shared preprocess",
    sub: "mel spectrogram + waveform, computed once",
  },
  parallelHeading: "Three detector heads read the raw mixture in parallel",
  parallelNote:
    "No decomposer, no separated stems. Each head answers one question on the full mixture — pre-trained detectors were built to find their signal in the presence of everything else.",
  layers: [
    {
      step: "E-A",
      label: "Ambient context",
      role: "k-NN against the learned latent index — 'what kind of bed is this?'",
      model: "CLAP embedding · similarity",
      status: "partial",
    },
    {
      step: "E-B",
      label: "Weather",
      role: "Wind / rain / thunder intensity directly from the spectrum.",
      model: "PANNs tagger · zero-shot",
      status: "placeholder",
    },
    {
      step: "E-C",
      label: "Events",
      role: "Species present + onsets — the strongest season/diel signal.",
      model: "BirdNET + CLAP fallback",
      status: "placeholder",
    },
  ],
  merge: {
    step: "Σ",
    label: "Aggregator",
    role: "Fuses latent context (season / diel) from per-head evidence; records disagreements.",
    model: "Deterministic fusion",
    status: "placeholder",
    modelLabel: "Method",
  },
  postMerge: {
    step: "LLM OSS",
    label: "Narration layer",
    role: "Writes the human-readable report from the aggregator record; it does not inspect audio or override detector evidence.",
    model: "Open-source LLM · report policy",
    status: "placeholder",
  },
  output: {
    icon: "✎",
    label: "Report package",
    sub: "observations · inferred context · evidence-backed narration",
  },
};

function StatusPill({ status }) {
  const meta = STATUS_META[status] ?? STATUS_META.placeholder;
  return <span className={`intro-wf-status ${meta.cls}`}>{meta.label}</span>;
}

function FlowNode({ node, variant = "" }) {
  return (
    <div className={`flow-node ${variant}`}>
      <span className="flow-node-icon">{node.icon}</span>
      <div className="flow-node-body">
        <strong>{node.label}</strong>
        <span>{node.sub}</span>
      </div>
    </div>
  );
}

function LayerCard({ layer, index = 0, variant = "" }) {
  return (
    <article
      className={`flow-layer-card panel ${variant}`}
      style={{ "--i": index }}
    >
      <div className="flow-layer-top">
        <span className="flow-layer-step">{layer.step}</span>
        <StatusPill status={layer.status} />
      </div>
      <strong className="flow-layer-label">{layer.label}</strong>
      <div className="flow-layer-facts">
        <div className="flow-layer-fact">
          <span>Role</span>
          <p>{layer.role}</p>
        </div>
        <div className="flow-layer-fact">
          <span>{layer.modelLabel || "Model"}</span>
          <p>{layer.model}</p>
        </div>
      </div>
    </article>
  );
}

function FlowDiagram({ spec }) {
  return (
    <div className="flow-diagram">
      <FlowNode node={spec.source} variant="flow-node-source" />
      <div className="flow-conn" aria-hidden="true" />

      {spec.pre ? (
        <>
          <FlowNode node={spec.pre} variant="flow-node-pre" />
          <div className="flow-conn" aria-hidden="true" />
        </>
      ) : null}

      {/* fan-out 1 → 3 */}
      <div className="flow-fan flow-fan-out" aria-hidden="true">
        <i className="flow-drop" />
        <i className="flow-drop" />
        <i className="flow-drop" />
      </div>

      <div className="flow-parallel-head">
        <p className="flow-parallel-title">{spec.parallelHeading}</p>
        <p className="flow-parallel-note">{spec.parallelNote}</p>
      </div>

      <div className="flow-layers">
        {spec.layers.map((l, i) => (
          <LayerCard key={l.step} layer={l} index={i} />
        ))}
      </div>

      {/* merge 3 → 1 */}
      <div className="flow-fan flow-fan-merge" aria-hidden="true">
        <i className="flow-drop" />
        <i className="flow-drop" />
        <i className="flow-drop" />
      </div>
      <div className="flow-conn" aria-hidden="true" />

      <div className="flow-merge-stack">
        <LayerCard layer={spec.merge} variant="flow-layer-card-merge" />
        <div className="flow-merge-wave" aria-hidden="true">
          {Array.from({ length: 7 }).map((_, i) => (
            <span key={i} style={{ "--b": i }} />
          ))}
        </div>
      </div>

      {spec.postMerge ? (
        <>
          <div className="flow-conn" aria-hidden="true" />
          <LayerCard layer={spec.postMerge} variant="flow-layer-card-llm" />
        </>
      ) : null}

      <div className="flow-conn" aria-hidden="true" />
      <FlowNode node={spec.output} variant="flow-node-output" />
    </div>
  );
}

const MODES = {
  generation: {
    key: "generation",
    tab: "Generation",
    caption: "Forward · additive",
    spec: GENERATION,
  },
  analysis: {
    key: "analysis",
    tab: "Analysis",
    caption: "Parallel detection · not reverse generation",
    spec: ANALYSIS,
  },
};

export default function PipelinePage() {
  const [mode, setMode] = useState("generation");
  const active = MODES[mode];

  return (
    <section className="generation-page pipeline-page">
      {/* ── Topbar (matches analysis/generation pages) ── */}
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">HOW IT WORKS</p>
          <span>Layered pipeline · signal flow</span>
        </div>
      </header>

      {/* ── Mode toggle + animated diagram ── */}
      <div className="pipeline-content">
        <main className="panel dev-result-card pipeline-diagram-card">
          <div className="generation-card-head">
            <h2>Signal Flow</h2>
            <p>{active.caption}</p>
          </div>

          <div className="pipeline-diagram-body">
            <div className="flow-mode-toggle" role="tablist" aria-label="Pipeline mode">
              {Object.values(MODES).map((m) => (
                <button
                  key={m.key}
                  type="button"
                  role="tab"
                  aria-selected={mode === m.key}
                  className={`flow-mode-tab${mode === m.key ? " active" : ""}`}
                  onClick={() => setMode(m.key)}
                >
                  {m.tab}
                </button>
              ))}
            </div>

            <div className="flow-stage" key={mode}>
              <FlowDiagram spec={active.spec} />
            </div>
          </div>
        </main>
      </div>
    </section>
  );
}
