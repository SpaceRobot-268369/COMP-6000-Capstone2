import { useState } from "react";

// ── Generation: forward / additive. env request → parser → A+B+C in parallel → mixer → out.
const GENERATION = {
  source: {
    icon: "✦",
    label: "Environmental request",
    sub: "raw text prompt · seed",
  },
  pre: {
    step: "LLM OSS",
    label: "Prompt Parser",
    role: "One in-process LLM call: pre-fills defaults, runs the correct-and-continue coherence gate, then decodes the request into per-layer contracts — returning a parse result (ok / corrected / rejected) with the defaults it filled.",
    model: "In-process LLM-OSS · prompt parser policy",
    status: "partial",
    visual: "llm",
  },
  // What the parser does — shown beside the Prompt Parser node.
  decoder: {
    why: "Generation is split into independent, modular layers, so a raw natural-language prompt can't be fed straight into any one layer's model — and most prompts under-specify. The Prompt Parser is an LLM-OSS layer, governed by a written policy, that does three things before any layer runs: (1) pre-fills sensible defaults for anything you didn't say — the ambient bed is always on, but there's no rain unless you ask and no fauna unless you name it; (2) validates coherence — and corrects rather than fails: an out-of-domain or phenologically implausible request (dense city traffic, or a species that doesn't occur in the requested season) is swapped for the nearest plausible scene and the change is explained; only genuinely unrecoverable prompts are rejected with a suggested alternative; (3) decodes the completed, validated request into the three aligned inputs each layer expects — Layer A a (season, diel) cell, Layer B structured weather JSON, Layer C a species checklist. The parser also resolves the arrangement: cadence like “a boobook every few seconds” is expanded into an explicit list of onset times. Those placement values — onsets, per-clip gains, continuous-vs-discrete flags — don't go to Layers B or C; they travel straight to the mixer (the only stage that places audio in time), so the mixer never has to reason about frequency, it just places clips where it's told. The whole call returns one parse result — ok, corrected, or rejected — so the UI can show exactly what was assumed.",
    example: "“A misty autumn dawn, light rain, with a boobook owl calling in the distance.”",
  },
  // Stage-2 coherence gate: every prompt resolves to one of three outcomes.
  gate: {
    lead: "Coherence gate — every prompt resolves to one of three outcomes",
    outcomes: [
      {
        tag: "ok",
        title: "Valid",
        body: "Complete and in-domain after defaults are filled — passed through unchanged.",
      },
      {
        tag: "corrected",
        title: "Partly invalid → corrected + recommended",
        body: "Implausible parts (a species out of its season, thunder, a near-miss weather term) are swapped for the nearest plausible scene; the parser proposes a recommended prompt, explains the change, and generation continues.",
      },
      {
        tag: "rejected",
        title: "Fully invalid → rejected",
        body: "Unrecoverable, wholly out-of-domain requests (city traffic, music, snow at an arid site) are rejected with a suggested alternative — no layers run.",
      },
    ],
  },
  // Stage-3 output ①: the distinct contract the parser hands each layer.
  contracts: [
    { step: "A", label: "Ambient", contract: "(season, diel) cell" },
    { step: "B", label: "Weather", contract: "weather JSON · type · intensity · duration" },
    { step: "C", label: "Events", contract: "species checklist · density" },
  ],
  parallelHeading: "Three independent layers compose in parallel",
  parallelNote:
    "Each layer owns one acoustic role and must not bleed into another — mixing events into the bed double-counts them and breaks layer separation. Layer A returns exactly one bed; Layers B and C each return 0–K clips (no weather, or a few gusts; no fauna, or many calls). Those clips carry audio only — the parser's placement parameters reach the mixer separately.",
  layers: [
    {
      step: "A",
      label: "Ambient bed",
      role: "Continuous site texture. Ingests decoded (season, diel) cell metadata.",
      model: "AudioLDM2 LoRA · per-cell bank (16 season×diel)",
      status: "live",
      outCount: "×1",
      sample: {
        input: '{ "seed": 42, "season": "autumn", "diel": "dawn" }',
        output: "Continuous ambient bed — insects, low foliage rustle, distant site tone. No discrete events. WAV + mel spectrogram.",
      },
    },
    {
      step: "B",
      label: "Weather",
      role: "Wind / rain. Ingests decoded weather type, intensity, and duration; selects from curated site-only intensity banks.",
      model: "Curated wind/rain banks + stem selector · retrieval",
      status: "mvp",
      outCount: "0–K",
      sample: {
        input: '{ "weather_type": "rain", "intensity": "medium", "duration_s": 10, "retrieval_seed": 42 }',
        output: "0–K weather clips for the mixer — continuous beds (wind, steady rain) loop to length; discrete claps (thunder) are placed at parser-supplied onsets. Loudness-normalised WAV.",
      },
    },
    {
      step: "C",
      label: "Events",
      role: "Species calls. Ingests checklist of plausible callers for requested time/season.",
      model: "Audited retrieval library (default) · AudioGen LoRA (smoke)",
      status: "demo",
      outCount: "0–K",
      sample: {
        input: '{ "seed": 42 }  // boobook LoRA — one species, server owns the caption',
        output: "0–K species clips — each named caller hands the mixer one clip; the parser supplies the onset times for its calls. WAV (16 kHz, resampled at the mixer).",
      },
    },
  ],
  // Labels for the per-layer input/output port popovers.
  portMeta: {
    in: { title: "Sample prompt · input", field: "Decoded input", code: true },
    out: { title: "Sample output", field: "Output", code: false },
  },
  merge: {
    step: "D",
    label: "Mixer",
    role: "Takes two inputs: the audio — A's single bed plus 0–K weather and 0–K event clips from B/C — and the placement parameters the parser computed (onsets, per-clip gains, continuous flags). Arranges them on one timeline: the ambient bed and any continuous weather loop to length; discrete clips drop at their onset times. Overlaps are summed, never resolved. Per-layer gain staging (ambient 0 / weather −2 / event −8 dB) with per-clip overrides, 0.95 peak ceiling, then export. A null onset list falls back to a seeded random placement.",
    model: "Multi-clip algorithmic combiner · seeded onset fallback · implemented + tested",
    status: "live",
    modelLabel: "Method",
  },
  output: {
    icon: "♪",
    label: "Soundscape + explanation",
    sub: "WAV · spectrogram · per-layer reasoning JSON",
  },
  // The parser's second output: placement parameters that skip B/C and go
  // straight to the mixer. Rendered as a labeled bypass line (parser → mixer).
  paramBus: "onsets · per-clip gain · continuous flags — computed by the parser, never seen by B/C",
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
      sample: {
        input: "Mel spectrogram + waveform of the uploaded mixture (shared preprocess).",
        output: "Nearest ambient cell + similarity — e.g. autumn · dawn bed (cosine 0.78).",
      },
    },
    {
      step: "E-B",
      label: "Weather",
      role: "Wind / rain / thunder intensity directly from the spectrum.",
      model: "PANNs tagger · zero-shot",
      status: "placeholder",
      sample: {
        input: "Mel spectrogram of the uploaded mixture (shared preprocess).",
        output: "Weather tags + intensity — e.g. light rain 0.62, no wind.",
      },
    },
    {
      step: "E-C",
      label: "Events",
      role: "Species present + onsets — the strongest season/diel signal.",
      model: "BirdNET + CLAP fallback",
      status: "placeholder",
      sample: {
        input: "Waveform of the uploaded mixture (shared preprocess).",
        output: "Species + onsets — e.g. southern boobook ×3 at 22:14, 22:31, 23:02.",
      },
    },
  ],
  portMeta: {
    in: { title: "Sample input", field: "Reads from", code: false },
    out: { title: "Sample output", field: "Detector result", code: false },
  },
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

// Status → visible badge. Maps the registry-ish status strings used in the
// specs above to a short label + a tone class (CSS colours the chip).
const STATUS_META = {
  live: { label: "Live", tone: "live" },
  production: { label: "Live", tone: "live" },
  demo: { label: "Demo-ready", tone: "live" },
  mvp: { label: "MVP", tone: "mvp" },
  partial: { label: "Partial", tone: "partial" },
  smoke: { label: "Smoke", tone: "partial" },
  placeholder: { label: "Placeholder", tone: "placeholder" },
};

function StatusBadge({ status }) {
  if (!status) return null;
  const meta = STATUS_META[status] || { label: status, tone: "placeholder" };
  return (
    <span className={`flow-status flow-status-${meta.tone}`}>{meta.label}</span>
  );
}

function FlowNode({ node, variant = "" }) {
  return (
    <div className={`flow-node ${variant}`}>
      <span className="flow-node-icon">{node.icon}</span>
      <div className="flow-node-body">
        <strong>{node.label}</strong>
        <span>{node.sub}</span>
      </div>
      {node.tag ? <span className="flow-node-tag">{node.tag}</span> : null}
    </div>
  );
}

// An input/output port: a circle that reveals its sample on hover, and
// pins it open on click (so it stays after the pointer leaves).
function SamplePort({ kind, glyph, title, fieldLabel, value, isCode, count }) {
  const [pinned, setPinned] = useState(false);
  // A count badge marks how many clips this port emits. "0–K" (variable,
  // prompt-dependent) is highlighted so it reads differently from a fixed "×1".
  const variable = Boolean(count) && /k/i.test(count);
  return (
    <span className={`flow-port flow-port-${kind}${pinned ? " is-pinned" : ""}`}>
      <button
        type="button"
        className="flow-port-dot"
        aria-expanded={pinned}
        aria-label={`${pinned ? "Hide" : "Show"} ${title}${count ? ` — emits ${count} clips` : ""}`}
        onClick={() => setPinned((p) => !p)}
      >
        <i>{glyph}</i>
        {count ? (
          <span
            className={`flow-port-count${variable ? " flow-port-count-var" : ""}`}
            aria-hidden="true"
          >
            {count}
          </span>
        ) : null}
      </button>
      <span className="flow-port-pop" role="tooltip">
        <span className="flow-port-pop-title">{title}</span>
        <span className="flow-port-pop-field">{fieldLabel}</span>
        {isCode ? (
          <code className="flow-port-pop-code">{value}</code>
        ) : (
          <span className="flow-port-pop-text">{value}</span>
        )}
      </span>
    </span>
  );
}

function LayerCard({ layer, index = 0, variant = "", ports = false, portMeta }) {
  // Ports (and their hover/click samples) only exist where a layer declares a
  // sample and the mode supplies port labels (the parallel layers of both modes).
  const showPorts = ports && Boolean(layer.sample) && Boolean(portMeta);
  return (
    <article
      className={`flow-layer-card panel ${showPorts ? "flow-has-ports" : ""} ${variant}`}
      style={{ "--i": index }}
    >
      {showPorts ? (
        <SamplePort
          kind="in"
          glyph="in"
          title={portMeta.in.title}
          fieldLabel={portMeta.in.field}
          value={layer.sample.input}
          isCode={portMeta.in.code}
        />
      ) : null}
      <div className="flow-layer-top">
        <span className="flow-layer-step">{layer.step}</span>
        <StatusBadge status={layer.status} />
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
      {showPorts ? (
        <SamplePort
          kind="out"
          glyph="out"
          title={portMeta.out.title}
          fieldLabel={portMeta.out.field}
          value={layer.sample.output}
          isCode={portMeta.out.code}
          count={layer.outCount}
        />
      ) : null}
    </article>
  );
}

function FlowDiagram({ spec }) {
  const preNode = spec.pre
    ? spec.pre.visual === "llm"
      ? <LayerCard layer={spec.pre} variant="flow-layer-card-llm flow-layer-card-prompt-parser" />
      : <FlowNode node={spec.pre} variant="flow-node-pre" />
    : null;

  return (
    <div className="flow-diagram">
      <FlowNode node={spec.source} variant="flow-node-source" />
      <div className="flow-conn" aria-hidden="true" />

      {spec.pre ? (
        <>
          {preNode}
          {spec.decoder ? (
            <div className="flow-decoder-why">
              <p className="flow-decoder-why-lead">What the Prompt Parser does</p>
              <p className="flow-decoder-why-text">{spec.decoder.why}</p>
              {spec.decoder.example ? (
                <p className="flow-decoder-why-eg">
                  <span>Example prompt</span>
                  {spec.decoder.example}
                </p>
              ) : null}
            </div>
          ) : null}
          {spec.gate ? (
            <div className="flow-gate">
              <p className="flow-gate-lead">{spec.gate.lead}</p>
              <div className="flow-gate-grid">
                {spec.gate.outcomes.map((o) => (
                  <div key={o.tag} className={`flow-gate-card flow-gate-${o.tag}`}>
                    <span className="flow-gate-tag">{o.title}</span>
                    <p>{o.body}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
          {spec.paramBus ? (
            <div className="flow-parambus flow-parambus-out">
              <span className="flow-parambus-tag">②</span>
              <div className="flow-parambus-body">
                <strong>Placement params → Mixer (D)</strong>
                <span>{spec.paramBus}</span>
              </div>
              <span className="flow-parambus-arrow" aria-hidden="true">↘</span>
            </div>
          ) : null}
          <div className="flow-conn" aria-hidden="true" />
        </>
      ) : null}

      {/* fan-out 1 → 3 — output ① of the parser: the per-layer contracts */}
      <div className="flow-fan flow-fan-out" aria-hidden="true">
        <i className="flow-drop" />
        <i className="flow-drop" />
        <i className="flow-drop" />
      </div>

      {spec.contracts ? (
        <div className="flow-contractbus">
          <span className="flow-contractbus-tag">①</span>
          <div className="flow-contractbus-body">
            <strong>Layer contracts → A · B · C</strong>
            <p className="flow-contractbus-sub">
              The parser decodes one prompt into a distinct contract per layer —
              each layer only ever sees its own.
            </p>
            <ul className="flow-contractbus-list">
              {spec.contracts.map((c) => (
                <li key={c.step}>
                  <span className="flow-contractbus-step">{c.step}</span>
                  <span className="flow-contractbus-layer">{c.label}</span>
                  <code>{c.contract}</code>
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : null}

      <div className="flow-parallel-head">
        <p className="flow-parallel-title">{spec.parallelHeading}</p>
        <p className="flow-parallel-note">{spec.parallelNote}</p>
      </div>

      <div className="flow-layers">
        {spec.layers.map((l, i) => (
          <LayerCard key={l.step} layer={l} index={i} ports portMeta={spec.portMeta} />
        ))}
      </div>

      {/* merge 3 → 1 */}
      <div className="flow-fan flow-fan-merge" aria-hidden="true">
        <i className="flow-drop" />
        <i className="flow-drop" />
        <i className="flow-drop" />
      </div>
      <div className="flow-conn" aria-hidden="true" />

      {spec.paramBus ? (
        <div className="flow-parambus flow-parambus-in">
          <span className="flow-parambus-arrow" aria-hidden="true">↳</span>
          <span className="flow-parambus-tag">②</span>
          <div className="flow-parambus-body">
            <strong>Placement params · from parser</strong>
            <span>second mixer input — arrives directly, not via B/C</span>
          </div>
        </div>
      ) : null}

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
