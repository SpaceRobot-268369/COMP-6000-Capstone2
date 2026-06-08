import { useEffect, useMemo, useState } from "react";
import {
  fetchAttemptSamples,
  fetchLayerRegistry,
  generateAttempt,
  generateSoundscape,
  sampleWavUrl,
} from "../lib/api.js";

const DEFAULT_SEED = 42;
const DEFAULT_LAYER_C_SPECIES = "Splendid Fairywren";
const WEATHER_TYPE_OPTIONS = [
  { value: "rain", label: "rain" },
  { value: "wind", label: "wind" },
  { value: "rain+wind", label: "rain+wind" },
];
const WEATHER_INTENSITY_OPTIONS = [
  { value: "light", label: "light" },
  { value: "medium", label: "medium" },
  { value: "heavy", label: "heavy" },
];
const SEASON_OPTIONS = ["spring", "summer", "autumn", "winter"]
  .map((value) => ({ value, label: value }));
const DIEL_OPTIONS = ["dawn", "morning", "afternoon", "night"]
  .map((value) => ({ value, label: value }));

const GENERATION_LAYER_IDS = ["layer_a", "layer_b", "layer_c", "layer_d"];
const ANALYSIS_LAYER_IDS   = ["layer_e"];

export default function LayerATestPage({
  mode = "generation",
  includeLayers,
  eyebrow = "DEVELOPER TOOLS",
  title = "Layer / Attempt Dev Test",
} = {}) {
  const allowedLayerIds = useMemo(() => {
    if (includeLayers && includeLayers.length) return includeLayers;
    return mode === "analysis" ? ANALYSIS_LAYER_IDS : GENERATION_LAYER_IDS;
  }, [mode, includeLayers]);

  // Registry state
  const [registry, setRegistry] = useState(null);
  const [regError, setRegError] = useState("");
  const [layerId,   setLayerId]   = useState("");
  const [attemptId, setAttemptId] = useState("");

  // Generation state
  const [seed,     setSeed]     = useState(DEFAULT_SEED);
  const [season,   setSeason]   = useState("");        // bank attempts only
  const [diel,     setDiel]     = useState("");        // bank attempts only
  const [weatherType, setWeatherType] = useState("rain");
  const [weatherIntensity, setWeatherIntensity] = useState("medium");
  const [weatherDuration, setWeatherDuration] = useState(10);
  const [layerDIncludeWeather, setLayerDIncludeWeather] = useState(true);
  const [layerDIncludeEvents, setLayerDIncludeEvents] = useState(true);
  const [layerDAttempts, setLayerDAttempts] = useState({
    layer_a: "",
    layer_b: "",
    layer_c: "",
  });
  const [status,   setStatus]   = useState("idle");   // idle | loading | done | error
  const [result,   setResult]   = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [progress, setProgress] = useState(0);

  // Cached samples (loaded whenever the attempt changes)
  const [samples,    setSamples]    = useState(null);  // {expected:[…], showcase:[…], canonical_seed}
  const [samplesErr, setSamplesErr] = useState("");
  const [expectedKey, setExpectedKey] = useState(""); // "<tier>/<stem>"
  const [selectedSpecies, setSelectedSpecies] = useState("");

  // Load the layer registry once on mount.
  useEffect(() => {
    fetchLayerRegistry()
      .then((doc) => {
        const filtered = {
          ...doc,
          layers: (doc.layers || []).filter((l) => allowedLayerIds.includes(l.id)),
        };
        setRegistry(filtered);
        const firstLayer = filtered.layers?.[0];
        if (firstLayer) {
          setLayerId(firstLayer.id);
          setAttemptId(firstLayer.default || firstLayer.attempts?.[0]?.id || "");
        }
      })
      .catch((e) => setRegError(e.message));
  }, [allowedLayerIds]);

  // When the layer changes, snap the attempt to that layer's default.
  useEffect(() => {
    if (!registry || !layerId) return;
    const layer = registry.layers.find((l) => l.id === layerId);
    if (!layer) return;
    if (!layer.attempts.some((a) => a.id === attemptId)) {
      setAttemptId(layer.default || layer.attempts[0]?.id || "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layerId, registry]);

  // Whenever (layer, attempt) changes, refresh the cached-samples panel and
  // sync the seed input to that attempt's canonical seed.
  useEffect(() => {
    if (!layerId || !attemptId) return;
    setSamples(null);
    setSamplesErr("");
    setExpectedKey("");
    setSelectedSpecies("");
    fetchAttemptSamples(layerId, attemptId)
      .then((doc) => {
        setSamples(doc);
        if (Number.isInteger(doc?.canonical_seed)) setSeed(doc.canonical_seed);
      })
      .catch((e) => setSamplesErr(e.message));
  }, [layerId, attemptId]);

  const currentLayer = useMemo(
    () => registry?.layers.find((l) => l.id === layerId),
    [registry, layerId],
  );
  const currentAttempt = useMemo(
    () => currentLayer?.attempts.find((a) => a.id === attemptId),
    [currentLayer, attemptId],
  );
  const generationLayersById = useMemo(() => {
    const byId = {};
    for (const layer of registry?.layers || []) {
      byId[layer.id] = layer;
    }
    return byId;
  }, [registry]);

  useEffect(() => {
    if (!registry) return;
    setLayerDAttempts((current) => {
      const next = { ...current };
      for (const layerKey of ["layer_a", "layer_b", "layer_c"]) {
        const layer = generationLayersById[layerKey];
        const attempts = layer?.attempts || [];
        if (!attempts.length) continue;
        if (!attempts.some((attempt) => attempt.id === next[layerKey])) {
          next[layerKey] = layer.default || attempts[0].id;
        }
      }
      return next;
    });
  }, [registry, generationLayersById]);
  const usesSeed = currentAttempt?.uses_seed === true;
  const usesCells = currentAttempt?.uses_cells === true;
  const usesWeatherControls = currentAttempt?.uses_weather_controls === true;
  const isLayerD = layerId === "layer_d";
  const cells = useMemo(() => currentAttempt?.cells || [], [currentAttempt]);

  // Derive the season / diel axes from the cell list (handles partial banks).
  const seasonOptions = useMemo(() => {
    const seen = [...new Set(cells.map((c) => c.split("_")[0]))];
    return seen.map((s) => ({ value: s, label: s }));
  }, [cells]);
  const dielOptions = useMemo(() => {
    const seen = [...new Set(
      cells.filter((c) => c.startsWith(`${season}_`)).map((c) => c.split("_")[1]),
    )];
    return seen.map((d) => ({ value: d, label: d }));
  }, [cells, season]);

  // When the attempt changes, seed the (season, diel) selectors from the
  // attempt's default_cell (or the first available cell).
  useEffect(() => {
    if (isLayerD) {
      setSeason("summer");
      setDiel("morning");
      return;
    }
    if (!usesCells || !cells.length) {
      setSeason("");
      setDiel("");
      return;
    }
    const base = currentAttempt?.default_cell && cells.includes(currentAttempt.default_cell)
      ? currentAttempt.default_cell
      : cells[0];
    const [s, d] = base.split("_");
    setSeason(s);
    setDiel(d);
  }, [usesCells, cells, currentAttempt, isLayerD]);

  const speciesOptions = useMemo(() => {
    if (!samples?.expected?.length) return [];
    const bySpecies = new Map();
    for (const sample of samples.expected) {
      const metadata = sample.metadata || {};
      const species = metadata.species_common_name || metadata.species || "";
      if (!species) continue;
      bySpecies.set(species, {
        value: species,
        label: species,
        slug: metadata.species_slug || sample.stem,
      });
    }
    return [...bySpecies.values()].sort((a, b) => a.label.localeCompare(b.label));
  }, [samples]);

  useEffect(() => {
    if (!speciesOptions.length) {
      setSelectedSpecies("");
      return;
    }
    if (speciesOptions.some((s) => s.value === selectedSpecies)) return;
    const defaultSpecies = speciesOptions.find((s) => s.value === DEFAULT_LAYER_C_SPECIES);
    setSelectedSpecies((defaultSpecies || speciesOptions[0]).value);
  }, [speciesOptions, selectedSpecies]);

  // Keep diel valid when season changes (pick the first diel for that season).
  useEffect(() => {
    if (isLayerD) return;
    if (!usesCells || !season) return;
    if (!cells.includes(`${season}_${diel}`)) {
      const firstDiel = dielOptions[0]?.value || "";
      setDiel(firstDiel);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [season, usesCells]);

  // Flatten cached samples in display order (expected first, then showcase).
  // For bank attempts (uses_cells), filter to entries whose `cell` matches the
  // currently selected (season, diel) so the Expected panel mirrors the run.
  const expectedEntries = useMemo(() => {
    if (!samples) return [];
    const activeCell = usesCells && season && diel ? `${season}_${diel}` : null;
    const tiers = [
      { tier: "expected", entries: samples.expected || [] },
      { tier: "showcase", entries: samples.showcase || [] },
    ];
    return tiers.flatMap((t) =>
      t.entries
        .filter((s) => {
          if (selectedSpecies) {
            const species = s.metadata?.species_common_name || s.metadata?.species || "";
            if (species !== selectedSpecies) return false;
          }
          if (!activeCell) return true;
          // Cell-grouped entries match the active cell; cell-less entries
          // (e.g. legacy flat samples) are kept as-is.
          return !s.cell || s.cell === activeCell;
        })
        .map((s) => ({
          tier: t.tier,
          sample: s,
          key: `${t.tier}/${s.cell ? `${s.cell}/` : ""}${s.stem}`,
        })),
    );
  }, [samples, usesCells, season, diel, selectedSpecies]);

  // Auto-select the first expected sample when entries change.
  useEffect(() => {
    if (!expectedEntries.length) {
      setExpectedKey("");
      return;
    }
    if (!expectedEntries.some((e) => e.key === expectedKey)) {
      setExpectedKey(expectedEntries[0].key);
    }
  }, [expectedEntries, expectedKey]);

  const expectedSelected = useMemo(
    () => expectedEntries.find((e) => e.key === expectedKey) || null,
    [expectedEntries, expectedKey],
  );

  // Fake progress ticker.
  useEffect(() => {
    if (status === "loading") {
      const startedAt = Date.now();
      setProgress(6);
      const timer = window.setInterval(() => {
        const elapsedS = (Date.now() - startedAt) / 1000;
        const next = Math.min(92, 6 + (1 - Math.exp(-elapsedS / 18)) * 88);
        setProgress((current) => Math.max(current, Math.round(next)));
      }, 500);
      return () => window.clearInterval(timer);
    }
    if (status === "done") {
      setProgress(100);
      return undefined;
    }
    setProgress(0);
    return undefined;
  }, [status]);

  async function handleRun() {
    if (!layerId || !attemptId) return;
    setStatus("loading");
    setErrorMsg("");
    setResult(null);
    try {
      const runParams = {};
      if (usesSeed) {
        if (usesWeatherControls) runParams.retrieval_seed = Number(seed) || DEFAULT_SEED;
        else runParams.seed = Number(seed) || DEFAULT_SEED;
      }
      if (usesCells && season && diel) {
        runParams.season = season;
        runParams.diel = diel;
      }
      if (usesWeatherControls) {
        runParams.weather_type = weatherType;
        runParams.intensity = weatherIntensity;
        runParams.duration_s = Number(weatherDuration) || 10;
      }
      if (selectedSpecies) {
        runParams.species_common_name = selectedSpecies;
      }
      const data = isLayerD
        ? await generateSoundscape({
            seed: Number(seed) || DEFAULT_SEED,
            duration_s: Number(weatherDuration) || 10,
            season: season || "summer",
            diel: diel || "morning",
            weather_type: weatherType,
            intensity: weatherIntensity,
            include_weather: layerDIncludeWeather,
            include_events: layerDIncludeEvents,
            include_stems: true,
            layer_a_attempt: layerDAttempts.layer_a,
            layer_b_attempt: layerDAttempts.layer_b,
            layer_c_attempt: layerDAttempts.layer_c,
            layer_d_attempt: attemptId,
          })
        : await generateAttempt(layerId, attemptId, runParams);
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }

  function downloadDataUrl(href, filename) {
    const a = document.createElement("a");
    a.href = href;
    a.download = filename;
    a.click();
  }

  function downloadJson() {
    if (!result?.metadata) return;
    const blob = new Blob([JSON.stringify(result.metadata, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    downloadDataUrl(url, `${tag}_metadata.json`);
    URL.revokeObjectURL(url);
  }

  const isLoading = status === "loading";
  const isDone    = status === "done";
  const tag       = `${layerId}__${attemptId}${(usesCells || isLayerD) && season && diel ? `__${season}_${diel}` : ""}${(usesWeatherControls || isLayerD) ? `__${weatherType}_${weatherIntensity}_${weatherDuration}s` : ""}__${usesWeatherControls ? "retrieval_seed" : "seed"}${seed || DEFAULT_SEED}`;
  const progressText = getProgressText(progress, status);

  const registryReady = Boolean(registry);
  const layerOptions = registryReady
    ? registry.layers.map((l) => ({ value: l.id, label: l.label }))
    : allowedLayerIds.map((id) => ({ value: id, label: id }));
  const attemptOptions = (currentLayer?.attempts || []).map((a) => ({
    value: a.id,
    label: `${a.status === "production" ? "🏆 " : ""}${a.label}  (${a.stage}, ${a.status})${a.available === false ? " — unavailable" : ""}`,
  }));
  const registryBanner = regError
    ? `Failed to load layer registry: ${regError}. Controls shown for preview only — AI server unreachable.`
    : !registryReady
      ? "Loading registry…"
      : "";

  return (
    <section className="generation-page theme-generation">
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">{eyebrow}</p>
          <span>{title}</span>
        </div>
      </header>

      {registryBanner && (
        <p className={regError ? "analysis-error" : ""}>{registryBanner}</p>
      )}

      <div className="dev-controls-row">
        {/* ── Top: controls (own row) ── */}
        <aside className="panel generation-sidebar-card">
          <div className="generation-card-head">
            <h2>
              {currentLayer?.label || layerId}
              {currentAttempt && currentAttempt.available === false && (
                <span className="dev-unavailable-pill" title={currentAttempt.unavailable_reason || ""}>
                  Unavailable
                </span>
              )}
            </h2>
            <p>{currentAttempt?.label || attemptId}</p>
          </div>

          <div className="dev-controls-body">
            <div className="dev-controls-form">
              <section className="dev-controls-section">
                <p className="dev-controls-section-label">Model</p>
                <div className="dev-controls-section-grid two-col">
                  <LabeledSelect
                    label="Layer"
                    value={layerId}
                    onChange={setLayerId}
                    options={layerOptions}
                  />
                  <LabeledSelect
                    label="Model / Attempt"
                    value={attemptId}
                    onChange={setAttemptId}
                    options={attemptOptions}
                  />
                </div>
              </section>

              <section className="dev-controls-section">
                <p className="dev-controls-section-label">
                  Description
                  {currentAttempt?.author && (
                    <span className="dev-controls-section-pill">
                      by {currentAttempt.author}
                    </span>
                  )}
                </p>
                {currentAttempt?.description ? (
                  currentAttempt.description
                    .split(/\n\s*\n/)
                    .map((para, i) => (
                      <p key={i} className="dev-model-description">{para.trim()}</p>
                    ))
                ) : (
                  <p className="dev-model-description">
                    Description placeholder — to be authored by the model owner.
                  </p>
                )}
              </section>

              {(usesCells || isLayerD) && (
                <section className="dev-controls-section">
                  <p className="dev-controls-section-label">
                    {isLayerD ? "Upstream scene" : "Scene"}
                    {season && diel && (
                      <span className="dev-controls-section-pill">
                        {season} · {diel}
                      </span>
                    )}
                  </p>
                  <div className="dev-controls-section-grid two-col">
                    <LabeledSelect
                      label="Season"
                      value={season}
                      onChange={setSeason}
                      options={isLayerD ? SEASON_OPTIONS : seasonOptions}
                    />
                    <LabeledSelect
                      label="Time of day"
                      value={diel}
                      onChange={setDiel}
                      options={isLayerD ? DIEL_OPTIONS : dielOptions}
                    />
                  </div>
                </section>
              )}

              {speciesOptions.length > 1 && (
                <section className="dev-controls-section">
                  <p className="dev-controls-section-label">
                    Species
                    {selectedSpecies && (
                      <span className="dev-controls-section-pill">
                        {selectedSpecies}
                      </span>
                    )}
                  </p>
                  <LabeledSelect
                    label="Bird"
                    value={selectedSpecies}
                    onChange={setSelectedSpecies}
                    options={speciesOptions}
                  />
                </section>
              )}

              {(usesWeatherControls || isLayerD) && (
                <section className="dev-controls-section">
                  <p className="dev-controls-section-label">
                    {isLayerD ? "Upstream weather" : "Weather stem"}
                    <span className="dev-controls-section-pill">
                      {weatherType} · {weatherIntensity}
                    </span>
                  </p>
                  <div className="dev-controls-section-grid three-col">
                    <LabeledSelect
                      label="Weather type"
                      value={weatherType}
                      onChange={setWeatherType}
                      options={WEATHER_TYPE_OPTIONS}
                    />
                    <LabeledSelect
                      label="Intensity"
                      value={weatherIntensity}
                      onChange={setWeatherIntensity}
                      options={WEATHER_INTENSITY_OPTIONS}
                    />
                    <LabeledNumber
                      label="Duration"
                      value={weatherDuration}
                      min={1}
                      max={30}
                      step={1}
                      hint="seconds"
                      onChange={setWeatherDuration}
                    />
                  </div>
                </section>
              )}

              {isLayerD && (
                <section className="dev-controls-section">
                  <p className="dev-controls-section-label">Upstream models</p>
                  <div className="dev-controls-section-grid three-col">
                    <LabeledSelect
                      label="Layer A attempt"
                      value={layerDAttempts.layer_a}
                      onChange={(value) => setLayerDAttempts((current) => ({ ...current, layer_a: value }))}
                      options={attemptOptionsForLayer(generationLayersById.layer_a)}
                    />
                    <LabeledSelect
                      label="Layer B attempt"
                      value={layerDAttempts.layer_b}
                      onChange={(value) => setLayerDAttempts((current) => ({ ...current, layer_b: value }))}
                      options={attemptOptionsForLayer(generationLayersById.layer_b)}
                    />
                    <LabeledSelect
                      label="Layer C attempt"
                      value={layerDAttempts.layer_c}
                      onChange={(value) => setLayerDAttempts((current) => ({ ...current, layer_c: value }))}
                      options={attemptOptionsForLayer(generationLayersById.layer_c)}
                    />
                  </div>
                </section>
              )}

              {isLayerD && (
                <section className="dev-controls-section">
                  <p className="dev-controls-section-label">Mixer inputs</p>
                  <div className="dev-controls-section-grid two-col">
                    <label className="layer-a-field">
                      <span>Include weather stem</span>
                      <input
                        type="checkbox"
                        checked={layerDIncludeWeather}
                        onChange={(e) => setLayerDIncludeWeather(e.target.checked)}
                      />
                      <small>Generate Layer B and pass it to Layer D.</small>
                    </label>
                    <label className="layer-a-field">
                      <span>Include event stem</span>
                      <input
                        type="checkbox"
                        checked={layerDIncludeEvents}
                        onChange={(e) => setLayerDIncludeEvents(e.target.checked)}
                      />
                      <small>Generate Layer C and pass it to Layer D.</small>
                    </label>
                  </div>
                </section>
              )}

              <section className="dev-controls-section dev-controls-section-run">
                <p className="dev-controls-section-label">Run</p>
                <div className="dev-controls-run-grid">
                  <SeedField
                    label={usesWeatherControls ? "Retrieval seed" : "Seed"}
                    value={seed}
                    onChange={setSeed}
                    disabled={!usesSeed && !isLayerD}
                    hint={
                      isLayerD
                        ? "Same seed + same orchestration settings should reproduce the upstream choices."
                        : usesWeatherControls
                        ? "Same retrieval_seed + same weather settings = same selected asset and start offset."
                        : usesSeed
                          ? "Same seed + same attempt = same audio."
                        : "This model does not use a seed."
                    }
                  />
                  <button
                    type="button"
                    className="gen-primary-btn dev-run-btn"
                    onClick={handleRun}
                    disabled={isLoading || !attemptId || !registryReady || currentAttempt?.available === false}
                    title={
                      currentAttempt?.available === false
                        ? currentAttempt?.unavailable_reason || "Model weights unavailable"
                        : undefined
                    }
                  >
                    {isLoading ? "Generating..." : isLayerD ? "Run ABC->D Mix" : "▶ Generate"}
                  </button>
                </div>
              </section>
            </div>

            {currentAttempt && currentAttempt.available === false && (
              <div className="dev-availability-warn" role="alert">
                <p className="dev-availability-title">⚠ Model weights unavailable</p>
                <p className="dev-availability-reason">
                  {currentAttempt.unavailable_reason ||
                    "Required weight files are not on disk."}
                </p>
                {currentAttempt.missing_files?.length > 0 && currentAttempt.checkpoint && (
                  <pre className="dev-availability-cmd">
{`dvc pull \\\n  ${currentAttempt.missing_files
  .map((f) => `${currentAttempt.checkpoint}/${f}`)
  .join(" \\\n  ")}`}
                  </pre>
                )}
              </div>
            )}

            {(isLoading || isDone) && (
              <ProgressBlock progress={progress} label={progressText} />
            )}

            {errorMsg && <p className="analysis-error">{errorMsg}</p>}

            <PromptDisplay
              prompt={result?.metadata?.prompt}
              cell={result?.metadata?.cell}
              locked={Boolean(result?.metadata?.prompt_locked)}
              show={isDone}
              loading={isLoading}
            />

            <div className="dev-controls-meta">
              <div className="dev-meta-chip">
                <span className="dev-meta-chip-label">Attempt</span>
                <code>{attemptId || "—"}</code>
              </div>
              <div className="dev-meta-chip">
                <span className="dev-meta-chip-label">Run tag</span>
                <code>{isDone ? tag : "—"}</code>
              </div>
              <div className="dev-meta-chip">
                <span className="dev-meta-chip-label">Audio stats</span>
                <code>
                  {isDone && result?.metadata?.audio
                    ? `RMS ${result.metadata.audio.rms?.toFixed?.(4)} · peak ${result.metadata.audio.peak?.toFixed?.(4)}`
                    : "—"}
                </code>
              </div>
            </div>

            {isDone && result?.metadata?.prompt_locked && (
              <p className="mock-badge">
                {usesCells
                  ? "Prompt fixed per scene — pick season + time of day to change it · free-text disabled"
                  : "Fixed prompt active — user prompts disabled"}
              </p>
            )}
          </div>
        </aside>

      </div>

      <div className="dev-results-row">
        {/* ── Left: expected results (cached) ── */}
        <main className="panel dev-result-card">
          <div className="generation-card-head">
            <h2>Expected Results</h2>
            <p>
              {selectedSpecies
                ? `${selectedSpecies} · layer C only expected sample`
                : samples?.canonical_seed != null
                  ? `Cached samples · canonical seed ${samples.canonical_seed}`
                  : "Cached expected / showcase samples"}
            </p>
          </div>

          <div className="dev-result-body">
            {samplesErr && <p className="analysis-error">{samplesErr}</p>}

            {expectedEntries.length > 1 && (
              <div className="dev-sample-tabs" role="tablist" aria-label="Cached samples">
                {expectedEntries.map((e, i) => {
                  // For cell-grouped entries every tab is the same (tier, cell);
                  // shorten the label to a numeric index + clip stem so the tabs
                  // don't all repeat the cell name.
                  const displayStem = e.sample.stem.startsWith("real_")
                    ? e.sample.stem.slice("real_".length)
                    : e.sample.stem;
                  const eyebrow = e.sample.cell ? `Sample ${i + 1}` : e.tier;
                  return (
                    <button
                      key={e.key}
                      type="button"
                      role="tab"
                      aria-selected={e.key === expectedKey}
                      className={`dev-sample-tab${e.key === expectedKey ? " active" : ""}`}
                      onClick={() => setExpectedKey(e.key)}
                    >
                      <span className="dev-sample-tab-tier">{eyebrow}</span>
                      <span className="dev-sample-tab-stem">{displayStem}</span>
                    </button>
                  );
                })}
              </div>
            )}

            {isLayerD ? (
              <LayerDExpectedNote />
            ) : (
              <ExpectedSample
                layerId={layerId}
                attemptId={attemptId}
                tier={expectedSelected?.tier}
                sample={expectedSelected?.sample}
                loading={!samplesErr && !samples}
                empty={!samplesErr && samples && expectedEntries.length === 0}
              />
            )}
          </div>
        </main>

        {/* ── Right: generated results (live run) ── */}
        <aside className="panel dev-result-card">
          <div className="generation-card-head">
            <h2>Generated Results</h2>
            <p>Live output from the latest run</p>
          </div>

          <div className="dev-result-body">
            {isLayerD && isDone && result?.stems && (
              <ReviewSection title="ABC Upstream Stems">
                <div className="dev-layer-d-stem-grid">
                  <LayerDStemCard title="Layer A - Ambient" stem={result.stems.layer_a} />
                  <LayerDStemCard title="Layer B - Weather" stem={result.stems.layer_b} />
                  <LayerDStemCard title="Layer C - Events" stem={result.stems.layer_c} />
                </div>
              </ReviewSection>
            )}

            <ReviewSection title="♪ Audio">
              {isDone && result?.audio_b64 ? (
                <>
                  <audio controls
                         src={`data:audio/wav;base64,${result.audio_b64}`}
                         style={{ width: "100%" }} />
                  <p style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
                    {result.sample_rate} Hz · {result.duration_s?.toFixed?.(1) ?? result.duration_s} s ·{" "}
                    {usesWeatherControls
                      ? `retrieval_seed ${result.metadata?.retrieval_seed ?? result.metadata?.seed ?? seed}`
                      : `seed ${result.metadata?.seed ?? seed}`}
                  </p>
                </>
              ) : (
                <Placeholder kind="audio" loading={isLoading}>
                  {isLoading
                    ? "Generating upstream stems and final mix..."
                    : isLayerD
                      ? "Click Run ABC->D Mix to generate upstream stems and final mix."
                      : "Click Generate to run the handler."}
                </Placeholder>
              )}
            </ReviewSection>

            <ReviewSection title="▤ Mel-Spectrogram">
              {isDone && result?.image_b64 ? (
                <img src={`data:image/png;base64,${result.image_b64}`}
                     alt="Mel-spectrogram"
                     className="gen-spectrogram-img layer-a-spectrogram-img" />
              ) : (
                <Placeholder kind="image" loading={isLoading}>
                  {isLoading ? "Rendering spectrogram…" : "Spectrogram appears after generation."}
                </Placeholder>
              )}
            </ReviewSection>

            <ReviewSection title="{ } Metadata">
              {isDone && result?.metadata ? (
                <pre className="layer-a-json">
                  {JSON.stringify(result.metadata, null, 2)}
                </pre>
              ) : (
                <Placeholder kind="json" loading={isLoading}>
                  {isLoading ? "Collecting metadata…" : "Metadata appears after generation."}
                </Placeholder>
              )}
            </ReviewSection>

            {isLoading && (
              <ProgressBlock progress={progress} label={progressText} />
            )}

            <div className="dev-download-row">
              <button type="button" className="gen-secondary-btn"
                      disabled={!isDone || !result?.audio_b64}
                      onClick={() => downloadDataUrl(
                        `data:audio/wav;base64,${result.audio_b64}`,
                        `${tag}.wav`,
                      )}>
                ↓ WAV
              </button>
              <button type="button" className="gen-secondary-btn"
                      disabled={!isDone || !result?.image_b64}
                      onClick={() => downloadDataUrl(
                        `data:image/png;base64,${result.image_b64}`,
                        `${tag}_spectrogram.png`,
                      )}>
                ↓ Spectrogram
              </button>
              <button type="button" className="gen-secondary-btn"
                      disabled={!isDone || !result?.metadata}
                      onClick={downloadJson}>
                ↓ Metadata
              </button>
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}

function ExpectedSample({ layerId, attemptId, tier, sample, loading, empty }) {
  // `layerId` / `attemptId` / `tier` kept in the signature for callers; the
  // playable URL comes from the server-built `sample.wav_url`.
  void layerId; void attemptId; void tier;
  const hasSample = Boolean(sample);
  const wavSrc = hasSample && sample.has_wav ? sampleWavUrl(sample) : null;

  const audioCaption = loading
    ? "Loading cached samples…"
    : empty
      ? "No cached samples on disk for this attempt yet."
      : !hasSample
        ? "Select a cached sample to preview."
        : "WAV not present locally — generate or run dvc pull.";

  const spectrogramCaption = loading
    ? "Loading…"
    : empty
      ? "No cached spectrogram available."
      : !hasSample
        ? "Spectrogram appears once a sample is selected."
        : sample.has_png
          ? "PNG is DVC-tracked — run dvc pull to render."
          : "No spectrogram on disk.";

  const metadataCaption = loading
    ? "Loading…"
    : empty
      ? "No metadata cached for this attempt yet."
      : !hasSample
        ? "Metadata appears once a sample is selected."
        : "No metadata on disk for this sample.";

  return (
    <>
      <ReviewSection title="♪ Audio">
        {wavSrc ? (
          <>
            <audio controls src={wavSrc} style={{ width: "100%" }}>
              Your browser doesn't support audio playback.
            </audio>
            <p style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
              <code>{sample.stem}</code> · {tier}
            </p>
          </>
        ) : (
          <Placeholder kind="audio" loading={loading}>{audioCaption}</Placeholder>
        )}
      </ReviewSection>

      <ReviewSection title="▤ Mel-Spectrogram">
        {hasSample && sample.png_b64 ? (
          <img
            src={`data:image/png;base64,${sample.png_b64}`}
            alt={`${tier} sample ${sample.stem}`}
            className="gen-spectrogram-img layer-a-spectrogram-img"
          />
        ) : (
          <Placeholder kind="image" loading={loading}>{spectrogramCaption}</Placeholder>
        )}
      </ReviewSection>

      <ReviewSection title="{ } Metadata">
        {hasSample && sample.metadata ? (
          <pre className="layer-a-json">
            {JSON.stringify(sample.metadata, null, 2)}
          </pre>
        ) : (
          <Placeholder kind="json" loading={loading}>{metadataCaption}</Placeholder>
        )}
      </ReviewSection>
    </>
  );
}

function LayerDExpectedNote() {
  return (
    <div className="dev-layer-d-expected-note">
      <h3>Layer D expects upstream stems</h3>
      <p>
        The mixer does not own cached expected samples by itself. Use Generated
        Results to inspect the Layer A ambient stem, Layer B weather stem,
        Layer C event stem, and the final Layer D mix for the current run.
      </p>
    </div>
  );
}

function LayerDStemCard({ title, stem }) {
  const attempt = stem?.metadata?.attempt || {};
  if (!stem) {
    return (
      <article className="dev-layer-d-stem-card">
        <h4>{title}</h4>
        <Placeholder kind="audio">Stem disabled for this run.</Placeholder>
      </article>
    );
  }
  return (
    <article className="dev-layer-d-stem-card">
      <h4>{title}</h4>
      <div className="dev-layer-d-model-line">
        <span>{attempt.label || attempt.id || "Unknown attempt"}</span>
        {attempt.author && <small>by {attempt.author}</small>}
      </div>
      <div className="dev-layer-d-chip-row">
        {attempt.id && <code>{attempt.id}</code>}
        {attempt.stage && <code>{attempt.stage}</code>}
        {attempt.status && <code>{attempt.status}</code>}
      </div>
      {stem.audio_b64 ? (
        <audio controls src={`data:audio/wav;base64,${stem.audio_b64}`} style={{ width: "100%" }} />
      ) : (
        <Placeholder kind="audio">No stem audio returned.</Placeholder>
      )}
      {stem.image_b64 ? (
        <img
          src={`data:image/png;base64,${stem.image_b64}`}
          alt={`${title} spectrogram`}
          className="gen-spectrogram-img layer-a-spectrogram-img"
        />
      ) : (
        <Placeholder kind="image">No stem spectrogram returned.</Placeholder>
      )}
      <pre className="layer-a-json">
        {JSON.stringify(stem.metadata || {}, null, 2)}
      </pre>
    </article>
  );
}

function Placeholder({ kind, loading, children }) {
  return (
    <div className={`dev-placeholder dev-placeholder-${kind}${loading ? " is-loading" : ""}`}>
      <div className="dev-placeholder-art" aria-hidden="true">
        {kind === "audio" && (
          <div className="dev-placeholder-waveform">
            {Array.from({ length: 28 }).map((_, i) => (
              <span key={i} style={{ ["--i"]: i }} />
            ))}
          </div>
        )}
        {kind === "image" && <div className="dev-placeholder-image" />}
        {kind === "json" && (
          <div className="dev-placeholder-json">
            <span style={{ width: "30%" }} />
            <span style={{ width: "65%" }} />
            <span style={{ width: "50%" }} />
            <span style={{ width: "72%" }} />
            <span style={{ width: "40%" }} />
          </div>
        )}
      </div>
      <p className="dev-placeholder-caption">{children}</p>
    </div>
  );
}

function SeedField({ label = "Seed", value, onChange, disabled, hint }) {
  const randomize = () => {
    onChange(Math.floor(Math.random() * 2147483648));
  };
  return (
    <label className={`layer-a-field${disabled ? " is-disabled" : ""}`}>
      <span>{label}</span>
      <div className="dev-seed-input-row">
        <input
          className="layer-a-input"
          type="number"
          value={value}
          min={0}
          max={2147483647}
          step={1}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        />
        <button
          type="button"
          className="dev-seed-random-btn"
          onClick={randomize}
          disabled={disabled}
          title="Pick a random seed"
          aria-label="Pick a random seed"
        >
          🎲
        </button>
      </div>
      {hint && <small>{hint}</small>}
    </label>
  );
}

function LabeledNumber({ label, value, min, max, step = 1, hint, onChange, disabled = false }) {
  return (
    <label className={`layer-a-field${disabled ? " is-disabled" : ""}`}>
      <span>{label}</span>
      <input className="layer-a-input" type="number"
             value={value} min={min} max={max} step={step}
             disabled={disabled}
             onChange={(e) => onChange(e.target.value)} />
      {hint && <small>{hint}</small>}
    </label>
  );
}

function LabeledSelect({ label, value, options, onChange }) {
  return (
    <label className="layer-a-field">
      <span>{label}</span>
      <select className="layer-a-input"
              value={value}
              onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

function attemptOptionsForLayer(layer) {
  const attempts = layer?.attempts || [];
  return attempts.map((attempt) => ({
    value: attempt.id,
    label: `${attempt.label} (${attempt.stage}, ${attempt.status})`,
  }));
}

function getProgressText(progress, status) {
  if (status === "done") return "Complete";
  if (progress < 18) return "Preparing";
  if (progress < 38) return "Loading model / LoRA";
  if (progress < 72) return "Generating audio";
  if (progress < 94) return "Rendering WAV and spectrogram";
  return "Finalizing";
}

function ProgressBlock({ progress, label }) {
  return (
    <div className="gen-progress-block layer-a-progress" aria-live="polite">
      <div className="gen-progress-line">
        <strong>{Math.round(progress)}%</strong>
        <p>{label}</p>
      </div>
      <div className="gen-progress-track"
           role="progressbar"
           aria-valuemin="0" aria-valuemax="100"
           aria-valuenow={Math.round(progress)}
           aria-label="Generation progress">
        <i style={{ width: `${Math.max(4, Math.min(100, progress))}%` }} />
      </div>
    </div>
  );
}

function ReviewSection({ title, children }) {
  return (
    <section className="dev-review-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function PromptDisplay({ prompt, cell, locked, show, loading }) {
  return (
    <section className="prompt-display">
      <div className="prompt-display-head">
        <h3>✎ Prompt used</h3>
        <div className="prompt-display-tags">
          {cell && <span className="prompt-display-cell">{cell}</span>}
          {locked && <span className="prompt-display-locked">locked</span>}
        </div>
      </div>
      {show && prompt ? (
        <p className="prompt-display-text">{prompt}</p>
      ) : (
        <p className="prompt-display-empty">
          {loading
            ? "Resolving prompt…"
            : "The server-side prompt for this run appears here after you generate."}
        </p>
      )}
    </section>
  );
}
