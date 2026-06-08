/**
 * API client for the Sonic Lab backend.
 *
 * All AI endpoints go through the Express backend (/api/*) which proxies to
 * the FastAPI inference server internally. No direct browser → Python calls.
 *
 * The dev test UI is layer/attempt-driven via a single dropdown:
 *   1. fetchLayerRegistry()                    → populate the dropdown
 *   2. generateAttempt(layerId, attemptId, …)  → run that attempt
 *
 * The legacy /api/analysis, /api/generation, /api/layer_a/* endpoints were
 * removed in the restructure to /layers/<X>/attempts/<id>/generate.
 */

const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

function apiErrorMessage(err, fallback) {
  return (
    err?.message ||
    err?.detail ||
    err?.upstream?.message ||
    err?.upstream?.detail ||
    fallback
  );
}

// ─── Layer registry (drives the dropdown) ─────────────────────────────────────

/**
 * Fetch the list of registered layers + attempts.
 * @returns {Promise<{layers: Array<{
 *   id: string,
 *   label: string,
 *   default: string,
 *   attempts: Array<{id:string,label:string,stage:string,author:string,status:string}>
 * }>}>}
 */
export async function fetchLayerRegistry() {
  const res = await fetch(`${API_BASE}/api/layers`, { credentials: "include" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(apiErrorMessage(err, `Failed to list layers (${res.status})`));
  }
  return res.json();
}

// ─── Generate ─────────────────────────────────────────────────────────────────

/**
 * Generate audio with a specific layer/attempt. Most model behavior stays
 * registry/handler-owned; the dev UI only forwards the small runtime selector
 * set needed by each attempt. `seed` is always supported; bank attempts may
 * also use `(season, diel)`, Layer B uses weather/generator controls, and
 * Layer C retrieval can use `species_common_name`.
 *
 * @param {string} layerId    e.g. "layer_a"
 * @param {string} attemptId  e.g. "lucas__smoke_1__audioldm2_spring_night"
 * @param {{
 *   seed?: number,
 *   retrieval_seed?: number,
 *   season?: string,
 *   diel?: string,
 *   weather_type?: string,
 *   intensity?: string,
 *   wind_intensity?: string,
 *   rain_intensity?: string,
 *   species_common_name?: string,
 *   duration_s?: number
 * }} params
 * @returns {Promise<{ok:boolean, audio_b64:string, image_b64:string, metadata:object, sample_rate:number, duration_s:number}>}
 */
// ─── Cached samples (no model load required) ──────────────────────────────────

/**
 * Fetch the cached reference + showcase samples for an attempt.
 * See .claude/context/dev/artifact_policy.md.
 * @returns {Promise<{
 *   attempt:string, layer:string, canonical_seed:number,
 *   reference: Array<{stem:string, has_png:boolean, has_wav:boolean, has_json:boolean,
 *                     png_b64:?string, metadata:?object, wav_url:?string}>,
 *   showcase:  Array<{stem:string, has_png:boolean, has_wav:boolean, has_json:boolean,
 *                     png_b64:?string, metadata:?object, wav_url:?string}>
 * }>}
 */
export async function fetchAttemptSamples(layerId, attemptId) {
  const res = await fetch(
    `${API_BASE}/api/layers/${encodeURIComponent(layerId)}/attempts/${encodeURIComponent(attemptId)}/samples`,
    { credentials: "include" },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(apiErrorMessage(err, `Failed to list samples (${res.status})`));
  }
  return res.json();
}

/**
 * Build a playable URL for a cached sample WAV. The server returns a fully
 * formed `wav_url` per sample (it knows the layout — flat / case-dir /
 * cell-grouped); the frontend just prefixes the API base + `/api`.
 */
export function sampleWavUrl(sample) {
  if (!sample?.wav_url) return null;
  const base = API_BASE || "";
  return `${base}/api${sample.wav_url}`;
}

// ─── Stage-3 product endpoints ──────────────────────────────────────────────────
// Generation is implemented through the A/B/C -> D orchestrator. Analysis uses
// the Layer E head orchestrator + aggregator. Transformation remains a
// placeholder.

const _PLACEHOLDER_MSG =
  "This product feature is being rebuilt on the new layer/attempt structure. " +
  "Use /dev/layers in the meantime.";

export async function analyseAudio(file, attempts = {}, register = "immersive") {
  const form = new FormData();
  form.append("file", file);
  form.append("register", register);
  for (const [key, value] of Object.entries(attempts)) {
    if (value) form.append(`${key}_attempt`, value);
  }
  const res = await fetch(`${API_BASE}/api/analysis`, {
    method: "POST",
    credentials: "include",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(apiErrorMessage(err, `Analysis failed (${res.status})`));
  }
  return res.json();
}

/**
 * Run one Layer E analysis head on an uploaded audio file. Analysis is
 * per-attempt and upload-based (not seed-based): a specific attempt owns a
 * single detector head (ambient / weather / events), so the dev page calls
 * this once per head with that head's selected attempt.
 *
 * Response shape (server.py POST /layers/{layer}/attempts/{id}/analyze):
 *   {
 *     ok: true,
 *     report:  { ...head-specific fields, confidence:0..1 },   // handler output
 *     attempt: { layer, id, label, stage, head, status, ... }  // spec snapshot
 *   }
 *
 * For the E-A ambient head the report carries:
 *   { estimated_conditions:{season, diel_bin, hour, month},
 *     similar_clips:[{segment_id, similarity}], confidence,
 *     season_confidence, head_agreement, ood_flag, k, tau }
 */
export async function analyseUpload(layerId, attemptId, file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(
    `${API_BASE}/api/layers/${encodeURIComponent(layerId)}/attempts/${encodeURIComponent(attemptId)}/analyze`,
    {
      method:      "POST",
      credentials: "include",
      body:        form,
    },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(apiErrorMessage(err, `Analysis failed (${res.status})`));
  }
  return res.json();
}

/**
 * Re-render an analysis report in a chosen tone register through the LLM-OSS
 * report writer (POST /api/analysis/narrative). No detectors re-run — the
 * caller supplies the already-computed fused report. Backs the scene-page tone
 * toggle.
 * @param {object} report   fused analysis report JSON (e.g. from analyseUpload)
 * @param {"analytical"|"immersive"} register
 * @returns {Promise<{ok:boolean, narrative:{register:string, text:string, faithful:boolean, violations:string[]}}>}
 */
export async function narrateReport(report, register = "analytical") {
  const res = await fetch(`${API_BASE}/api/analysis/narrative`, {
    method:      "POST",
    headers:     { "Content-Type": "application/json" },
    credentials: "include",
    body:        JSON.stringify({ report, register }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(apiErrorMessage(err, `Narrative failed (${res.status})`));
  }
  return res.json();
}

export async function parseGenerationPrompt(prompt) {
  const res = await fetch(`${API_BASE}/api/generation/parse`, {
    method:      "POST",
    headers:     { "Content-Type": "application/json" },
    credentials: "include",
    body:        JSON.stringify({ prompt }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(apiErrorMessage(err, `Prompt parsing failed (${res.status})`));
  }
  return res.json();
}

export async function generateSoundscape(conditions = {}) {
  const layerA = conditions.layer_a || {};
  const layerB = conditions.layer_b || null;
  const layerC = conditions.layer_c || null;
  const windSpeed = Number(conditions.wind_speed_ms) || 0;
  const precipitation = Number(conditions.precipitation_mm) || 0;
  const fallbackIntensity = windSpeed >= 10 || precipitation >= 20
    ? "heavy"
    : windSpeed >= 4 || precipitation > 0
      ? "medium"
      : "light";
  const hasExplicitLayerContracts =
    conditions.layer_a !== undefined ||
    conditions.layer_b !== undefined ||
    conditions.layer_c !== undefined;
  const hasParsedEvents = Boolean(layerC?.species?.length);
  const includeWeather = conditions.include_weather ??
    (hasExplicitLayerContracts ? Boolean(layerB) : true);
  const includeEvents = conditions.include_events ??
    (hasExplicitLayerContracts ? hasParsedEvents : true);
  const payload = {
    seed: Number.isInteger(conditions.seed) ? conditions.seed : 42,
    duration_s: Number(conditions.duration_s ?? layerB?.duration_s) || 30,
    season: conditions.season ?? layerA.season,
    diel: conditions.diel ?? conditions.sample_bin ?? conditions.time ?? layerA.diel,
    weather_type: conditions.weather_type ?? layerB?.weather_type ?? (precipitation > 0 ? "rain+wind" : "wind"),
    intensity: conditions.intensity ?? layerB?.intensity ?? fallbackIntensity,
    include_weather: includeWeather,
    include_events: includeEvents,
    include_stems: conditions.include_stems === true,
    layer_a_attempt: conditions.layer_a_attempt,
    layer_b_attempt: conditions.layer_b_attempt,
    layer_c_attempt: conditions.layer_c_attempt,
    layer_d_attempt: conditions.layer_d_attempt,
  };
  const res = await fetch(`${API_BASE}/api/generation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(apiErrorMessage(err, `Generation failed (${res.status})`));
  }
  return res.json();
}
export async function transformSoundscape() { throw new Error(_PLACEHOLDER_MSG); }

export async function generateAttempt(
  layerId,
  attemptId,
  {
    seed,
    retrieval_seed,
    season,
    diel,
    weather_type,
    intensity,
    wind_intensity,
    rain_intensity,
    duration_s,
    species_common_name: speciesCommonName,
  } = {},
) {
  const payload = {};
  if (seed !== undefined) payload.seed = seed;
  if (retrieval_seed !== undefined) payload.retrieval_seed = retrieval_seed;
  if (season && diel) {
    payload.season = season;
    payload.diel = diel;
  }
  if (weather_type) payload.weather_type = weather_type;
  if (intensity) payload.intensity = intensity;
  if (wind_intensity) payload.wind_intensity = wind_intensity;
  if (rain_intensity) payload.rain_intensity = rain_intensity;
  if (duration_s) payload.duration_s = duration_s;
  if (speciesCommonName) payload.species_common_name = speciesCommonName;
  const res = await fetch(
    `${API_BASE}/api/layers/${encodeURIComponent(layerId)}/attempts/${encodeURIComponent(attemptId)}/generate`,
    {
      method:      "POST",
      headers:     { "Content-Type": "application/json" },
      credentials: "include",
      body:        JSON.stringify(payload),
    },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(apiErrorMessage(err, `Generation failed (${res.status})`));
  }
  return res.json();
}
