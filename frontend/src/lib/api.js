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
 * Generate audio with a specific layer/attempt. Only `seed` is sent; every
 * other parameter is owned by the registry/handler server-side (see CLAUDE.md
 * → "Layer A dev-generation contract").
 *
 * @param {string} layerId    e.g. "layer_a"
 * @param {string} attemptId  e.g. "lucas__smoke_1__audioldm2_spring_night"
 * @param {{seed?: number}} params
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

// ─── Stage-3 product endpoints (placeholders) ─────────────────────────────────
// Analysis / Generation / Transformation routes will be reimplemented on top of
// the new layer/attempt registry in a follow-up. The current stubs keep the
// build green and surface a clear message at click time.

const _PLACEHOLDER_MSG =
  "This product feature is being rebuilt on the new layer/attempt structure. " +
  "Use /dev/layers in the meantime.";

export async function analyseAudio()        { throw new Error(_PLACEHOLDER_MSG); }

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

export async function generateSoundscape()  { throw new Error(_PLACEHOLDER_MSG); }
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
    duration_s,
    species,
    count,
    layer_c_only,
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
  if (duration_s) payload.duration_s = duration_s;
  if (species) payload.species = species;
  if (count !== undefined) payload.count = count;
  if (layer_c_only !== undefined) payload.layer_c_only = layer_c_only;
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
