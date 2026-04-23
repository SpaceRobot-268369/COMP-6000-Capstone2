/**
 * API client for the Sonic Lab backend.
 *
 * All AI endpoints go through the Express backend (/api/*) which proxies to
 * the FastAPI inference server internally.  No direct browser→Python calls.
 */

const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

// ─── Analysis ─────────────────────────────────────────────────────────────────

/**
 * Analyse an audio file.
 * @param {File} file  WAV audio file
 * @returns {Promise<object>} latent vector + metadata
 */
export async function analyseAudio(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/analysis`, {
    method:      "POST",
    body:        form,
    credentials: "include",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message || err.detail || `Analysis failed (${res.status})`);
  }
  return res.json();
}

// ─── Generation ───────────────────────────────────────────────────────────────

/**
 * Generate a spectrogram from environmental conditions.
 * @param {object} conditions  env feature object (see EnvControls.DEFAULT_CONDITIONS)
 * @returns {Promise<{image_b64: string, shape: number[], mel_db: number[][], mock: false}>}
 */
export async function generateSoundscape(conditions) {
  const res = await fetch(`${API_BASE}/api/generation`, {
    method:      "POST",
    headers:     { "Content-Type": "application/json" },
    credentials: "include",
    body:        JSON.stringify(conditions),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message || err.detail || `Generation failed (${res.status})`);
  }
  const data = await res.json();
  return { ...data, mock: false };
}

// ─── Transformation ───────────────────────────────────────────────────────────

/**
 * Transform an audio file under new environmental conditions.
 * @param {File}   file        source audio
 * @param {object} conditions  target env conditions
 * @returns {Promise<object>}
 */
export async function transformSoundscape(file, conditions) {
  const form = new FormData();
  form.append("file", file);
  form.append("conditions", JSON.stringify(conditions));
  const res = await fetch(`${API_BASE}/api/transformation`, {
    method:      "POST",
    body:        form,
    credentials: "include",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message || err.detail || `Transformation failed (${res.status})`);
  }
  return res.json();
}
