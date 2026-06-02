/**
 * Filesystem-backed reader for cached "expected" / "showcase" samples.
 *
 * The Express backend serves these directly from the repo checkout (the
 * always-on "Server A" design) instead of proxying to the GPU worker — they're
 * static artefacts with no model dependency, so there's no reason to wake
 * serverB to look at them. Only /generate still goes through the AI tunnel.
 *
 * This mirrors the Python `registry.list_samples` scan in
 * acoustic_ai/server/registry.py. Three on-disk layouts are supported:
 *
 *   <tier>/<case>/{audio.wav, spectrogram.png, metadata.json}            ← canonical case-dir
 *   <tier>/<cell>/<case>/{audio.wav, spectrogram.png, metadata.json}     ← bank (uses_cells)
 *   <tier>/<stem>.{wav,png,metadata.json}                                ← legacy flat
 *
 * PNG/JSON contents are inlined (small). `wav_url` mirrors the on-disk path so
 * the frontend can request the WAV without knowing the layout. WAV blobs are
 * DVC-tracked: when only the `.dvc` pointer is present `has_wav` is still true
 * but the stream route returns 404 with a "run dvc pull" hint.
 */

import fs from "node:fs";
import path from "node:path";

// Root of the attempts tree. In Docker the acoustic_ai dir is mounted read-only
// (see services/dev/docker-compose.yml); on a native host point this at the
// repo checkout's acoustic_ai/layers. Some local stacks provide the acoustic_ai
// root instead, so accept both shapes.
const RAW_LAYERS_ROOT = process.env.AI_LAYERS_ROOT || "/acoustic_ai/layers";
const LAYERS_ROOT = path.basename(RAW_LAYERS_ROOT) === "layers"
  ? RAW_LAYERS_ROOT
  : path.join(RAW_LAYERS_ROOT, "layers");

const TIERS = ["expected", "showcase"];
const CANONICAL_SEED_DEFAULT = 42;

function attemptRoot(layer, attempt) {
  return path.join(LAYERS_ROOT, layer, "attempts", attempt);
}

function isCaseDir(dir) {
  // A case dir holds the fixed triplet; presence of audio.wav or its DVC
  // pointer is the canonical signal.
  return (
    fs.existsSync(path.join(dir, "audio.wav")) ||
    fs.existsSync(path.join(dir, "audio.wav.dvc"))
  );
}

function readCaseDir(layer, attempt, tier, caseDir, caseName, cell) {
  const relParts = [tier];
  if (cell) relParts.push(cell);
  relParts.push(caseName);
  const relWav = [...relParts, "audio.wav"].join("/");

  const entry = {
    stem: caseName,
    cell: cell || null,
    has_wav: false,
    has_png: false,
    has_json: false,
    png_b64: null,
    metadata: null,
    wav_url: null,
  };

  if (
    fs.existsSync(path.join(caseDir, "audio.wav")) ||
    fs.existsSync(path.join(caseDir, "audio.wav.dvc"))
  ) {
    entry.has_wav = true;
    entry.wav_url = `/layers/${layer}/attempts/${attempt}/samples/${relWav}`;
  }

  const png = path.join(caseDir, "spectrogram.png");
  if (fs.existsSync(png)) {
    entry.has_png = true;
    try {
      entry.png_b64 = fs.readFileSync(png).toString("base64");
    } catch {
      /* leave null */
    }
  } else if (fs.existsSync(path.join(caseDir, "spectrogram.png.dvc"))) {
    entry.has_png = true;
  }

  const md = path.join(caseDir, "metadata.json");
  if (fs.existsSync(md)) {
    entry.has_json = true;
    try {
      entry.metadata = JSON.parse(fs.readFileSync(md, "utf-8"));
    } catch {
      /* leave null */
    }
  } else if (fs.existsSync(path.join(caseDir, "metadata.json.dvc"))) {
    entry.has_json = true;
  }

  return entry;
}

const FLAT_SUFFIXES = [
  ".wav.dvc",
  ".png.dvc",
  ".metadata.json.dvc",
  ".wav",
  ".png",
  ".metadata.json",
];

function readFlatEntries(layer, attempt, tier, dir) {
  // Legacy fallback: flat files `<tier>/<stem>.{wav,png,metadata.json}`.
  const byStem = new Map();
  for (const name of fs.readdirSync(dir).sort()) {
    const full = path.join(dir, name);
    if (fs.statSync(full).isDirectory()) continue;
    if (name === ".gitkeep" || name === ".gitignore") continue;

    let stem = name;
    for (const suf of FLAT_SUFFIXES) {
      if (stem.endsWith(suf)) {
        stem = stem.slice(0, -suf.length);
        break;
      }
    }
    if (!byStem.has(stem)) {
      byStem.set(stem, {
        stem,
        cell: null,
        has_wav: false,
        has_png: false,
        has_json: false,
        png_b64: null,
        metadata: null,
        wav_url: null,
      });
    }
    const e = byStem.get(stem);
    const wavUrl = `/layers/${layer}/attempts/${attempt}/samples/${tier}/${stem}.wav`;
    if (name.endsWith(".png")) {
      e.has_png = true;
      try {
        e.png_b64 = fs.readFileSync(full).toString("base64");
      } catch {
        /* ignore */
      }
    } else if (name.endsWith(".metadata.json")) {
      e.has_json = true;
      try {
        e.metadata = JSON.parse(fs.readFileSync(full, "utf-8"));
      } catch {
        /* ignore */
      }
    } else if (name.endsWith(".wav")) {
      e.has_wav = true;
      e.wav_url = wavUrl;
    } else if (name.endsWith(".png.dvc")) {
      e.has_png = true;
    } else if (name.endsWith(".wav.dvc")) {
      e.has_wav = true;
      if (!e.wav_url) e.wav_url = wavUrl;
    } else if (name.endsWith(".metadata.json.dvc")) {
      e.has_json = true;
    }
  }
  return [...byStem.values()];
}

function scanTier(layer, attempt, tier) {
  const dir = path.join(attemptRoot(layer, attempt), tier);
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return [];

  const entries = [];
  let hasDirs = false;
  for (const name of fs.readdirSync(dir).sort()) {
    if (name === ".gitkeep" || name === ".gitignore") continue;
    const child = path.join(dir, name);
    if (!fs.statSync(child).isDirectory()) continue;
    hasDirs = true;

    if (isCaseDir(child)) {
      entries.push(readCaseDir(layer, attempt, tier, child, name, null));
    } else {
      // Cell-grouped: walk one level deeper.
      for (const caseName of fs.readdirSync(child).sort()) {
        const caseDir = path.join(child, caseName);
        if (fs.statSync(caseDir).isDirectory() && isCaseDir(caseDir)) {
          entries.push(readCaseDir(layer, attempt, tier, caseDir, caseName, name));
        }
      }
    }
  }

  // Fall back to legacy flat layout only when no sub-dirs exist.
  if (!hasDirs && entries.length === 0) {
    return readFlatEntries(layer, attempt, tier, dir);
  }
  return entries;
}

/** Build the `/samples` payload (same shape as the FastAPI endpoint). */
export function listSamples(layer, attempt) {
  const out = {
    attempt,
    layer,
    canonical_seed: CANONICAL_SEED_DEFAULT,
    expected: [],
    showcase: [],
  };
  for (const tier of TIERS) {
    out[tier] = scanTier(layer, attempt, tier);
  }
  return out;
}

/**
 * Resolve a `samples/<tier>/<relPath>` request to an absolute file path,
 * guarding against traversal. Returns null for illegal tiers/paths.
 */
export function resolveSampleWavPath(layer, attempt, tier, relPath) {
  if (!TIERS.includes(tier)) return null;
  const parts = relPath
    .replace(/\\/g, "/")
    .split("/")
    .filter((p) => p !== "" && p !== ".");
  if (parts.some((p) => p === "..")) return null;
  return path.join(attemptRoot(layer, attempt), tier, ...parts);
}

export const samplesLayersRoot = LAYERS_ROOT;
