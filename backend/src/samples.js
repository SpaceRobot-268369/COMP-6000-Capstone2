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
 * PNG/JSON contents are inlined (small). `wav_url` is returned only when the
 * real WAV is materialised locally. When only the `.dvc` pointer is present,
 * `wav_dvc` is true so the frontend can show a pull hint instead of rendering
 * a broken audio player.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Root of the attempts tree. In Docker the acoustic_ai dir is mounted read-only
// (see services/dev/docker-compose.yml); on a native host point this at the
// repo checkout's acoustic_ai/layers. Some local stacks provide the acoustic_ai
// root instead, so accept both shapes.
const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(MODULE_DIR, "../..");
const REPO_LAYERS_ROOT = path.resolve(MODULE_DIR, "../../acoustic_ai/layers");
const REGISTRY_PATH = path.join(REPO_ROOT, "acoustic_ai/registry.yaml");

function normalizeLayersRoot(root) {
  return path.basename(root) === "layers" ? root : path.join(root, "layers");
}

function resolveLayersRoot() {
  if (process.env.AI_LAYERS_ROOT) {
    return normalizeLayersRoot(process.env.AI_LAYERS_ROOT);
  }
  if (fs.existsSync(REPO_LAYERS_ROOT)) return REPO_LAYERS_ROOT;
  return "/acoustic_ai/layers";
}

const LAYERS_ROOT = resolveLayersRoot();

const TIERS = ["expected", "showcase"];
const CANONICAL_SEED_DEFAULT = 42;
const ASSET_BANK_PREFIX = "__asset_bank__";

function attemptRoot(layer, attempt) {
  return path.join(LAYERS_ROOT, layer, "attempts", attempt);
}

function registryAssetBank(layer, attempt) {
  if (!fs.existsSync(REGISTRY_PATH)) return null;
  const lines = fs.readFileSync(REGISTRY_PATH, "utf-8").split(/\r?\n/);
  const layerLine = `  ${layer}:`;
  const attemptLine = `      ${attempt}:`;
  let inLayer = false;
  let inAttempt = false;

  for (const line of lines) {
    if (line.startsWith("  ") && !line.startsWith("    ")) {
      inLayer = line.trim() === layerLine.trim();
      inAttempt = false;
      continue;
    }
    if (!inLayer) continue;
    if (line.startsWith("      ") && !line.startsWith("        ")) {
      inAttempt = line.trim() === attemptLine.trim();
      continue;
    }
    if (!inAttempt) continue;
    const match = line.match(/^\s+asset_bank:\s*(.+?)\s*$/);
    if (match) return path.resolve(REPO_ROOT, match[1].replace(/^["']|["']$/g, ""));
  }
  return null;
}

function safeJoin(root, relPath) {
  const parts = relPath
    .replace(/\\/g, "/")
    .split("/")
    .filter((p) => p !== "" && p !== ".");
  if (parts.some((p) => p === "..")) return null;
  const full = path.resolve(root, ...parts);
  const rootResolved = path.resolve(root);
  if (full !== rootResolved && !full.startsWith(`${rootResolved}${path.sep}`)) return null;
  return full;
}

function assetBankMelPath(audioPath) {
  const dir = path.dirname(audioPath);
  const base = path.basename(audioPath);
  if (base === "crop_bandpass.wav") return path.join(dir, "mel_bandpass.png");
  if (base.endsWith(".wav")) return path.join(dir, `${base.slice(0, -4)}.png`);
  return path.join(dir, "mel_bandpass.png");
}

function readAssetBankExpected(layer, attempt) {
  const bankDir = registryAssetBank(layer, attempt);
  if (!bankDir) return [];
  const indexPath = path.join(bankDir, "index.json");
  if (!fs.existsSync(indexPath)) return [];

  let doc;
  try {
    doc = JSON.parse(fs.readFileSync(indexPath, "utf-8"));
  } catch {
    return [];
  }

  const bySpecies = new Map();
  for (const asset of doc.assets || []) {
    const attrs = asset.attributes || {};
    const slug = attrs.event_type || attrs.species_slug;
    const commonName = attrs.species_common_name;
    const audioPath = asset.audio_path;
    if (!slug || !commonName || !audioPath || bySpecies.has(slug)) continue;
    bySpecies.set(slug, { asset, attrs, slug, commonName });
  }

  return [...bySpecies.values()]
    .sort((a, b) => a.commonName.localeCompare(b.commonName))
    .map(({ asset, attrs, slug, commonName }) => {
      const audioPath = asset.audio_path;
      const audioFull = safeJoin(bankDir, audioPath);
      const pngRel = assetBankMelPath(audioPath);
      const pngFull = safeJoin(bankDir, pngRel);
      const metadataRel = path.join(path.dirname(audioPath), "metadata.json");
      const metadataFull = safeJoin(bankDir, metadataRel);
      const stem = asset.id || `${slug}_reference`;
      const relWav = [
        ASSET_BANK_PREFIX,
        ...audioPath.replace(/\\/g, "/").split("/").filter(Boolean),
      ].join("/");
      const entry = {
        stem,
        cell: null,
        has_wav: Boolean(audioFull && fs.existsSync(audioFull)),
        wav_dvc: Boolean(!(audioFull && fs.existsSync(audioFull))),
        has_png: Boolean((pngFull && fs.existsSync(pngFull)) || pngRel),
        has_json: true,
        png_b64: null,
        metadata: {
          ...attrs,
          species_common_name: commonName,
          species_scientific_name: attrs.species_scientific_name || "",
          species_slug: slug,
          expected_source: "retrieval_asset_bank",
          expected_item: asset.id || stem,
          display_title: `${commonName} · retrieval reference`,
          display_audio: path.basename(audioPath),
          display_spectrogram: path.basename(pngRel),
          asset_bank_audio_path: audioPath,
        },
        wav_url: null,
      };

      if (entry.has_wav) {
        entry.wav_url = `/layers/${layer}/attempts/${attempt}/samples/expected/${relWav}`;
      }

      if (pngFull && fs.existsSync(pngFull)) {
        try {
          entry.png_b64 = fs.readFileSync(pngFull).toString("base64");
        } catch {
          /* leave null */
        }
      }
      if (metadataFull && fs.existsSync(metadataFull)) {
        try {
          entry.metadata = {
            ...entry.metadata,
            ...JSON.parse(fs.readFileSync(metadataFull, "utf-8")),
            species_common_name: commonName,
            species_scientific_name: attrs.species_scientific_name || "",
            species_slug: slug,
          };
        } catch {
          /* keep index metadata */
        }
      }
      return entry;
    });
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
    wav_dvc: false,
    has_png: false,
    has_json: false,
    png_b64: null,
    metadata: null,
    wav_url: null,
  };

  if (fs.existsSync(path.join(caseDir, "audio.wav"))) {
    entry.has_wav = true;
    entry.wav_url = `/layers/${layer}/attempts/${attempt}/samples/${relWav}`;
  } else if (fs.existsSync(path.join(caseDir, "audio.wav.dvc"))) {
    entry.wav_dvc = true;
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
        wav_dvc: false,
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
      e.wav_dvc = true;
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
  const assetBankExpected = readAssetBankExpected(layer, attempt);
  if (assetBankExpected.length) {
    out.expected = assetBankExpected;
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
  if (tier === "expected" && parts[0] === ASSET_BANK_PREFIX) {
    const bankDir = registryAssetBank(layer, attempt);
    if (!bankDir) return null;
    return safeJoin(bankDir, parts.slice(1).join("/"));
  }
  return path.join(attemptRoot(layer, attempt), tier, ...parts);
}

export const samplesLayersRoot = LAYERS_ROOT;
