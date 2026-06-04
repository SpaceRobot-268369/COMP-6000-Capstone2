import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import cors from "cors";
import express from "express";
import session from "express-session";
import connectPgSimple from "connect-pg-simple";
import pg from "pg";

import { listSamples, resolveSampleWavPath } from "./samples.js";

const app = express();
const port = Number(process.env.PORT || 4000);
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

const PgSession = connectPgSimple(session);

// Compare origins case-insensitively: browsers lowercase the scheme+host of
// the Origin header, so an allow-list entry with different casing would never
// match an otherwise-valid origin.
const allowedOrigins = new Set(
  (process.env.FRONTEND_URL || "http://localhost:5173,http://127.0.0.1:5173")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean),
);
app.use(cors({
  origin(origin, callback) {
    if (!origin) return callback(null, false);
    if (allowedOrigins.has(origin.toLowerCase())) return callback(null, origin);
    // Reject cleanly: omit the Access-Control-Allow-Origin header so the
    // browser blocks the response as a CORS error. Passing an Error here would
    // hit Express's default error handler and surface a confusing 500 instead.
    return callback(null, false);
  },
  credentials: true,
}));
app.use(express.json());
app.use(session({
  store: new PgSession({
    pool,
    tableName: "sessions",
    pruneSessionInterval: 60 * 15, // prune expired sessions every 15 min
  }),
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
  },
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function query(text, params = []) {
  return pool.query(text, params);
}

function normalizeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function scryptPassword(password, salt) {
  return new Promise((resolve, reject) => {
    crypto.scrypt(password, salt, 64, (err, key) => {
      if (err) reject(err);
      else resolve(key);
    });
  });
}

async function hashPassword(password) {
  const salt = crypto.randomBytes(16).toString("hex");
  const key = await scryptPassword(password, salt);
  return `${salt}:${key.toString("hex")}`;
}

async function verifyPassword(password, storedHash) {
  const [salt, expected] = storedHash.split(":");
  if (!salt || !expected) return false;
  const key = await scryptPassword(password, salt);
  const expectedBuf = Buffer.from(expected, "hex");
  if (expectedBuf.length !== key.length) return false;
  return crypto.timingSafeEqual(expectedBuf, key);
}

function requireAuth(req, res, next) {
  // AUTH TEMPORARILY DISABLED — re-enable before production
  // if (!req.session?.userId) {
  //   return res.status(401).json({ ok: false, message: "Not authenticated." });
  // }
  next();
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------
app.get("/api/health", async (_req, res) => {
  try {
    const { rows } = await query("SELECT NOW() AS now");
    res.json({ ok: true, db: "connected", now: rows[0].now });
  } catch (err) {
    res.status(500).json({ ok: false, db: "error", message: String(err.message || err) });
  }
});

app.post("/api/register", async (req, res) => {
  const username = normalizeString(req.body.username);
  const email    = normalizeString(req.body.email).toLowerCase();
  const password = typeof req.body.password === "string" ? req.body.password : "";

  if (!username || !email || !password) {
    return res.status(400).json({ ok: false, message: "Username, email, and password are required." });
  }

  try {
    const passwordHash = await hashPassword(password);
    const { rows } = await query(
      `INSERT INTO users (username, email, password_hash)
       VALUES ($1, $2, $3)
       RETURNING id, username, email, created_at`,
      [username, email, passwordHash],
    );

    const user = rows[0];
    req.session.userId   = user.id;
    req.session.username = user.username;

    res.status(201).json({ ok: true, user: { id: user.id, username: user.username, email: user.email } });
  } catch (err) {
    if (err.code === "23505") {
      const message = err.constraint === "users_email_key"
        ? "This email is already registered."
        : "This username is already taken.";
      return res.status(409).json({ ok: false, message });
    }
    console.error("Register failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

app.post("/api/login", async (req, res) => {
  const account  = normalizeString(req.body.account);
  const password = typeof req.body.password === "string" ? req.body.password : "";

  if (!account || !password) {
    return res.status(400).json({ ok: false, message: "Account and password are required." });
  }

  try {
    const { rows } = await query(
      `SELECT id, username, email, password_hash
       FROM users
       WHERE username = $1 OR email = $1
       LIMIT 1`,
      [account],
    );

    const user = rows[0];
    if (!user || !(await verifyPassword(password, user.password_hash))) {
      return res.status(401).json({ ok: false, message: "Invalid account or password." });
    }

    req.session.userId   = user.id;
    req.session.username = user.username;

    res.json({ ok: true, user: { id: user.id, username: user.username, email: user.email } });
  } catch (err) {
    console.error("Login failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

app.post("/api/logout", (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      console.error("Logout failed:", err);
      return res.status(500).json({ ok: false, message: "Logout failed." });
    }
    res.clearCookie("connect.sid");
    res.json({ ok: true });
  });
});

app.get("/api/me", requireAuth, async (req, res) => {
  try {
    const { rows } = await query(
      `SELECT id, username, email, created_at FROM users WHERE id = $1`,
      [req.session.userId],
    );
    if (!rows[0]) return res.status(404).json({ ok: false, message: "User not found." });
    res.json({ ok: true, user: rows[0] });
  } catch (err) {
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

// ---------------------------------------------------------------------------
// AI routes — proxy to FastAPI inference server (port 8000)
// ---------------------------------------------------------------------------
const DEFAULT_AI_SERVER = "http://ai-tunnel:8000";
const AI_SERVER = (process.env.AI_SERVER_URL || DEFAULT_AI_SERVER).replace(/\/+$/, "");
const AI_SERVER_LABEL = process.env.AI_SERVER_LABEL || aiServerLabelFromUrl(AI_SERVER);
const AI_CONNECTION_MODE = process.env.AI_CONNECTION_MODE || "direct";
const AI_TUNNEL_LOCAL_PORT = process.env.AI_TUNNEL_LOCAL_PORT || "8000";
const AI_TUNNEL_REMOTE_HOST = process.env.AI_TUNNEL_REMOTE_HOST || "127.0.0.1";
const AI_TUNNEL_REMOTE_PORT = process.env.AI_TUNNEL_REMOTE_PORT || "8000";
const aiRequestTimeoutMs = Number(process.env.AI_REQUEST_TIMEOUT_MS || 15000);
const AI_REQUEST_TIMEOUT_MS = Number.isFinite(aiRequestTimeoutMs) && aiRequestTimeoutMs > 0
  ? aiRequestTimeoutMs
  : 15000;

class AiProxyError extends Error {
  constructor({ message, stage, status = 502, detail, hints = [], cause }) {
    super(message);
    this.name = "AiProxyError";
    this.stage = stage;
    this.status = status;
    this.detail = detail;
    this.hints = hints;
    this.cause = cause;
  }
}

function aiServerLabelFromUrl(value) {
  try {
    return new URL(value).hostname.split(".")[0] || "AI server";
  } catch {
    return "AI server";
  }
}

function aiServerBaseUrl() {
  try {
    return new URL(AI_SERVER);
  } catch (err) {
    throw new AiProxyError({
      message: "AI_SERVER_URL is invalid, so the backend cannot determine which AI server to use.",
      stage: "ai-config",
      status: 500,
      detail: `AI_SERVER_URL="${AI_SERVER}" is not a valid URL.`,
      hints: ["Check AI_SERVER_URL in services/dev/.env, for example http://ai-tunnel:8000."],
      cause: err,
    });
  }
}

function aiServerEndpoint(path) {
  const base = aiServerBaseUrl();
  return new URL(path, `${base.origin}/`);
}

function aiConnectionHints(port) {
  if (AI_CONNECTION_MODE === "ssh_tunnel") {
    return [
      "Check the Docker Compose ai-tunnel service logs and confirm the tunnel container is healthy.",
      `Confirm the tunnel maps ai-tunnel:${AI_TUNNEL_LOCAL_PORT} -> ${AI_SERVER_LABEL}:${AI_TUNNEL_REMOTE_HOST}:${AI_TUNNEL_REMOTE_PORT}.`,
      "If ai-tunnel logs show Permission denied (publickey), the pem file is likely incorrect or the SSH username is not ubuntu.",
      `Confirm the ${AI_SERVER_LABEL} machine is RUNNING and the AI service is listening on serverB at ${AI_TUNNEL_REMOTE_HOST}:${AI_TUNNEL_REMOTE_PORT}.`,
    ];
  }

  return [
    `Confirm the ${AI_SERVER_LABEL} machine is RUNNING.`,
    `Confirm FastAPI/uvicorn is listening on ${AI_SERVER_LABEL} at 0.0.0.0:${port}.`,
    "If port 8000 is exposed through an SSH tunnel, check the tunnel process, SSH user, pem file, and pem permissions.",
  ];
}

function aiFetchError(err, targetUrl, operation) {
  const cause = err.cause || err;
  const code = cause.code || cause.name || err.code || err.name;
  const port = targetUrl.port || (targetUrl.protocol === "https:" ? "443" : "80");
  const commonHints = aiConnectionHints(port);

  if (err.name === "AbortError" || code === "ABORT_ERR") {
    const tunnelMessage = `${AI_SERVER_LABEL} SSH tunnel timed out: ai-tunnel may not be running, may be unhealthy, or the serverB AI service is not responding.`;
    return new AiProxyError({
      message: AI_CONNECTION_MODE === "ssh_tunnel"
        ? tunnelMessage
        : `${AI_SERVER_LABEL} connection timed out: ${AI_SERVER_LABEL} may be stopped, the AI service may be stopped, or port ${port}/firewall may be unreachable.`,
      stage: AI_CONNECTION_MODE === "ssh_tunnel" ? "ai-tunnel-timeout" : "ai-connect-timeout",
      status: 504,
      detail: `${operation} timed out after ${AI_REQUEST_TIMEOUT_MS}ms while connecting to ${targetUrl.origin}.`,
      hints: commonHints,
      cause: err,
    });
  }

  if (code === "ENOTFOUND" || code === "EAI_AGAIN") {
    const tunnelMessage = targetUrl.hostname === "ai-tunnel"
      ? "Docker backend could not resolve ai-tunnel: the Compose ai-tunnel service may be stopped or on a different network."
      : `${AI_SERVER_LABEL} hostname could not be resolved: check the AI_SERVER_URL hostname.`;
    return new AiProxyError({
      message: tunnelMessage,
      stage: AI_CONNECTION_MODE === "ssh_tunnel" ? "ai-tunnel-dns" : "ai-dns",
      status: 502,
      detail: `${operation} could not resolve ${targetUrl.hostname}: ${cause.message || err.message}`,
      hints: [`Check services/dev/.env and confirm AI_SERVER_URL is currently ${AI_SERVER}.`],
      cause: err,
    });
  }

  if (code === "ECONNREFUSED") {
    const tunnelMessage = `${AI_SERVER_LABEL} SSH tunnel is not running, or the ai-tunnel container is not listening on port ${AI_TUNNEL_LOCAL_PORT}.`;
    return new AiProxyError({
      message: AI_CONNECTION_MODE === "ssh_tunnel"
        ? tunnelMessage
        : `${AI_SERVER_LABEL} was reached, but port ${port} refused the connection: the AI service may be stopped or not listening on 0.0.0.0:${port}.`,
      stage: AI_CONNECTION_MODE === "ssh_tunnel" ? "ai-tunnel-not-running" : "ai-port-refused",
      status: 503,
      detail: `${operation} was refused by ${targetUrl.origin}: ${cause.message || err.message}`,
      hints: commonHints,
      cause: err,
    });
  }

  if (code === "ETIMEDOUT" || code === "UND_ERR_CONNECT_TIMEOUT" || code === "EHOSTUNREACH" || code === "ENETUNREACH") {
    const tunnelMessage = `${AI_SERVER_LABEL} SSH tunnel is unreachable: ai-tunnel may be stopped, unhealthy, or inaccessible from the Docker backend at ${targetUrl.origin}.`;
    return new AiProxyError({
      message: AI_CONNECTION_MODE === "ssh_tunnel"
        ? tunnelMessage
        : `${AI_SERVER_LABEL} cannot be reached: ${AI_SERVER_LABEL} may be stopped, network/firewall access may be blocked, or serverB may not expose port ${port}.`,
      stage: AI_CONNECTION_MODE === "ssh_tunnel" ? "ai-tunnel-unreachable" : "ai-network-unreachable",
      status: 504,
      detail: `${operation} could not reach ${targetUrl.origin}: ${cause.message || err.message}`,
      hints: commonHints,
      cause: err,
    });
  }

  if (code === "UND_ERR_HEADERS_TIMEOUT") {
    return new AiProxyError({
      message: `${AI_SERVER_LABEL} connected but timed out waiting for response headers: the AI service may be stuck loading a model or running inference.`,
      stage: "ai-response-timeout",
      status: 504,
      detail: `${operation} did not receive response headers from ${targetUrl.origin}: ${cause.message || err.message}`,
      hints: commonHints,
      cause: err,
    });
  }

  if (code === "ECONNRESET") {
    return new AiProxyError({
      message: `${AI_SERVER_LABEL} connection was reset: the AI service process may have crashed, restarted, or closed the connection mid-request.`,
      stage: "ai-connection-reset",
      status: 502,
      detail: `${operation} connection reset by ${targetUrl.origin}: ${cause.message || err.message}`,
      hints: commonHints,
      cause: err,
    });
  }

  return new AiProxyError({
    message: `${AI_SERVER_LABEL} request failed: an unknown error occurred in the backend-to-AI proxy path.`,
    stage: "ai-proxy",
    status: 502,
    detail: `${operation} failed for ${targetUrl.origin}: ${cause.message || err.message}`,
    hints: commonHints,
    cause: err,
  });
}

async function fetchAi(path, options = {}, operation = "AI request") {
  const targetUrl = aiServerEndpoint(path);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), AI_REQUEST_TIMEOUT_MS);

  try {
    return await fetch(targetUrl, {
      ...options,
      signal: controller.signal,
    });
  } catch (err) {
    throw aiFetchError(err, targetUrl, operation);
  } finally {
    clearTimeout(timeout);
  }
}

async function readAiJson(response, operation) {
  try {
    return await response.json();
  } catch (err) {
    throw new AiProxyError({
      message: `${AI_SERVER_LABEL} returned a non-JSON response: the AI service may have returned an error page, crash log, or reverse proxy error.`,
      stage: "ai-invalid-json",
      status: 502,
      detail: `${operation} received HTTP ${response.status}, but the body was not valid JSON.`,
      hints: [`Check the uvicorn/FastAPI logs on ${AI_SERVER_LABEL}.`],
      cause: err,
    });
  }
}

function sendAiProxyError(res, err, operation) {
  const status = err instanceof AiProxyError ? err.status : 502;
  const payload = {
    ok: false,
    message: err.message || `${AI_SERVER_LABEL} request failed.`,
    stage: err.stage || "ai-proxy",
    aiServer: {
      label: AI_SERVER_LABEL,
      url: AI_SERVER,
    },
    detail: err.detail || String(err.message || err),
    hints: err.hints || [],
  };

  console.error(`[AI proxy] ${operation} failed`, payload, err.cause || err);
  res.status(status).json(payload);
}

function sendAiUpstreamError(res, response, body, operation) {
  const payload = {
    ok: false,
    message: body?.message || `${AI_SERVER_LABEL} AI service returned HTTP ${response.status} while ${operation}.`,
    stage: "ai-upstream-response",
    aiServer: {
      label: AI_SERVER_LABEL,
      url: AI_SERVER,
    },
    upstreamStatus: response.status,
    upstream: body,
  };

  console.error(`[AI proxy] ${operation} upstream error`, payload);
  res.status(response.status).json(payload);
}

console.log(`[AI proxy] forwarding AI requests to ${AI_SERVER_LABEL} at ${AI_SERVER} (${AI_CONNECTION_MODE})`);

app.get("/api/ai/health", async (_req, res) => {
  const operation = "AI health check";
  try {
    const r = await fetchAi("/health", {}, operation);
    const body = await readAiJson(r, operation);
    if (!r.ok) return sendAiUpstreamError(res, r, body, operation);
    res.json(body);
  } catch (err) {
    sendAiProxyError(res, err, operation);
  }
});

// Layer registry — dropdown population
app.get("/api/layers", requireAuth, async (_req, res) => {
  const operation = "list layer registry";
  try {
    const r = await fetchAi("/layers", {}, operation);
    const body = await readAiJson(r, operation);
    if (!r.ok) return sendAiUpstreamError(res, r, body, operation);
    res.status(r.status).json(body);
  } catch (err) {
    sendAiProxyError(res, err, operation);
  }
});

// Cached samples for an attempt — drives the "Expected Results" preview in the
// frontend. Served straight from the repo checkout (AI_LAYERS_ROOT), NOT the
// AI worker: these are static artefacts with no model dependency, so serverB
// needn't be awake to view them. See backend/src/samples.js.
app.get("/api/layers/:layer/attempts/:attempt/samples", requireAuth, (req, res) => {
  const { layer, attempt } = req.params;
  try {
    res.json(listSamples(layer, attempt));
  } catch (err) {
    res.status(500).json({
      ok: false,
      message: `Failed to list samples for ${layer}/${attempt}: ${err.message || err}`,
    });
  }
});

// Stream a cached sample WAV from the local checkout (browser plays it via
// <audio>). `relPath` is the remainder after the tier — supports all three
// layouts (flat, case-dir, cell-grouped). WAVs are DVC-tracked: a missing blob
// (only the .dvc pointer present) yields a 404 with a `dvc pull` hint.
app.get("/api/layers/:layer/attempts/:attempt/samples/:tier/*", requireAuth, (req, res) => {
  const { layer, attempt, tier } = req.params;
  const relPath = req.params[0] || "";
  if (!relPath.endsWith(".wav")) {
    return res.status(404).json({ message: "only .wav samples are served" });
  }
  const filePath = resolveSampleWavPath(layer, attempt, tier, relPath);
  if (!filePath) {
    return res.status(404).json({ message: `illegal sample path: ${tier}/${relPath}` });
  }
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({
      message: `WAV not materialised locally (${path.basename(filePath)}). Run \`dvc pull\` then retry.`,
    });
  }
  res.type("audio/wav");
  res.sendFile(filePath);
});

// Per-attempt generation. Forwarded runtime params: `seed` / `retrieval_seed`,
// optional
// `(season, diel)` for bank attempts, and Layer B weather-stem controls.
// The handler picks up every other parameter from the attempt's registry entry.
const ALLOWED_SEASONS = new Set(["spring", "summer", "autumn", "winter"]);
const ALLOWED_DIELS = new Set(["dawn", "morning", "afternoon", "night"]);
const ALLOWED_WEATHER_TYPES = new Set(["rain", "wind", "thunder", "storm"]);
const ALLOWED_WEATHER_INTENSITIES = new Set(["light", "medium", "heavy"]);

app.post("/api/layers/:layer/attempts/:attempt/generate", requireAuth, async (req, res) => {
  const { layer, attempt } = req.params;
  const operation = `generate ${layer}/${attempt}`;
  try {
    const requestedSeed = Number(req.body?.seed);
    const requestedRetrievalSeed = Number(req.body?.retrieval_seed);
    const seed = Number.isInteger(requestedSeed) && requestedSeed >= 0 ? requestedSeed : null;
    const retrievalSeed = Number.isInteger(requestedRetrievalSeed) && requestedRetrievalSeed >= 0
      ? requestedRetrievalSeed
      : null;

    // Cell selector — only forwarded if both are valid; otherwise omitted
    // (server falls back to the attempt's default_cell).
    const season = typeof req.body?.season === "string" ? req.body.season.toLowerCase() : null;
    const diel = typeof req.body?.diel === "string" ? req.body.diel.toLowerCase() : null;
    const payload = { seed };
    if (retrievalSeed !== null) {
      payload.retrieval_seed = retrievalSeed;
    }
    if (ALLOWED_SEASONS.has(season) && ALLOWED_DIELS.has(diel)) {
      payload.season = season;
      payload.diel = diel;
    }
    const weatherType = typeof req.body?.weather_type === "string"
      ? req.body.weather_type.toLowerCase()
      : null;
    const intensity = typeof req.body?.intensity === "string"
      ? req.body.intensity.toLowerCase()
      : null;
    const requestedDuration = Number(req.body?.duration_s);
    if (ALLOWED_WEATHER_TYPES.has(weatherType)) {
      payload.weather_type = weatherType;
    }
    if (ALLOWED_WEATHER_INTENSITIES.has(intensity)) {
      payload.intensity = intensity;
    }
    if (Number.isFinite(requestedDuration) && requestedDuration > 0 && requestedDuration <= 30) {
      payload.duration_s = requestedDuration;
    }

    const r = await fetchAi(
      `/layers/${encodeURIComponent(layer)}/attempts/${encodeURIComponent(attempt)}/generate`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      },
      operation,
    );
    const body = await readAiJson(r, operation);
    if (!r.ok) return sendAiUpstreamError(res, r, body, operation);
    res.status(r.status).json(body);
  } catch (err) {
    sendAiProxyError(res, err, operation);
  }
});

// Per-attempt upload analysis (Layer E). Unlike /generate (JSON seed), this
// forwards a multipart audio upload. `express.json()` ignores non-JSON bodies,
// so the raw multipart stream is still intact on `req` and we pipe it straight
// to the AI worker (boundary + content-type preserved) without buffering.
app.post("/api/layers/:layer/attempts/:attempt/analyze", requireAuth, async (req, res) => {
  const { layer, attempt } = req.params;
  const operation = `analyze ${layer}/${attempt}`;
  try {
    const r = await fetchAi(
      `/layers/${encodeURIComponent(layer)}/attempts/${encodeURIComponent(attempt)}/analyze`,
      {
        method: "POST",
        headers: { "content-type": req.headers["content-type"] || "application/octet-stream" },
        body: req,
        duplex: "half",
      },
      operation,
    );
    const body = await readAiJson(r, operation);
    if (!r.ok) return sendAiUpstreamError(res, r, body, operation);
    res.status(r.status).json(body);
  } catch (err) {
    sendAiProxyError(res, err, operation);
  }
});

app.get("/", (_req, res) => {
  res.json({ service: "backend", status: "running" });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
app.listen(port, "0.0.0.0", () => {
  console.log(`Backend running on http://0.0.0.0:${port}`);
});
