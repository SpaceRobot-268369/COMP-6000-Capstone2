import crypto from "node:crypto";
import cors from "cors";
import express from "express";
import session from "express-session";
import connectPgSimple from "connect-pg-simple";
import pg from "pg";

const app = express();
const port = Number(process.env.PORT || 4000);
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

const PgSession = connectPgSimple(session);

app.use(cors({
  origin: process.env.FRONTEND_URL || "http://localhost:5173",
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
  if (!req.session?.userId) {
    return res.status(401).json({ ok: false, message: "Not authenticated." });
  }
  next();
}

const JOB_TYPES = new Set(["generation", "training"]);

function serializeJob(row) {
  return {
    id: row.id,
    type: row.type,
    status: row.status,
    priority: row.priority,
    payload: row.payload,
    result: row.result,
    artifact_uri: row.artifact_uri,
    log_uri: row.log_uri,
    error_message: row.error_message,
    claimed_by: row.claimed_by,
    claimed_at: row.claimed_at,
    heartbeat_at: row.heartbeat_at,
    started_at: row.started_at,
    finished_at: row.finished_at,
    attempt_count: row.attempt_count,
    max_attempts: row.max_attempts,
    created_by: row.created_by,
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
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
// Job routes
// ---------------------------------------------------------------------------
app.post("/api/jobs", requireAuth, async (req, res) => {
  const type = normalizeString(req.body?.type);
  const payload = req.body?.payload ?? {};
  const requestedPriority = Number(req.body?.priority ?? 0);
  const priority = Number.isInteger(requestedPriority) ? requestedPriority : 0;

  if (!JOB_TYPES.has(type)) {
    return res.status(400).json({ ok: false, message: "Job type must be generation or training." });
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return res.status(400).json({ ok: false, message: "Job payload must be an object." });
  }

  try {
    const { rows } = await query(
      `INSERT INTO jobs (type, status, priority, payload, created_by)
       VALUES ($1, 'queued', $2, $3::jsonb, $4)
       RETURNING *`,
      [type, priority, JSON.stringify(payload), req.session.userId],
    );

    res.status(201).json({ ok: true, job: serializeJob(rows[0]) });
  } catch (err) {
    console.error("Create job failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

app.get("/api/jobs/:id", requireAuth, async (req, res) => {
  const jobId = Number(req.params.id);
  if (!Number.isInteger(jobId) || jobId <= 0) {
    return res.status(400).json({ ok: false, message: "Invalid job id." });
  }

  try {
    const { rows } = await query(
      `SELECT * FROM jobs WHERE id = $1 AND created_by = $2 LIMIT 1`,
      [jobId, req.session.userId],
    );

    if (!rows[0]) {
      return res.status(404).json({ ok: false, message: "Job not found." });
    }

    res.json({ ok: true, job: serializeJob(rows[0]) });
  } catch (err) {
    console.error("Get job failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

// ---------------------------------------------------------------------------
// AI routes — proxy to FastAPI inference server (port 8000)
// ---------------------------------------------------------------------------
const AI_SERVER = process.env.AI_SERVER_URL || "http://localhost:8000";

app.get("/api/ai/health", async (_req, res) => {
  try {
    const r = await fetch(`${AI_SERVER}/health`);
    const body = await r.json();
    res.json(body);
  } catch (err) {
    res.status(503).json({ ok: false, message: "AI server unreachable.", detail: String(err.message) });
  }
});

app.post("/api/analysis", requireAuth, async (req, res) => {
  try {
    // Forward the multipart file + query params to FastAPI
    const url = new URL(`${AI_SERVER}/analysis`);
    Object.entries(req.body).forEach(([k, v]) => url.searchParams.set(k, v));

    const r = await fetch(url.toString(), {
      method: "POST",
      headers: req.headers["content-type"]
        ? { "content-type": req.headers["content-type"] }
        : {},
      body: req,  // stream the raw request body through
      duplex: "half",
    });

    const body = await r.json();
    res.status(r.status).json(body);
  } catch (err) {
    console.error("Analysis proxy failed:", err);
    res.status(502).json({ ok: false, message: "AI server error.", detail: String(err.message) });
  }
});

app.post("/api/generation", requireAuth, async (req, res) => {
  try {
    const r = await fetch(`${AI_SERVER}/generation`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req.body),
    });

    const body = await r.json();
    res.status(r.status).json(body);
  } catch (err) {
    console.error("Generation proxy failed:", err);
    res.status(502).json({ ok: false, message: "AI server error.", detail: String(err.message) });
  }
});

app.post("/api/layer_a/generate", requireAuth, async (req, res) => {
  await proxyLayerAGeneration(req, res, "/layer_a/generate", "Layer A proxy failed:");
});

app.post("/api/layer_a/smoke_test_1/generate", requireAuth, async (req, res) => {
  await proxyLayerAGeneration(req, res, "/layer_a/smoke_test_1/generate", "Layer A smoke test 1 proxy failed:");
});

app.post("/api/layer_a/smoke_test_2/generate", requireAuth, async (req, res) => {
  await proxyLayerAGeneration(req, res, "/layer_a/smoke_test_2/generate", "Layer A smoke test 2 proxy failed:");
});

async function proxyLayerAGeneration(req, res, aiPath, logLabel) {
  try {
    const requestedSeed = Number(req.body?.seed);
    const seed = Number.isInteger(requestedSeed) && requestedSeed >= 0 ? requestedSeed : 42;
    const r = await fetch(`${AI_SERVER}${aiPath}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ seed }),
    });
    const body = await r.json();
    res.status(r.status).json(body);
  } catch (err) {
    console.error(logLabel, err);
    res.status(502).json({ ok: false, message: "AI server error.", detail: String(err.message) });
  }
}

app.get("/", (_req, res) => {
  res.json({ service: "backend", status: "running" });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
app.listen(port, "0.0.0.0", () => {
  console.log(`Backend running on http://0.0.0.0:${port}`);
});
