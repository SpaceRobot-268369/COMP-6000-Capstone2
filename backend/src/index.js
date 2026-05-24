import crypto from "node:crypto";
import cors from "cors";
import express from "express";
import session from "express-session";
import connectPgSimple from "connect-pg-simple";
import pg from "pg";

const app = express();
const port = Number(process.env.PORT || 4000);
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });
const sessionCookieSecure = process.env.SESSION_COOKIE_SECURE
  ? process.env.SESSION_COOKIE_SECURE === "true"
  : process.env.NODE_ENV === "production";

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
    secure: sessionCookieSecure,
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

function safeTokenEqual(actual, expected) {
  const actualBuf = Buffer.from(actual);
  const expectedBuf = Buffer.from(expected);
  if (actualBuf.length !== expectedBuf.length) return false;
  return crypto.timingSafeEqual(actualBuf, expectedBuf);
}

function requireAuth(req, res, next) {
  if (!req.session?.userId) {
    return res.status(401).json({ ok: false, message: "Not authenticated." });
  }
  next();
}

function requireWorkerAuth(req, res, next) {
  const expectedToken = process.env.WORKER_API_TOKEN;
  if (!expectedToken) {
    return res.status(500).json({ ok: false, message: "Worker API token is not configured." });
  }

  const authHeader = req.get("authorization") || "";
  const [scheme, token] = authHeader.split(" ");
  if (scheme !== "Bearer" || !token) {
    return res.status(401).json({ ok: false, message: "Worker authorization token is required." });
  }

  if (!safeTokenEqual(token, expectedToken)) {
    return res.status(403).json({ ok: false, message: "Invalid worker authorization token." });
  }

  next();
}

const JOB_TYPES = new Set(["generation", "training"]);
const USER_CANCELLABLE_JOB_STATUSES = new Set(["queued", "claimed", "running", "uploading"]);
const WORKER_SETTABLE_JOB_STATUSES = new Set(["running", "uploading", "completed", "failed", "cancelled"]);
const WORKER_STATUS_TRANSITIONS = {
  claimed: new Set(["running", "failed"]),
  running: new Set(["uploading", "failed"]),
  uploading: new Set(["completed", "failed"]),
  cancel_requested: new Set(["cancelled", "failed"]),
};

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

app.post("/api/jobs/:id/cancel", requireAuth, async (req, res) => {
  const jobId = Number(req.params.id);
  if (!Number.isInteger(jobId) || jobId <= 0) {
    return res.status(400).json({ ok: false, message: "Invalid job id." });
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    const { rows: currentRows } = await client.query(
      `SELECT *
       FROM jobs
       WHERE id = $1 AND created_by = $2
       FOR UPDATE`,
      [jobId, req.session.userId],
    );

    const current = currentRows[0];
    if (!current) {
      await client.query("ROLLBACK");
      return res.status(404).json({ ok: false, message: "Job not found." });
    }

    if (!USER_CANCELLABLE_JOB_STATUSES.has(current.status)) {
      await client.query("ROLLBACK");
      return res.status(409).json({
        ok: false,
        message: `Job cannot be cancelled from status ${current.status}.`,
      });
    }

    const nextStatus = current.status === "queued" ? "cancelled" : "cancel_requested";
    const { rows } = await client.query(
      `UPDATE jobs
       SET status = $1,
           finished_at = CASE WHEN $1 = 'cancelled' THEN NOW() ELSE finished_at END
       WHERE id = $2 AND created_by = $3
       RETURNING *`,
      [nextStatus, jobId, req.session.userId],
    );

    await client.query("COMMIT");
    res.json({ ok: true, job: serializeJob(rows[0]) });
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    console.error("Cancel job failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  } finally {
    client.release();
  }
});

// ---------------------------------------------------------------------------
// Worker job routes
// ---------------------------------------------------------------------------
app.post("/api/worker/jobs/claim", requireWorkerAuth, async (req, res) => {
  const workerId = normalizeString(req.body?.worker_id);
  const requestedTypes = Array.isArray(req.body?.types) && req.body.types.length > 0
    ? req.body.types.map(normalizeString)
    : ["generation", "training"];
  const types = requestedTypes.filter((type) => JOB_TYPES.has(type));

  if (!workerId) {
    return res.status(400).json({ ok: false, message: "worker_id is required." });
  }

  if (types.length !== requestedTypes.length) {
    return res.status(400).json({ ok: false, message: "Worker types must be generation or training." });
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    const { rows: candidates } = await client.query(
      `SELECT id
       FROM jobs
       WHERE status = 'queued'
         AND type = ANY($1::text[])
         AND attempt_count < max_attempts
       ORDER BY priority DESC, created_at ASC
       FOR UPDATE SKIP LOCKED
       LIMIT 1`,
      [types],
    );

    if (!candidates[0]) {
      await client.query("COMMIT");
      return res.json({ ok: true, job: null });
    }

    const { rows } = await client.query(
      `UPDATE jobs
       SET status = 'claimed',
           claimed_by = $2,
           claimed_at = NOW(),
           heartbeat_at = NOW(),
           attempt_count = attempt_count + 1
       WHERE id = $1
       RETURNING *`,
      [candidates[0].id, workerId],
    );

    await client.query("COMMIT");
    res.json({ ok: true, job: serializeJob(rows[0]) });
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    console.error("Claim job failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  } finally {
    client.release();
  }
});

app.post("/api/worker/jobs/:id/heartbeat", requireWorkerAuth, async (req, res) => {
  const jobId = Number(req.params.id);
  const workerId = normalizeString(req.body?.worker_id);

  if (!Number.isInteger(jobId) || jobId <= 0) {
    return res.status(400).json({ ok: false, message: "Invalid job id." });
  }

  if (!workerId) {
    return res.status(400).json({ ok: false, message: "worker_id is required." });
  }

  try {
    const { rows } = await query(
      `UPDATE jobs
       SET heartbeat_at = NOW()
       WHERE id = $1
         AND claimed_by = $2
         AND status IN ('claimed', 'running', 'uploading', 'cancel_requested')
       RETURNING *`,
      [jobId, workerId],
    );

    if (!rows[0]) {
      return res.status(404).json({ ok: false, message: "Active job not found for worker." });
    }

    res.json({ ok: true, job: serializeJob(rows[0]) });
  } catch (err) {
    console.error("Heartbeat failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

app.post("/api/worker/jobs/:id/status", requireWorkerAuth, async (req, res) => {
  const jobId = Number(req.params.id);
  const workerId = normalizeString(req.body?.worker_id);
  const nextStatus = normalizeString(req.body?.status);
  const result = req.body?.result;
  const artifactUri = req.body?.artifact_uri ?? null;
  const logUri = req.body?.log_uri ?? null;
  const errorMessage = req.body?.error_message ?? null;

  if (!Number.isInteger(jobId) || jobId <= 0) {
    return res.status(400).json({ ok: false, message: "Invalid job id." });
  }

  if (!workerId) {
    return res.status(400).json({ ok: false, message: "worker_id is required." });
  }

  if (!WORKER_SETTABLE_JOB_STATUSES.has(nextStatus)) {
    return res.status(400).json({ ok: false, message: "Invalid worker job status." });
  }

  if (result !== undefined && (!result || typeof result !== "object" || Array.isArray(result))) {
    return res.status(400).json({ ok: false, message: "Job result must be an object when provided." });
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    const { rows: currentRows } = await client.query(
      `SELECT *
       FROM jobs
       WHERE id = $1 AND claimed_by = $2
       FOR UPDATE`,
      [jobId, workerId],
    );

    const current = currentRows[0];
    if (!current) {
      await client.query("ROLLBACK");
      return res.status(404).json({ ok: false, message: "Job not found for worker." });
    }

    const allowedNext = WORKER_STATUS_TRANSITIONS[current.status];
    if (!allowedNext?.has(nextStatus)) {
      await client.query("ROLLBACK");
      return res.status(409).json({
        ok: false,
        message: `Invalid job status transition from ${current.status} to ${nextStatus}.`,
      });
    }

    const { rows } = await client.query(
      `UPDATE jobs
       SET status = $1,
           heartbeat_at = NOW(),
           started_at = CASE
               WHEN $1 = 'running' AND started_at IS NULL THEN NOW()
               ELSE started_at
           END,
           finished_at = CASE
               WHEN $1 IN ('completed', 'failed', 'cancelled') THEN NOW()
               ELSE finished_at
           END,
           result = COALESCE($4::jsonb, result),
           artifact_uri = COALESCE($5, artifact_uri),
           log_uri = COALESCE($6, log_uri),
           error_message = COALESCE($7, error_message)
       WHERE id = $2 AND claimed_by = $3
       RETURNING *`,
      [
        nextStatus,
        jobId,
        workerId,
        result === undefined ? null : JSON.stringify(result),
        artifactUri,
        logUri,
        errorMessage,
      ],
    );

    await client.query("COMMIT");
    res.json({ ok: true, job: serializeJob(rows[0]) });
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    console.error("Update job status failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  } finally {
    client.release();
  }
});

app.post("/api/worker/jobs/recover-stale", requireWorkerAuth, async (req, res) => {
  const requestedTimeoutSeconds = Number(req.body?.timeout_seconds ?? 300);
  const timeoutSeconds = Number.isInteger(requestedTimeoutSeconds) && requestedTimeoutSeconds > 0
    ? requestedTimeoutSeconds
    : 300;

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    const { rows: staleJobs } = await client.query(
      `SELECT *
       FROM jobs
       WHERE status IN ('claimed', 'running', 'uploading')
         AND heartbeat_at IS NOT NULL
         AND heartbeat_at < NOW() - ($1::int * INTERVAL '1 second')
       FOR UPDATE SKIP LOCKED`,
      [timeoutSeconds],
    );

    const requeued = [];
    const failed = [];

    for (const job of staleJobs) {
      if (job.attempt_count < job.max_attempts) {
        const { rows } = await client.query(
          `UPDATE jobs
           SET status = 'queued',
               claimed_by = NULL,
               claimed_at = NULL,
               heartbeat_at = NULL,
               started_at = NULL,
               error_message = 'worker heartbeat expired; requeued'
           WHERE id = $1
           RETURNING *`,
          [job.id],
        );
        requeued.push(serializeJob(rows[0]));
      } else {
        const { rows } = await client.query(
          `UPDATE jobs
           SET status = 'failed',
               finished_at = NOW(),
               error_message = 'worker heartbeat expired; max attempts reached'
           WHERE id = $1
           RETURNING *`,
          [job.id],
        );
        failed.push(serializeJob(rows[0]));
      }
    }

    await client.query("COMMIT");
    res.json({
      ok: true,
      timeout_seconds: timeoutSeconds,
      recovered_count: requeued.length + failed.length,
      requeued,
      failed,
    });
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    console.error("Recover stale jobs failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  } finally {
    client.release();
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
