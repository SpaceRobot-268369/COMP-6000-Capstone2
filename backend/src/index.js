import crypto from "node:crypto";
import cors from "cors";
import express from "express";
import session from "express-session";
import connectPgSimple from "connect-pg-simple";
import pg from "pg";

const app = express();
const port = Number(process.env.PORT || 4000);
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });
const workerApiToken = process.env.WORKER_API_TOKEN || "local-dev-worker-token";

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

async function withTransaction(callback) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const result = await callback(client);
    await client.query("COMMIT");
    return result;
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();
  }
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

function requireWorkerAuth(req, res, next) {
  const header = normalizeString(req.headers.authorization);
  const token = header.startsWith("Bearer ") ? header.slice("Bearer ".length).trim() : "";
  if (!token || token !== workerApiToken) {
    return res.status(401).json({ ok: false, message: "Worker authentication failed." });
  }
  next();
}

async function ensureJobTables() {
  await query(`
    CREATE TABLE IF NOT EXISTS jobs (
      id                 BIGSERIAL PRIMARY KEY,
      job_type           TEXT        NOT NULL CHECK (job_type IN ('generation', 'training')),
      layer              TEXT        NOT NULL DEFAULT 'unknown',
      status             TEXT        NOT NULL DEFAULT 'queued',
      priority           INTEGER     NOT NULL DEFAULT 100,
      created_by         TEXT,
      repo_branch        TEXT,
      command            TEXT,
      params_json        JSONB       NOT NULL DEFAULT '{}'::jsonb,
      output_path        TEXT,
      expected_artifacts JSONB       NOT NULL DEFAULT '[]'::jsonb,
      claimed_by         TEXT,
      lease_until        TIMESTAMPTZ,
      heartbeat_at       TIMESTAMPTZ,
      created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      started_at         TIMESTAMPTZ,
      completed_at       TIMESTAMPTZ,
      error_message      TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_jobs_status_priority ON jobs (status, priority, created_at);
    CREATE INDEX IF NOT EXISTS idx_jobs_claimed_by      ON jobs (claimed_by);
    CREATE INDEX IF NOT EXISTS idx_jobs_lease_until     ON jobs (lease_until);

    CREATE TABLE IF NOT EXISTS workers (
      worker_id      TEXT PRIMARY KEY,
      status         TEXT        NOT NULL DEFAULT 'unknown',
      current_job_id BIGINT      REFERENCES jobs(id) ON DELETE SET NULL,
      metadata_json  JSONB       NOT NULL DEFAULT '{}'::jsonb,
      heartbeat_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS job_events (
      id            BIGSERIAL PRIMARY KEY,
      job_id        BIGINT      NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
      worker_id     TEXT,
      status        TEXT,
      message       TEXT,
      metadata_json JSONB       NOT NULL DEFAULT '{}'::jsonb,
      created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_job_events_job_id_created_at ON job_events (job_id, created_at);

    CREATE TABLE IF NOT EXISTS job_artifacts (
      id            BIGSERIAL PRIMARY KEY,
      job_id        BIGINT      NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
      kind          TEXT        NOT NULL,
      path          TEXT        NOT NULL,
      metadata_json JSONB       NOT NULL DEFAULT '{}'::jsonb,
      created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_job_artifacts_job_id ON job_artifacts (job_id);
  `);
}

async function recordJobEvent(clientOrPool, { jobId, workerId = null, status = null, message = "", metadata = {} }) {
  await clientOrPool.query(
    `INSERT INTO job_events (job_id, worker_id, status, message, metadata_json)
     VALUES ($1, $2, $3, $4, $5::jsonb)`,
    [jobId, workerId, status, message, JSON.stringify(metadata || {})],
  );
}

function parsePositiveInt(value, fallback) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function normalizeJobRow(row) {
  if (!row) return null;
  return {
    id: Number(row.id),
    type: row.job_type,
    layer: row.layer,
    status: row.status,
    priority: row.priority,
    created_by: row.created_by,
    repo_branch: row.repo_branch,
    command: row.command,
    params_json: row.params_json,
    output_path: row.output_path,
    expected_artifacts: row.expected_artifacts,
    claimed_by: row.claimed_by,
    lease_until: row.lease_until,
    heartbeat_at: row.heartbeat_at,
    created_at: row.created_at,
    started_at: row.started_at,
    completed_at: row.completed_at,
    error_message: row.error_message,
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
// Job routes — Server A control plane
// ---------------------------------------------------------------------------
app.post("/api/jobs", requireAuth, async (req, res) => {
  const type = normalizeString(req.body.type || req.body.job_type).toLowerCase();
  const layer = normalizeString(req.body.layer || "unknown").toLowerCase() || "unknown";
  const priority = Number.isInteger(Number(req.body.priority)) ? Number(req.body.priority) : 100;
  const createdBy = normalizeString(req.body.created_by || req.session?.username || "local-dev");
  const repoBranch = normalizeString(req.body.repo_branch);
  const command = normalizeString(req.body.command);
  const outputPath = normalizeString(req.body.output_path);
  const params = req.body.params_json && typeof req.body.params_json === "object" ? req.body.params_json : {};
  const expectedArtifacts = Array.isArray(req.body.expected_artifacts) ? req.body.expected_artifacts : [];

  if (!["generation", "training"].includes(type)) {
    return res.status(400).json({ ok: false, message: "type must be generation or training." });
  }
  if (!command) {
    return res.status(400).json({ ok: false, message: "command is required for MVP local job tests." });
  }

  try {
    const job = await withTransaction(async (client) => {
      const { rows } = await client.query(
        `INSERT INTO jobs (
           job_type, layer, priority, created_by, repo_branch, command,
           params_json, output_path, expected_artifacts
         )
         VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9::jsonb)
         RETURNING *`,
        [
          type,
          layer,
          priority,
          createdBy,
          repoBranch,
          command,
          JSON.stringify(params),
          outputPath,
          JSON.stringify(expectedArtifacts),
        ],
      );
      await recordJobEvent(client, {
        jobId: rows[0].id,
        status: "queued",
        message: "Job submitted.",
        metadata: { type, layer, repo_branch: repoBranch },
      });
      return rows[0];
    });
    res.status(201).json({ ok: true, job: normalizeJobRow(job) });
  } catch (err) {
    console.error("Create job failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

app.get("/api/jobs", requireAuth, async (req, res) => {
  const status = normalizeString(req.query.status);
  const limit = parsePositiveInt(req.query.limit, 50);
  try {
    const params = [];
    let where = "";
    if (status) {
      params.push(status);
      where = "WHERE status = $1";
    }
    params.push(limit);
    const { rows } = await query(
      `SELECT * FROM jobs
       ${where}
       ORDER BY created_at DESC
       LIMIT $${params.length}`,
      params,
    );
    res.json({ ok: true, jobs: rows.map(normalizeJobRow) });
  } catch (err) {
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

app.get("/api/jobs/:id", requireAuth, async (req, res) => {
  const jobId = parsePositiveInt(req.params.id, 0);
  if (!jobId) return res.status(400).json({ ok: false, message: "Invalid job id." });

  try {
    const jobResult = await query("SELECT * FROM jobs WHERE id = $1", [jobId]);
    if (!jobResult.rows[0]) return res.status(404).json({ ok: false, message: "Job not found." });

    const events = await query(
      `SELECT id, worker_id, status, message, metadata_json, created_at
       FROM job_events
       WHERE job_id = $1
       ORDER BY created_at ASC`,
      [jobId],
    );
    const artifacts = await query(
      `SELECT id, kind, path, metadata_json, created_at
       FROM job_artifacts
       WHERE job_id = $1
       ORDER BY created_at ASC`,
      [jobId],
    );
    res.json({
      ok: true,
      job: normalizeJobRow(jobResult.rows[0]),
      events: events.rows,
      artifacts: artifacts.rows,
    });
  } catch (err) {
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

app.post("/api/jobs/:id/cancel", requireAuth, async (req, res) => {
  const jobId = parsePositiveInt(req.params.id, 0);
  if (!jobId) return res.status(400).json({ ok: false, message: "Invalid job id." });

  try {
    const job = await withTransaction(async (client) => {
      const { rows } = await client.query(
        `UPDATE jobs
         SET status = CASE
             WHEN status IN ('completed', 'failed', 'cancelled') THEN status
             ELSE 'cancel_requested'
           END
         WHERE id = $1
         RETURNING *`,
        [jobId],
      );
      if (!rows[0]) return null;
      await recordJobEvent(client, {
        jobId,
        status: rows[0].status,
        message: "Cancellation requested.",
      });
      return rows[0];
    });
    if (!job) return res.status(404).json({ ok: false, message: "Job not found." });
    res.json({ ok: true, job: normalizeJobRow(job) });
  } catch (err) {
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

// ---------------------------------------------------------------------------
// Worker routes — Server B bridge
// ---------------------------------------------------------------------------
app.post("/api/worker/heartbeat", requireWorkerAuth, async (req, res) => {
  const workerId = normalizeString(req.body.worker_id || "shinypokemon");
  const status = normalizeString(req.body.status || "idle") || "idle";
  const currentJobId = req.body.current_job_id ? parsePositiveInt(req.body.current_job_id, 0) : null;
  const metadata = req.body.metadata_json && typeof req.body.metadata_json === "object" ? req.body.metadata_json : {};

  try {
    await query(
      `INSERT INTO workers (worker_id, status, current_job_id, metadata_json, heartbeat_at, updated_at)
       VALUES ($1, $2, $3, $4::jsonb, NOW(), NOW())
       ON CONFLICT (worker_id)
       DO UPDATE SET
         status = EXCLUDED.status,
         current_job_id = EXCLUDED.current_job_id,
         metadata_json = EXCLUDED.metadata_json,
         heartbeat_at = NOW(),
         updated_at = NOW()`,
      [workerId, status, currentJobId, JSON.stringify(metadata)],
    );
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

app.post("/api/worker/claim", requireWorkerAuth, async (req, res) => {
  const workerId = normalizeString(req.body.worker_id || "shinypokemon");
  const leaseSeconds = parsePositiveInt(req.body.lease_seconds, 120);

  try {
    const job = await withTransaction(async (client) => {
      await client.query(
        `INSERT INTO workers (worker_id, status, heartbeat_at, updated_at)
         VALUES ($1, 'claiming', NOW(), NOW())
         ON CONFLICT (worker_id)
         DO UPDATE SET status = 'claiming', heartbeat_at = NOW(), updated_at = NOW()`,
        [workerId],
      );

      const { rows } = await client.query(
        `UPDATE jobs
         SET status = 'claimed',
             claimed_by = $1,
             lease_until = NOW() + ($2::text || ' seconds')::interval,
             heartbeat_at = NOW()
         WHERE id = (
           SELECT id
           FROM jobs
           WHERE status = 'queued'
           ORDER BY
             CASE job_type WHEN 'generation' THEN 0 ELSE 1 END,
             priority ASC,
             created_at ASC
           FOR UPDATE SKIP LOCKED
           LIMIT 1
         )
         RETURNING *`,
        [workerId, leaseSeconds],
      );

      if (!rows[0]) {
        await client.query(
          `UPDATE workers
           SET status = 'idle', current_job_id = NULL, heartbeat_at = NOW(), updated_at = NOW()
           WHERE worker_id = $1`,
          [workerId],
        );
        return null;
      }

      await client.query(
        `UPDATE workers
         SET status = 'claimed', current_job_id = $2, heartbeat_at = NOW(), updated_at = NOW()
         WHERE worker_id = $1`,
        [workerId, rows[0].id],
      );
      await recordJobEvent(client, {
        jobId: rows[0].id,
        workerId,
        status: "claimed",
        message: "Job claimed by worker.",
        metadata: { lease_seconds: leaseSeconds },
      });
      return rows[0];
    });

    if (!job) return res.json({ ok: true, job: null });
    res.json({ ok: true, job: normalizeJobRow(job), lease_seconds: leaseSeconds });
  } catch (err) {
    console.error("Worker claim failed:", err);
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

app.post("/api/worker/jobs/:id/status", requireWorkerAuth, async (req, res) => {
  const jobId = parsePositiveInt(req.params.id, 0);
  const workerId = normalizeString(req.body.worker_id || "shinypokemon");
  const status = normalizeString(req.body.status).toLowerCase();
  const message = normalizeString(req.body.message);
  const errorMessage = normalizeString(req.body.error_message);
  const metadata = req.body.metadata_json && typeof req.body.metadata_json === "object" ? req.body.metadata_json : {};
  const allowed = new Set(["claimed", "running", "paused", "uploading", "completed", "failed", "cancelled"]);

  if (!jobId) return res.status(400).json({ ok: false, message: "Invalid job id." });
  if (!allowed.has(status)) return res.status(400).json({ ok: false, message: "Invalid job status." });

  try {
    const job = await withTransaction(async (client) => {
      const { rows } = await client.query(
        `UPDATE jobs
         SET status = $2,
             heartbeat_at = NOW(),
             started_at = CASE WHEN $2 = 'running' AND started_at IS NULL THEN NOW() ELSE started_at END,
             completed_at = CASE WHEN $2 IN ('completed', 'failed', 'cancelled') THEN NOW() ELSE completed_at END,
             error_message = CASE WHEN $2 = 'failed' THEN $3 ELSE error_message END
         WHERE id = $1
         RETURNING *`,
        [jobId, status, errorMessage || null],
      );
      if (!rows[0]) return null;

      await client.query(
        `UPDATE workers
         SET status = $2,
             current_job_id = CASE
               WHEN $2 IN ('completed', 'failed', 'cancelled') THEN NULL::bigint
               ELSE $3::bigint
             END,
             heartbeat_at = NOW(),
             updated_at = NOW()
         WHERE worker_id = $1`,
        [workerId, status, jobId],
      );
      await recordJobEvent(client, {
        jobId,
        workerId,
        status,
        message,
        metadata,
      });
      return rows[0];
    });
    if (!job) return res.status(404).json({ ok: false, message: "Job not found." });
    res.json({ ok: true, job: normalizeJobRow(job) });
  } catch (err) {
    res.status(500).json({ ok: false, message: String(err.message || err) });
  }
});

app.post("/api/worker/jobs/:id/artifacts", requireWorkerAuth, async (req, res) => {
  const jobId = parsePositiveInt(req.params.id, 0);
  const artifacts = Array.isArray(req.body.artifacts) ? req.body.artifacts : [];
  const workerId = normalizeString(req.body.worker_id || "shinypokemon");

  if (!jobId) return res.status(400).json({ ok: false, message: "Invalid job id." });
  if (!artifacts.length) return res.status(400).json({ ok: false, message: "artifacts must be a non-empty array." });

  try {
    const inserted = await withTransaction(async (client) => {
      const job = await client.query("SELECT id FROM jobs WHERE id = $1", [jobId]);
      if (!job.rows[0]) return null;

      const rows = [];
      for (const artifact of artifacts) {
        const kind = normalizeString(artifact.kind || "log");
        const path = normalizeString(artifact.path);
        const metadata = artifact.metadata_json && typeof artifact.metadata_json === "object" ? artifact.metadata_json : {};
        if (!path) continue;
        const result = await client.query(
          `INSERT INTO job_artifacts (job_id, kind, path, metadata_json)
           VALUES ($1, $2, $3, $4::jsonb)
           RETURNING id, kind, path, metadata_json, created_at`,
          [jobId, kind, path, JSON.stringify(metadata)],
        );
        rows.push(result.rows[0]);
      }
      await recordJobEvent(client, {
        jobId,
        workerId,
        status: "artifact_registered",
        message: `Registered ${rows.length} artifact(s).`,
      });
      return rows;
    });
    if (inserted === null) return res.status(404).json({ ok: false, message: "Job not found." });
    res.status(201).json({ ok: true, artifacts: inserted });
  } catch (err) {
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
async function start() {
  try {
    await ensureJobTables();
    app.listen(port, "0.0.0.0", () => {
      console.log(`Backend running on http://0.0.0.0:${port}`);
    });
  } catch (err) {
    console.error("Backend startup failed:", err);
    process.exit(1);
  }
}

start();
