import crypto from "node:crypto";
import cors from "cors";
import express from "express";
import pg from "pg";

const app = express();
const port = Number(process.env.PORT || 4000);
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

app.use(cors());
app.use(express.json());

function query(text, params = []) {
  return pool.query(text, params);
}

function scryptPassword(password, salt) {
  return new Promise((resolve, reject) => {
    crypto.scrypt(password, salt, 64, (error, derivedKey) => {
      if (error) {
        reject(error);
        return;
      }

      resolve(derivedKey);
    });
  });
}

async function createPasswordHash(password) {
  const salt = crypto.randomBytes(16).toString("hex");
  const derivedKey = await scryptPassword(password, salt);
  return `${salt}:${derivedKey.toString("hex")}`;
}

async function verifyPassword(password, storedHash) {
  const [salt, expectedHash] = storedHash.split(":");
  if (!salt || !expectedHash) {
    return false;
  }

  const derivedKey = await scryptPassword(password, salt);
  const expectedBuffer = Buffer.from(expectedHash, "hex");

  if (expectedBuffer.length !== derivedKey.length) {
    return false;
  }

  return crypto.timingSafeEqual(expectedBuffer, derivedKey);
}

function normalizeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

async function ensureSchema() {
  await query(`
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      username TEXT NOT NULL UNIQUE,
      email TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
  `);
}

app.get("/api/health", async (_req, res) => {
  try {
    const { rows } = await query("SELECT NOW() AS now");
    res.json({ ok: true, db: "connected", now: rows[0].now });
  } catch (error) {
    res.status(500).json({ ok: false, db: "error", message: String(error.message || error) });
  }
});

app.post("/api/register", async (req, res) => {
  const username = normalizeString(req.body.username);
  const email = normalizeString(req.body.email).toLowerCase();
  const password = typeof req.body.password === "string" ? req.body.password : "";

  if (!username || !email || !password) {
    res.status(400).json({ ok: false, message: "Username, email, and password are required." });
    return;
  }

  try {
    const passwordHash = await createPasswordHash(password);
    const { rows } = await query(
      `INSERT INTO users (username, email, password_hash)
       VALUES ($1, $2, $3)
       RETURNING id, username, email, created_at`,
      [username, email, passwordHash],
    );

    res.status(201).json({ ok: true, user: rows[0] });
  } catch (error) {
    if (error.code === "23505") {
      const message = error.constraint === "users_email_key"
        ? "This email is already registered."
        : "This account name is already taken.";

      res.status(409).json({ ok: false, message });
      return;
    }

    console.error("Register failed:", error);
    res.status(500).json({ ok: false, message: String(error.message || error) });
  }
});

app.post("/api/login", async (req, res) => {
  const account = normalizeString(req.body.account);
  const password = typeof req.body.password === "string" ? req.body.password : "";

  if (!account || !password) {
    res.status(400).json({ ok: false, message: "Account and password are required." });
    return;
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
    if (!user) {
      res.status(401).json({ ok: false, message: "Invalid account or password." });
      return;
    }

    const isValidPassword = await verifyPassword(password, user.password_hash);
    if (!isValidPassword) {
      res.status(401).json({ ok: false, message: "Invalid account or password." });
      return;
    }

    res.json({
      ok: true,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
      },
    });
  } catch (_error) {
    console.error("Login failed:", _error);
    res.status(500).json({ ok: false, message: String(_error.message || _error) });
  }
});

app.get("/", (_req, res) => {
  res.json({ service: "backend", status: "running" });
});

async function startServer() {
  try {
    await ensureSchema();
    app.listen(port, "0.0.0.0", () => {
      console.log(`Backend running on http://0.0.0.0:${port}`);
    });
  } catch (error) {
    console.error("Backend startup failed:", error);
    process.exit(1);
  }
}

startServer();
